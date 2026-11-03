# Copyright 2026 SYNAPTICS Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


MODEL = """
module {
  func.func @model(%x: tensor<4xbf16>, %y: tensor<4xbf16>)
      -> (tensor<4xbf16>, tensor<4xbf16>) {
    return %y, %x : tensor<4xbf16>, tensor<4xbf16>
  }
}
"""

WRAPPER = """
module {
  func.func private @model(tensor<4xbf16>, tensor<4xbf16>)
      -> (tensor<4xbf16>, tensor<4xbf16>)
  func.func private @prepare(tensor<4xi8>, tensor<4xi8>)
      -> (tensor<4xbf16>, tensor<4xbf16>)
  func.func private @finish(tensor<4xbf16>, tensor<4xbf16>) -> tensor<8xi32>

  func.func @wrapped(%x: tensor<4xi8>, %y: tensor<4xi8>) -> tensor<8xi32> {
    %a, %b = call @prepare(%x, %y) : (tensor<4xi8>, tensor<4xi8>)
        -> (tensor<4xbf16>, tensor<4xbf16>)
    %c, %d = call @model(%a, %b) : (tensor<4xbf16>, tensor<4xbf16>)
        -> (tensor<4xbf16>, tensor<4xbf16>)
    %out = call @finish(%c, %d) : (tensor<4xbf16>, tensor<4xbf16>) -> tensor<8xi32>
    return %out : tensor<8xi32>
  }
}
"""

LINALG_WRAPPER = """
module {
    func.func private @model(tensor<4xi16>) -> tensor<4xi16>

    func.func @wrapped(%input: tensor<4xi16>) -> tensor<4xi16> {
        %pre_empty = tensor.empty() : tensor<4xi16>
        %prepared = linalg.generic {
                indexing_maps = [affine_map<(index) -> (index)>,
                                                 affine_map<(index) -> (index)>],
                iterator_types = ["parallel"]}
                ins(%input : tensor<4xi16>) outs(%pre_empty : tensor<4xi16>) {
            ^bb0(%value: i16, %init: i16):
                %one = arith.constant 1 : i16
                %incremented = arith.addi %value, %one : i16
                linalg.yield %incremented : i16
        } -> tensor<4xi16>

        %result = call @model(%prepared) : (tensor<4xi16>) -> tensor<4xi16>

        %post_empty = tensor.empty() : tensor<4xi16>
        %output = linalg.generic {
                indexing_maps = [affine_map<(index) -> (index)>,
                                                 affine_map<(index) -> (index)>],
                iterator_types = ["parallel"]}
                ins(%result : tensor<4xi16>) outs(%post_empty : tensor<4xi16>) {
            ^bb0(%value: i16, %init: i16):
                %two = arith.constant 2 : i16
                %doubled = arith.muli %value, %two : i16
                linalg.yield %doubled : i16
        } -> tensor<4xi16>
        return %output : tensor<4xi16>
    }
}
"""


@pytest.fixture
def compiler_path(request):
    if path := os.environ.get("TORQ_COMPILE"):
        return path
    return str(request.getfixturevalue("torq_compiler").file_path)


@pytest.fixture
def runtime_path(request):
    if path := os.environ.get("TORQ_RUN_MODULE"):
        return path
    return str(request.getfixturevalue("torq_runtime").file_path)


def lower_css_external(compiler_path, source):
    optimizer = os.environ.get("IREE_OPT", str(Path(compiler_path).parent / "iree-opt"))
    return subprocess.run(
        [optimizer, "--torq-lower-css-external-functions"],
        input=source, capture_output=True, text=True,
    )


@pytest.mark.ci
@pytest.mark.parametrize("declaration", ["", "llvm.func @hook(!llvm.ptr, !llvm.ptr)"])
def test_lower_css_external_pointer_abi(compiler_path, declaration):
    result = lower_css_external(compiler_path, f"""
module {{
  {declaration}
  func.func @invoke(%input: memref<4xi16>, %output: memref<4xi32>) {{
    "torq_hl.external_call"(%input, %output) <{{function_name = "hook"}}>
        : (memref<4xi16>, memref<4xi32>) -> ()
    return
  }}
}}
""")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("memref.extract_aligned_pointer_as_index") == 2
    assert result.stdout.count("arith.index_cast") == 2
    assert result.stdout.count("llvm.inttoptr") == 2
    assert result.stdout.count("llvm.func @hook(!llvm.ptr, !llvm.ptr)") == 1
    assert "hal.import.static" in result.stdout
    assert re.search(r"llvm.call @hook\([^\n]+ : \(!llvm.ptr, !llvm.ptr\) -> \(\)", result.stdout)
    assert "torq_hl.external_call" not in result.stdout


@pytest.mark.ci
@pytest.mark.parametrize("declaration, input_type, result_type, diagnostic", [
    ("llvm.func @hook(!llvm.ptr)", "memref<4xi16>", "()",
     "conflicting external CSS function signature"),
    ("func.func private @hook(memref<4xi16>, memref<4xi32>)", "memref<4xi16>", "()",
     "conflicting external CSS function signature"),
    ("", "memref<4xi16, strided<[2], offset: 0>>", "()", "require dense memrefs"),
    ("", "memref<4xi4>", "()", "byte-addressable elements"),
    ("", "i32", "()", "require dense memrefs"),
    ("", "tensor<4xi16>", "tensor<4xi32>", "must be bufferized"),
])
def test_lower_css_external_diagnostics(compiler_path, declaration, input_type, result_type, diagnostic):
    assignment = "%result = " if result_type != "()" else ""
    result = lower_css_external(compiler_path, f"""
module {{
  {declaration}
  func.func @invoke(%input: {input_type}, %output: memref<4xi32>) {{
    {assignment}"torq_hl.external_call"(%input, %output) <{{function_name = "hook"}}>
        : ({input_type}, memref<4xi32>) -> {result_type}
    return
  }}
}}
""")
    assert result.returncode != 0
    assert diagnostic in result.stderr


def compile_wrapper(compiler_path, tmp_path, wrapper=WRAPPER, phase="input", flags=(), model=MODEL):
    model_path = tmp_path / "model.mlir"
    model_path.write_text(model)
    wrapper_path = tmp_path / "wrapper.mlir"
    if wrapper is not None:
        wrapper_path.write_text(wrapper)
    return subprocess.run(
        [
            compiler_path, str(model_path),
            f"--torq-wrapper={wrapper_path}",
            "--iree-input-type=linalg-torq", f"--compile-to={phase}", "-o",
            str(tmp_path / "model.vmfb") if phase == "end" else "-",
            "--mlir-print-ir-after=torq-apply-wrapper",
            *flags,
        ],
        capture_output=True, text=True,
    )


@pytest.mark.ci
@pytest.mark.parametrize("phase", ["input", "abi"])
def test_wrapper_multi_io(compiler_path, tmp_path, phase):
    result = compile_wrapper(compiler_path, tmp_path, phase=phase)
    assert result.returncode == 0, result.stderr
    ir = result.stderr
    assert "func.func @wrapped(" in ir
    assert "-> tensor<8xi32>" in ir
    assert "func.func private @model(" in ir
    assert "call @model" not in ir
    assert 'torq_hl.external_call "prepare"' in ir
    assert 'torq_hl.external_call "finish"' in ir
    assert ir.count('torq_hl.external_call') == 2
    assert "tensor<4xi8>, tensor<4xi8>" in ir
    assert "tensor<4xbf16>, tensor<4xbf16>" in ir
    prepared = re.search(r'(%\w+):2 = torq_hl\.external_call "prepare"', ir)
    assert prepared is not None
    assert f'torq_hl.external_call "finish"({prepared[1]}#1, {prepared[1]}#0)' in ir


@pytest.mark.ci
@pytest.mark.parametrize("multiblock", [False, True])
def test_wrapper_helper(compiler_path, tmp_path, multiblock):
    helper = """
  func.func private @helper(%x: tensor<4xbf16>, %y: tensor<4xbf16>)
      -> (tensor<4xbf16>, tensor<4xbf16>) {
    %a, %b = call @model(%x, %y) : (tensor<4xbf16>, tensor<4xbf16>)
        -> (tensor<4xbf16>, tensor<4xbf16>)
    return %a, %b : tensor<4xbf16>, tensor<4xbf16>
  }
"""
    if multiblock:
        helper = helper.replace("    %a, %b =", "    cf.br ^body\n  ^body:\n    %a, %b =")
    wrapper = WRAPPER.replace("call @model", "call @helper").replace(
        "  func.func @wrapped", helper + "  func.func @wrapped"
    )
    result = compile_wrapper(compiler_path, tmp_path, wrapper)
    assert result.returncode == 0, result.stderr
    assert "call @helper" not in result.stderr
    assert "call @model" not in result.stderr


@pytest.mark.ci
def test_wrapper_precedes_dispatch_creation(compiler_path, tmp_path):
    computation = """
    %empty = tensor.empty() : tensor<4xbf16>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(index) -> (index)>,
                         affine_map<(index) -> (index)>,
                         affine_map<(index) -> (index)>],
        iterator_types = ["parallel"]}
        ins(%x, %y : tensor<4xbf16>, tensor<4xbf16>)
        outs(%empty : tensor<4xbf16>) {
      ^bb0(%left: bf16, %right: bf16, %init: bf16):
        %value = arith.addf %left, %right : bf16
        linalg.yield %value : bf16
    } -> tensor<4xbf16>
    return %sum, %x
"""
    model = MODEL.replace("    return %y, %x", computation.rstrip())
    result = compile_wrapper(compiler_path, tmp_path, model=model)
    assert result.returncode == 0, result.stderr
    assert "linalg.generic" in result.stderr
    assert "flow.dispatch.region" not in result.stderr
    assert "flow.dispatch.region" in result.stdout


@pytest.mark.ci
@pytest.mark.parametrize(
    "wrapper, diagnostic",
    [
        (None, "could not load torq wrapper"),
        (WRAPPER.replace("bf16", "f32"), "does not match model signature"),
        (WRAPPER.replace("@wrapped", "@model"), "redefinition of symbol"),
        (WRAPPER.replace("func.func @wrapped", "func.func private @wrapped"),
         "requires a public definition"),
        (WRAPPER.replace("@model", "@missing"), "declaration of a model function"),
        (WRAPPER.replace("    return %out", "    %recursive = call @wrapped(%x, %y) : "
                 "(tensor<4xi8>, tensor<4xi8>) -> tensor<8xi32>\n    return %out"),
         "recursive calls are not supported"),
        (WRAPPER.replace("func.func private @model(tensor<4xbf16>, tensor<4xbf16>)\n"
                         "      -> (tensor<4xbf16>, tensor<4xbf16>)", ""),
         "does not reference a valid function"),
    ],
)
def test_wrapper_diagnostics(compiler_path, tmp_path, wrapper, diagnostic):
    result = compile_wrapper(compiler_path, tmp_path, wrapper)
    assert result.returncode != 0
    assert diagnostic in result.stderr


@pytest.mark.ci
@pytest.mark.parametrize("target, march, abi", [
    ("SL2610", "rv32im", "ilp32"),
    ("SR200", "rv32imf_zve32x", "ilp32f"),
])
def test_css_target_flags(compiler_path, target, march, abi):
    result = subprocess.run(
        [compiler_path, f"--torq-print-css-target-flags={target}"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert shlex.split(result.stdout) == [
        "--target=riscv32-unknown-elf", f"-march={march}", f"-mabi={abi}",
    ]


@pytest.mark.ci
def test_css_target_flags_unknown_platform(compiler_path):
    result = subprocess.run(
        [compiler_path, "--torq-print-css-target-flags=unknown"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "Unknown Torq platform 'unknown'" in result.stderr
    assert "--torq-hw=list" in result.stderr


@pytest.fixture(params=["SL2610", "SR200"])
def css_library(tmp_path, request, compiler_path):
    target = request.param
    query = subprocess.run(
        [compiler_path, f"--torq-print-css-target-flags={target}"],
        capture_output=True, text=True,
    )
    assert query.returncode == 0, query.stderr
    target_flags = shlex.split(query.stdout)
    clang = shutil.which("clang")
    archiver = shutil.which("llvm-ar")
    assert clang
    assert archiver
    assert shutil.which("llvm-nm"), "CSS library tests require llvm-nm"
    source = tmp_path / "external.c"
    source.write_text("""
void prepare(const signed char *x, const signed char *y,
                         unsigned short *a, unsigned short *b) {
    for (int index = 0; index < 4; ++index) {
        a[index] = x[index];
        b[index] = y[index];
    }
}
void finish(const unsigned short *a, const unsigned short *b, int *out) {
    for (int index = 0; index < 4; ++index) {
        out[index] = a[index];
        out[index + 4] = b[index];
    }
}
extern void unavailable(void);
void unused_same_member(void) { unavailable(); }
""")
    unused = tmp_path / "unused.c"
    unused.write_text("extern void unavailable(void);\n"
                      "void unused_archive_member(void) { unavailable(); }\n")
    objects = []
    for path in (source, unused):
        object_path = path.with_suffix(".o")
        subprocess.run(
            [clang, *target_flags, "-ffreestanding", "-ffunction-sections", "-fdata-sections",
             "-O2", "-c", str(path), "-o", str(object_path)],
            check=True, capture_output=True, text=True,
        )
        objects.append(str(object_path))
    archive = tmp_path / "libexternal.a"
    subprocess.run([archiver, "rcs", str(archive), *objects], check=True)
    return archive, target


@pytest.mark.ci
@pytest.mark.parametrize("strided", [False, True])
def test_wrapper_css_library(compiler_path, tmp_path, css_library, strided):
    archive, target = css_library
    symbols = tmp_path / "symbols"
    wrapper = WRAPPER
    if strided:
        wrapper = wrapper.replace(
            "@wrapped(%x: tensor<4xi8>, %y: tensor<4xi8>)",
            "@wrapped(%x: tensor<8xi8>, %y: tensor<8xi8>)",
        ).replace(
            "    %a, %b = call @prepare(%x, %y)",
            "    %xs = tensor.extract_slice %x[1] [4] [2] : tensor<8xi8> to tensor<4xi8>\n"
            "    %ys = tensor.extract_slice %y[0] [4] [2] : tensor<8xi8> to tensor<4xi8>\n"
            "    %a, %b = call @prepare(%xs, %ys)",
        )
    result = compile_wrapper(
        compiler_path, tmp_path, wrapper, phase="end",
        flags=(f"--torq-css-library={archive}", f"--torq-hw={target}",
               f"--torq-create-css-symbols={symbols}",
               "--mlir-print-ir-after=iree-convert-to-llvm"),
    )
    assert result.returncode == 0, result.stderr
    assert "llvm.call @prepare(" in result.stderr
    assert "llvm.call @finish(" in result.stderr
    assert "llvm.func @prepare(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)" in result.stderr
    assert "llvm.func @finish(!llvm.ptr, !llvm.ptr, !llvm.ptr)" in result.stderr
    assert "__import_ordinal_prepare" not in result.stderr
    executables = list(symbols.rglob("*.elf"))
    assert len(executables) == 2, result.stderr
    nm = shutil.which("llvm-nm")
    assert nm
    table = "\n".join(subprocess.check_output([nm, str(path)], text=True)
                      for path in executables)
    assert " prepare" in table
    assert " finish" in table
    assert "unused_" not in table
    assert "unavailable" not in table


@pytest.mark.ci
@pytest.mark.parametrize(
    "flags, diagnostic",
    [
        (("--torq-disable-css",), "external functions require CSS execution"),
        (("--torq-css-library=/nonexistent/libexternal.a",), "could not read CSS library"),
        ((), "undefined symbol: prepare"),
    ],
)
def test_css_library_diagnostics(compiler_path, tmp_path, flags, diagnostic):
    result = compile_wrapper(compiler_path, tmp_path, phase="end", flags=flags)
    assert result.returncode != 0
    assert diagnostic in result.stderr


@pytest.mark.ci
def test_wrapper_css_library_cmodel(compiler_path, runtime_path, tmp_path, css_library):
    archive, target = css_library
    model = """
module {
  func.func @model(%x: tensor<4xi16>, %y: tensor<4xi16>)
      -> (tensor<4xi16>, tensor<4xi16>) {
    %empty = tensor.empty() : tensor<4xi16>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(index) -> (index)>,
                         affine_map<(index) -> (index)>,
                         affine_map<(index) -> (index)>],
        iterator_types = ["parallel"]}
        ins(%x, %y : tensor<4xi16>, tensor<4xi16>)
        outs(%empty : tensor<4xi16>) {
      ^bb0(%left: i16, %right: i16, %init: i16):
        %value = arith.addi %left, %right : i16
        linalg.yield %value : i16
    } -> tensor<4xi16>
    return %sum, %x : tensor<4xi16>, tensor<4xi16>
  }
}
"""
    result = compile_wrapper(
        compiler_path, tmp_path, WRAPPER.replace("bf16", "i16"), phase="end", model=model,
        flags=(f"--torq-css-library={archive}", f"--torq-hw={target}",
               "--torq-target-host-triple=native"),
    )
    assert result.returncode == 0, result.stderr
    left = np.array([1, 3, 5, 7], dtype=np.int8)
    right = np.array([2, 4, 6, 8], dtype=np.int8)
    for name, value in (("left", left), ("right", right)):
        np.save(tmp_path / f"{name}.npy", value)
    output = tmp_path / "output.npy"
    run = subprocess.run(
        [runtime_path, f"--module={tmp_path / 'model.vmfb'}", "--function=wrapped",
         "--torq_hw_type=sim", f"--input=@{tmp_path / 'left.npy'}",
         f"--input=@{tmp_path / 'right.npy'}", f"--output=@{output}"],
        capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    expected = np.concatenate((left.astype(np.int32) + right, left.astype(np.int32)))
    actual = np.load(output)
    assert actual.dtype == np.int32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.ci
@pytest.mark.parametrize("target", ["SL2610", "SR200"])
def test_linalg_wrapper_cmodel(compiler_path, runtime_path, tmp_path, target):
        model = """
module {
    func.func @model(%input: tensor<4xi16>) -> tensor<4xi16> {
        %empty = tensor.empty() : tensor<4xi16>
        %square = linalg.generic {
                indexing_maps = [affine_map<(index) -> (index)>,
                                                 affine_map<(index) -> (index)>],
                iterator_types = ["parallel"]}
                ins(%input : tensor<4xi16>) outs(%empty : tensor<4xi16>) {
            ^bb0(%value: i16, %init: i16):
                %squared = arith.muli %value, %value : i16
                linalg.yield %squared : i16
        } -> tensor<4xi16>
        return %square : tensor<4xi16>
    }
}
"""
        result = compile_wrapper(
                compiler_path, tmp_path, LINALG_WRAPPER, phase="end", model=model,
                flags=(f"--torq-hw={target}", "--torq-target-host-triple=native",
                             "--torq-disable-css", "--torq-disable-host"),
        )
        assert result.returncode == 0, result.stderr
        assert "torq_hl.external_call" not in result.stderr
        assert "call @model" not in result.stderr
        values = np.array([-7, -1, 0, 5], dtype=np.int16)
        input_path = tmp_path / "input.npy"
        output_path = tmp_path / "output.npy"
        np.save(input_path, values)
        run = subprocess.run(
                [runtime_path, f"--module={tmp_path / 'model.vmfb'}", "--function=wrapped",
                 "--torq_hw_type=sim", f"--input=@{input_path}", f"--output=@{output_path}"],
                capture_output=True, text=True, timeout=60,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        actual = np.load(output_path)
        assert actual.dtype == np.int16
        np.testing.assert_array_equal(actual, 2 * (values + 1) ** 2)


@pytest.mark.ci
def test_wrapper_drops_unwrapped_exports(compiler_path, tmp_path):
        model = MODEL.replace("module {", """module {
    func.func @unwrapped(%input: tensor<4xbf16>) -> tensor<4xbf16> {
        return %input : tensor<4xbf16>
    }
""", 1)
        result = compile_wrapper(compiler_path, tmp_path, model=model, phase="abi")
        assert result.returncode == 0, result.stderr
        assert "func.func private @unwrapped(" in result.stderr
        assert "func.func @unwrapped(" not in result.stderr


@pytest.mark.ci
def test_external_call_assembly_roundtrip(compiler_path):
        optimizer = os.environ.get("IREE_OPT", str(Path(compiler_path).parent / "iree-opt"))
        source = """
module {
    func.func @invoke(%input: tensor<4xi16>) -> tensor<4xi32> {
        %result = torq_hl.external_call "hook"(%input)
                : (tensor<4xi16>) -> tensor<4xi32>
        return %result : tensor<4xi32>
    }
}
"""
        first = subprocess.run([optimizer], input=source, capture_output=True, text=True)
        assert first.returncode == 0, first.stderr
        assert 'torq_hl.external_call "hook"(' in first.stdout
        second = subprocess.run([optimizer], input=first.stdout, capture_output=True, text=True)
        assert second.returncode == 0, second.stderr
        assert second.stdout == first.stdout
