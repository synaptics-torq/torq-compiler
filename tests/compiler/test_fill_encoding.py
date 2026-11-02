"""Fill lowering must preserve the complete destination tensor type."""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.ci
@pytest.mark.parametrize("shape", ["1x8x112x112", "1x8x?x112"])
@pytest.mark.parametrize("element_type,value", [("i8", -128), ("i16", -1), ("i32", 0)])
def test_fill_preserves_lram_encoding(torq_compiler, shape, element_type, value):
    ty = f"tensor<{shape}x{element_type}, #torq_hl<enc mem_space = lram>>"
    source = f"""
module {{
  func.func @fill(%init: {ty}) -> {ty} {{
    %c = arith.constant {value} : {element_type}
    %r = linalg.fill ins(%c : {element_type}) outs(%init : {ty}) -> {ty}
    return %r : {ty}
  }}
}}
"""
    optimizer = Path(torq_compiler.file_path).parent / "iree-opt"
    proc = subprocess.run(
        [
            str(optimizer),
            "--pass-pipeline=builtin.module(func.func(torq-linalg-to-torqhl-conversion))",
        ],
        input=source, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "torq_hl.fill" in proc.stdout, proc.stdout
    assert "linalg.fill" not in proc.stdout, proc.stdout
    assert ty in proc.stdout, proc.stdout