"""Pass-level regression tests for linalg.index across layout transforms."""

import itertools
import re
import subprocess
from pathlib import Path

import pytest


TESTDATA = Path(__file__).parent / "testdata" / "layout_index_remapping"


def layout_index_ir(loop_axes):
    source = (TESTDATA / "hwc-index-remapping-bf16.mlir").read_text()
    # Permute the input and output maps of the bias generic consistently.
    output_map = "(" + ", ".join(f"d{axis}" for axis in loop_axes) + ")"
    source = source.replace(
        "affine_map<(d0, d1, d2) -> (d0, d1, d2)>",
        f"affine_map<(d0, d1, d2) -> {output_map}>",
    )
    return source


def convert_layout(torq_compiler, source):
    optimizer = Path(torq_compiler.file_path).parent / "iree-opt"
    proc = subprocess.run(
        [
            str(optimizer),
            "--pass-pipeline=builtin.module(func.func(torq-convert-nhwc-op-to-nchw))",
        ],
        input=source,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def converted_generic(ir, result_type, expected_maps=None, expected_indices=None):
    # Select by converted result type, not the original generic retained for
    # users outside the converted cluster. Inspect before cleanup can erase it.
    matches = re.findall(
        r"linalg.generic \{indexing_maps = \[([^\]]+)\],"
        r"[^{}]*\}[^{}]*\{\s*\^bb0\([^\n]*\):\s*([^{}]*?)"
        r"linalg.yield[^\n]*\n\s*}\s*-> " + re.escape(result_type),
        ir,
        re.S,
    )
    definitions = dict(re.findall(r"^(#\w+) = (affine_map<[^\n]+>)$", ir, re.M))
    candidates = []
    for map_names, body in matches:
        maps = [definitions[name.strip()] for name in map_names.split(",")]
        indices = [int(dim) for dim in re.findall(r"linalg.index (\d+)", body)]
        if expected_maps is not None and maps != expected_maps:
            continue
        if expected_indices is not None and indices != expected_indices:
            continue
        candidates.append((maps, indices))
    assert len(candidates) == 1, ir
    return candidates[0]


@pytest.mark.ci
@pytest.mark.parametrize("loop_axes", list(itertools.permutations(range(3))))
def test_hwc_to_chw_preserves_index_queries(torq_compiler, loop_axes):
    ir = convert_layout(torq_compiler, layout_index_ir(loop_axes))
    assert "linalg.depthwise_conv_2d_nchw_chw" in ir
    # New loops are C,H,W. Locate each original loop among those axes.
    chw_loops = (loop_axes[2], loop_axes[0], loop_axes[1])
    expected_indices = [chw_loops.index(axis) for axis in range(3)]
    identity = "affine_map<(d0, d1, d2) -> (d0, d1, d2)>"
    expected_maps = [identity, identity]
    maps, indices = converted_generic(
        ir, "tensor<5x5x5xbf16>", expected_maps, expected_indices
    )
    assert maps == expected_maps, ir
    assert indices == expected_indices, ir


@pytest.mark.ci
def test_nhwc_to_nchw_preserves_index_queries(torq_compiler):
    source = (TESTDATA / "nhwc-index-remapping-bf16.mlir").read_text()
    ir = convert_layout(torq_compiler, source)
    assert "linalg.depthwise_conv_2d_nchw_chw" in ir
    maps, indices = converted_generic(ir, "tensor<1x7x3x16xbf16>")
    identity = "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>"
    assert maps == [identity, identity], ir
    assert indices == [0, 2, 3, 1], ir
