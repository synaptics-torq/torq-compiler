import re
import subprocess
from pathlib import Path

import pytest


def pooling_ir(y_expr, x_expr, width=56):
    """A two-row tile of the 56x56, exclude-padding integer avgpool."""
    source = (
        Path(__file__).parent / "testdata" / "sliding_avgpool" / "sliding-avgpool-affine-i8.mlir"
    ).read_text()
    source = source.replace("tensor<1x4x58x11xi8>", f"tensor<1x4x{width + 2}x11xi8>")
    source = source.replace("tensor<1x2x56x11xi32>", f"tensor<1x2x{width}x11xi32>")
    source = source.replace("tensor<1x2x56x11xi8>", f"tensor<1x2x{width}x11xi8>")
    source = source.replace("%gy = affine.apply affine_map<()[s0] -> (s0)>()[%y]", y_expr)
    source = source.replace("%gx = affine.apply affine_map<()[s0] -> (s0)>()[%x]", x_expr)
    if "%offset" in y_expr or "%offset" in x_expr:
        # Only the standalone unknown-symbol case needs a runtime index input.
        source = source.replace(
            f"%input: tensor<1x4x{width + 2}x11xi8>)",
            f"%input: tensor<1x4x{width + 2}x11xi8>, %offset: index)",
        )
    return source


def convert(torq_compiler, source, pass_name):
    optimizer = Path(torq_compiler.file_path).parent / "iree-opt"
    proc = subprocess.run(
        [str(optimizer), f"--pass-pipeline=builtin.module(func.func({pass_name}))"],
        input=source, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.ci
@pytest.mark.parametrize("pass_name", [
    "torq-linalg-to-torqhl-pre-conversion",
    "torq-linalg-to-torqhl-pre-conversion,torq-linalg-to-torqhl-conversion",
])
@pytest.mark.parametrize("y_expr,rows", [
    ("affine.apply affine_map<()[s0] -> (s0 + 28)>()[%y]", [28, 29]),
    ("affine.apply affine_map<()[s0] -> (s0)>()[%y]", [0, 1]),
    ("affine.apply affine_map<()[s0] -> (s0 + 54)>()[%y]", [54, 55]),
    ("affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%y)[%c54]", [54, 55]),
    ("affine.apply affine_map<()[s0] -> (((s0 - 1) floordiv 2) + 1)>()[%y]", [0, 1]),
    ("affine.apply affine_map<()[s0] -> (((s0 - 1) ceildiv 2) + 1)>()[%y]", [1, 1]),
    ("affine.apply affine_map<()[s0] -> ((s0 - 1) mod 2)>()[%y]", [1, 0]),
    ("affine.min affine_map<()[s0] -> (s0 + 54, 54)>()[%y]", [54, 54]),
    ("affine.max affine_map<()[s0] -> (s0, 1)>()[%y]", [1, 1]),
])
@pytest.mark.parametrize("width,x_offset", [(56, 0), (2, 54), (2, 28)])
def test_affine_pool_scale_preserves_global_coordinates(
    torq_compiler, pass_name, y_expr, rows, width, x_offset,
):
    x_expr = f"%gx = affine.apply affine_map<()[s0] -> (s0 + {x_offset})>()[%x]"
    source = pooling_ir(f"%gy = {y_expr}", x_expr, width)
    assert "linalg.pooling_nhwc_sum" in source and "tosa.apply_scale" in source
    output = convert(torq_compiler, source, pass_name)
    assert output.count('"torq_hl.avg_pool2d"') == 1, output
    assert "linalg.pooling_nhwc_sum" not in output and "tosa.apply_scale" not in output

    # Check exact fixed-point scales, not merely whether fusion fired. In
    # particular, the middle tile's local row 0 is NOT an image border.
    packed = re.search(r"dense<\[([^\]]+)\]> : tensor<\d+xi32>", output)
    assert packed, output
    scales = [int(v) for v in packed.group(1).split(",")]
    regions_match = re.search(r"scale_regions = array<i64: ([^>]+)>", output)
    if regions_match:
        regions = [int(v) for v in regions_match.group(1).split(",")]
    else:
        shift = int(re.search(r"shift_factor = (\d+) : i32", output).group(1))
        regions = [0, 0, 2, width, shift]
    assert len(scales) == 2 * (len(regions) // 5)
    actual = {}
    for i in range(0, len(regions), 5):
        y, x, height, region_width, shift = regions[i:i + 5]
        bias, multiplier = scales[2 * (i // 5):2 * (i // 5) + 2]
        assert bias == 0
        for yy in range(y, y + height):
            for xx in range(x, x + region_width):
                assert (yy, xx) not in actual
                actual[yy, xx] = (multiplier, shift)
    expected = {}
    for y, global_y in enumerate(rows):
        for x in range(width):
            global_x = x + x_offset
            count = (2 if global_y in (0, 55) else 3) * (2 if global_x in (0, 55) else 3)
            log = (count - 1).bit_length()
            expected[y, x] = ((1073741825 << log) // count, 30 + log)
    assert actual == expected


@pytest.mark.ci
def test_affine_pool_scale_unknown_symbol_stays_unfused(torq_compiler):
    source = pooling_ir(
        "%gy = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%y, %offset]",
        "%gx = affine.apply affine_map<()[s0] -> (s0)>()[%x]",
    )
    output = convert(torq_compiler, source, "torq-linalg-to-torqhl-pre-conversion")
    assert '"torq_hl.avg_pool2d"' not in output
    assert "linalg.pooling_nhwc_sum" in output and "tosa.apply_scale" in output


@pytest.mark.ci
@pytest.mark.parametrize("pass_name", [
    "torq-linalg-to-torqhl-pre-conversion",
    "torq-linalg-to-torqhl-pre-conversion,torq-linalg-to-torqhl-conversion",
])
def test_sliding_pool_accepts_zero_shift(torq_compiler, pass_name):
    source = pooling_ir(
        "%gy = affine.apply affine_map<()[s0] -> (s0)>()[%y]",
        "%gx = affine.apply affine_map<()[s0] -> (s0)>()[%x]",
    ).replace(
        "%shift = arith.addi %log8, %c30_i8 : i8", "%shift = arith.constant 0 : i8"
    )
    output = convert(torq_compiler, source, pass_name)
    assert output.count('"torq_hl.avg_pool2d"') == 1, output
    assert "shift_factor = 0 : i32" in output
    assert "linalg.pooling_nhwc_sum" not in output and "tosa.apply_scale" not in output


@pytest.mark.ci
@pytest.mark.parametrize(
    "row_start,row_end,step,expect_fused",
    # Step 4 reaches rows 48 and 52. The old bound includes row 55,
    # where the padding changes the scale.
    [(17, 19, 1, True), (0, 2, 1, False), (48, 56, 4, True)],
)
def test_sliding_pool_scale_across_scf_loop(
    torq_compiler, row_start, row_end, step, expect_fused
):
    source = (
        Path(__file__).parent
        / "testdata"
        / "sliding_avgpool"
        / "sliding-avgpool-loop-invariant.mlir"
    ).read_text()
    source = source.replace(
        "%c17 = arith.constant 17 : index", f"%c17 = arith.constant {row_start} : index"
    )
    source = source.replace(
        "%c19 = arith.constant 19 : index", f"%c19 = arith.constant {row_end} : index"
    )
    source = source.replace(
        "%step = arith.constant 1 : index", f"%step = arith.constant {step} : index"
    )
    output = convert(
        torq_compiler,
        source,
        "torq-linalg-to-torqhl-pre-conversion,torq-linalg-to-torqhl-conversion",
    )
    if expect_fused:
        assert output.count('"torq_hl.avg_pool2d"') == 1, output
        assert "linalg.pooling_nhwc_sum" not in output and "tosa.apply_scale" not in output
        # Both output rows use a full 3x3 window on every loop iteration.
        # Check the bias, multiplier, and shift for division by nine.
        packed = re.search(r"dense<\[([^\]]+)\]> : tensor<2xi32>", output)
        assert packed, output
        assert [int(value) for value in packed.group(1).split(",")] == [
            0, (1073741825 << 4) // 9,
        ], output
        assert "shift_factor = 34 : i32" in output, output
        assert "scale_regions" not in output, output
    else:
        assert '"torq_hl.avg_pool2d"' not in output
        assert "linalg.pooling_nhwc_sum" in output and "tosa.apply_scale" in output
