import re
import subprocess

import pytest

from torq.testing.iree import MODELS_DIR

# On one slice a torq_hl.segmentation lands directly on its producer conv (sr200 layout). On two
# slices the producer is in an scf.forall over output channels, and
# torq-torqhl-optimize-segmentation folds the segmentation through that loop.
ONE_SLICE_HW = "SR200"
TWO_SLICE_HW = "SL2610"

CONV_RE = re.compile(r'"torq_hl\.(conv2d|depthwise_conv2d)".*?stride = array<i64: ([0-9, ]+)>')


def _segmented_producers(torq_compiler, tmp_path, name, hw):
    """Return the stride of every conv that writes segmented output after the segmentation pass."""
    model = MODELS_DIR / "torch_ops" / f"{name}.mlir"
    cmd = [
        str(torq_compiler.file_path),
        str(model),
        "-o",
        str(tmp_path / "model.vmfb"),
        f"--torq-hw={hw}",
        "--torq-target-host-triple=native",
        "--torq-disable-host",
        "--torq-disable-css",
        "--mlir-print-ir-after=torq-torqhl-optimize-segmentation",
        "--mlir-disable-threading",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    strides = []
    for line in result.stderr.splitlines():
        m = CONV_RE.search(line)
        if m and "segment_output = true" in line:
            strides.append(tuple(int(s) for s in m.group(2).split(",")))
    return strides


@pytest.mark.ci
@pytest.mark.parametrize("hw", [ONE_SLICE_HW, TWO_SLICE_HW], ids=["one_slice", "two_slices"])
@pytest.mark.parametrize(
    "name,stride",
    [
        ("conv2d_bf16_8x8x3x3_s2_into_s2_segment_output_8x8x40", (2, 2)),
        ("depthwise_conv2d_bf16_8x3x3_s2_into_s2_segment_output_8x8x40", (2, 2)),
        ("conv2d_bf16_8x8x3x3_s1_into_s2_segment_output_8x8x40", (1, 1)),
    ],
)
def test_producer_writes_segmented_output(torq_compiler, tmp_path, name, stride, hw):
    """The consumer's segmentation folds into the producer on one and two slices, so the numeric
    fixtures of the same name exercise the segmented write of stride-1 and stride-2 producers."""
    assert stride in _segmented_producers(torq_compiler, tmp_path, name, hw)
