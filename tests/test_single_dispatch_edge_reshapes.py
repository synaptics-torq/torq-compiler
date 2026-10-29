import re
import subprocess
from pathlib import Path

import pytest

# IR check for a function that only reshapes its input. The reshape must stay outside the
# dispatch as a flow.tensor.reshape, and the dispatch left behind must write in place. If the
# reshape is kept inside the dispatch, the dispatch loads the input and stores a new output,
# which is a full copy on the NPU. The numerics are the same either way, so only the IR shows it.

TESTDATA = Path(__file__).parent / "testdata"

CASES = [
    # onnx.Reshape 1x256x20x20 -> 1x256x400 and nothing else
    TESTDATA / "torch_ops" / "reshape-only-bf16.mlir",
]


@pytest.mark.ci
@pytest.mark.parametrize("model", CASES, ids=lambda p: p.name)
# runtime_hw_type is unused, but requesting it makes the plugin run the test for each chip.
def test_boundary_reshape_is_not_a_copy(torq_compiler, runtime_hw_type, chip_config, model):
    cmd = [
        str(torq_compiler.file_path),
        str(model),
        "-o",
        "-",
        f"--torq-hw={chip_config.data['target']}",
        "--torq-target-host-triple=native",
        "--compile-to=flow",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr[-4000:]}"
    flow = proc.stdout

    assert "flow.tensor.reshape" in flow, "reshape was not kept outside the dispatch:\n" + flow
    # A result that is tied to an operand is printed as a value (%x); a new tensor is printed as
    # its type. Check every result, also when the dispatch has several.
    for line in flow.splitlines():
        call = re.search(r"flow\.dispatch @.*\) -> (.*)$", line)
        if call:
            assert "tensor<" not in call.group(1), "dispatch writes a new tensor:\n" + line
