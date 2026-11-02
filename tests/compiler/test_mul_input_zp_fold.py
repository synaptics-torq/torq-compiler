"""Structure tests of the mul input zero-point fold (--torq-fold-mul-input-zp).

The tosa_ops cases mul-rescale-input-zp-offset-*.mlir check the values. These tests check that
the fold fires: after the LinalgToTorqHL conversion, each input rescale that stays is a
torq_hl.fma, and a mul with an input1 offset has 5 operands.
"""
import re
import subprocess

import pytest

from torq.testing.iree import MODELS_DIR, _find_iree_tool

# case: (input rescales that stay, mul has an input1 offset) with the fold
CASES = {
    # zero points 127 and -128: 127 folds as an offset, -128 does not fit in i8
    "mul-rescale-input-zp-offset-int8": (1, True),
    # zero points -127 and 0: both fold, the mul is one task
    "mul-rescale-input-zp-offset-both-int8": (0, True),
    # zero points -3 and 5: only one input can carry the offset
    "mul-rescale-input-zp-offset-two-nonzero-int8": (1, True),
}


def torqhl_ops(case, fold):
    """The torq_hl ops in the last IR dump after the LinalgToTorqHL conversion."""
    cmd = [_find_iree_tool("TORQ_COMPILE", "torq-compile"),
           str(MODELS_DIR / "tosa_ops" / f"{case}.mlir"), "-o", "/dev/null", "--torq-hw=SL2610",
           f"--torq-fold-mul-input-zp={str(fold).lower()}",
           "--mlir-print-ir-after=torq-linalg-to-torqhl-conversion"]
    dump = subprocess.run(cmd, check=True, capture_output=True, text=True).stderr
    last = dump.split("IR Dump After LinalgToTorqHLConversion")[-1]
    return re.findall(r'"(torq_hl\.\w+)"\(([^)]*)\)', last)


def count_ops(ops):
    rescales = sum(1 for name, _ in ops if name == "torq_hl.fma")
    muls = [operands for name, operands in ops if name == "torq_hl.mul"]
    assert len(muls) == 1
    has_offset = len(muls[0].split(",")) == 5
    return rescales, has_offset


@pytest.mark.parametrize("case", CASES)
def test_mul_input_zp_fold(case):
    assert count_ops(torqhl_ops(case, fold=True)) == CASES[case]
    assert count_ops(torqhl_ops(case, fold=False)) == (2, False)
