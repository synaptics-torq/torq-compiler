import re
import subprocess
from pathlib import Path

import pytest

# IR checks for the i16 narrowing of a squared add or sub in torq-optimize-linalg-for-torq:
#   %a = rescale(%a8)   %b = rescale(%b8)   %d = %a - %b (or %a + %b)   %m = %d * %d
# The NPU has no i32 x i32 mul. When |%a| + |%b| fits in i16, the mul must read i16. A mul
# whose range does not fit, or that is not a square, must stay i32 and the IR must stay valid.
# The Keras case squared_difference_mean_1x1x238 and tosa_ops/squared-sum-i8.mlir check the values.

# Two i8 -> i32 rescales with scale 2^(30 - SHIFT), their add or sub, and a mul of the result.
TEMPLATE = """
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(%a8: tensor<1x1x238xi8>, %b8: tensor<1x1x238xi8>) -> tensor<1x1x238xi32> {
    %zp = arith.constant -15 : i32
    %m = arith.constant 1073741824 : i32
    %s = arith.constant SHIFT : i8
    %c3 = arith.constant 3 : i32
    %e = tensor.empty() : tensor<1x1x238xi32>
    %a = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%a8 : tensor<1x1x238xi8>) outs(%e : tensor<1x1x238xi32>) {
    ^bb0(%in: i8, %out: i32):
      %0 = arith.extsi %in : i8 to i32
      %1 = arith.subi %0, %zp : i32
      %2 = tosa.apply_scale %1, %m, %s rounding_mode<SINGLE_ROUND> : (i32, i32, i8) -> i32
      linalg.yield %2 : i32
    } -> tensor<1x1x238xi32>
    %b = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%b8 : tensor<1x1x238xi8>) outs(%e : tensor<1x1x238xi32>) {
    ^bb0(%in: i8, %out: i32):
      %0 = arith.extsi %in : i8 to i32
      %1 = arith.subi %0, %zp : i32
      %2 = tosa.apply_scale %1, %m, %s rounding_mode<SINGLE_ROUND> : (i32, i32, i8) -> i32
      linalg.yield %2 : i32
    } -> tensor<1x1x238xi32>
    %d = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%a, %b : tensor<1x1x238xi32>, tensor<1x1x238xi32>) outs(%e : tensor<1x1x238xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %0 = arith.ADDSUB %in, %in_0 : i32
      linalg.yield %0 : i32
    } -> tensor<1x1x238xi32>
MUL
    return %r : tensor<1x1x238xi32>
  }
}
"""

# %d * %d with %d as one input
SQUARE_ONE_INPUT = """
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%d : tensor<1x1x238xi32>) outs(%e : tensor<1x1x238xi32>) {
    ^bb0(%in: i32, %out: i32):
      %0 = arith.muli %in, %in : i32
      linalg.yield %0 : i32
    } -> tensor<1x1x238xi32>"""

# %d * %d with %d as two inputs, so %d has two uses
SQUARE_TWO_INPUTS = """
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%d, %d : tensor<1x1x238xi32>, tensor<1x1x238xi32>) outs(%e : tensor<1x1x238xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %0 = arith.muli %in, %in_0 : i32
      linalg.yield %0 : i32
    } -> tensor<1x1x238xi32>"""

# %d * 3: not a square
TIMES_CONST = """
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%d : tensor<1x1x238xi32>) outs(%e : tensor<1x1x238xi32>) {
    ^bb0(%in: i32, %out: i32):
      %0 = arith.muli %in, %c3 : i32
      linalg.yield %0 : i32
    } -> tensor<1x1x238xi32>"""

# case: (add or sub, mul, shift, mul reads i16)
# |%a8 - zp| is at most 142. With shift 30 each rescale is at most 143 in magnitude (1 for the
# rounding), so |%a| + |%b| fits in i16. With shift 22 each rescale can reach 142 * 256 + 1,
# so it does not.
CASES = {
    "sub": ("subi", SQUARE_ONE_INPUT, 30, True),
    "add": ("addi", SQUARE_ONE_INPUT, 30, True),
    "sub-two-inputs": ("subi", SQUARE_TWO_INPUTS, 30, True),
    "sub-too-wide": ("subi", SQUARE_ONE_INPUT, 22, False),
    "sub-times-const": ("subi", TIMES_CONST, 30, False),
}


@pytest.mark.ci
@pytest.mark.parametrize("case", CASES)
def test_squared_add_sub_mul_type(torq_compiler, tmp_path, case):
    add_sub, mul, shift, narrowed = CASES[case]
    model = tmp_path / "model.mlir"
    model.write_text(
        TEMPLATE.replace("ADDSUB", add_sub).replace("MUL", mul).replace("SHIFT", str(shift))
    )
    iree_opt = Path(torq_compiler.file_path).parent / "iree-opt"
    proc = subprocess.run(
        [
            str(iree_opt),
            str(model),
            "--torq-hw=SL2610",
            "--pass-pipeline=builtin.module(func.func(torq-optimize-linalg-for-torq))",
        ],
        capture_output=True,
        text=True,
    )
    # iree-opt verifies the IR after the pass, so a mul rebuilt with mixed types fails here
    assert proc.returncode == 0, proc.stderr

    mul_types = re.findall(r"arith\.muli [^:]+: (i\d+)", proc.stdout)
    assert mul_types == (["i16"] if narrowed else ["i32"]), proc.stdout
