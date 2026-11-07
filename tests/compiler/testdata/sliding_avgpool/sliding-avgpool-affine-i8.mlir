// Lowered two-row tile used only by the affine pooling pass-level tests.
// Keep outside tosa_ops: this intentionally mixes Linalg and scalar TOSA IR.
func.func @pool(%input: tensor<1x4x58x11xi8>) -> tensor<1x2x56x11xi8> {
  %zero = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c54 = arith.constant 54 : index
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %base = arith.constant 1073741825 : i64
  %c30_i8 = arith.constant 30 : i8
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %kernel = tensor.empty() : tensor<3x3xi32>
  %empty = tensor.empty() : tensor<1x2x56x11xi32>
  %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<1x2x56x11xi32>) -> tensor<1x2x56x11xi32>
  %sum = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%input, %kernel : tensor<1x4x58x11xi8>, tensor<3x3xi32>) outs(%init : tensor<1x2x56x11xi32>) -> tensor<1x2x56x11xi32>
  %output = tensor.empty() : tensor<1x2x56x11xi8>
  %result = linalg.generic {
    indexing_maps = [affine_map<(n, h, w, c) -> (n, h, w, c)>, affine_map<(n, h, w, c) -> (n, h, w, c)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%sum : tensor<1x2x56x11xi32>) outs(%output : tensor<1x2x56x11xi8>) {
  ^bb0(%in: i32, %out: i8):
    %y = linalg.index 1 : index
    %x = linalg.index 2 : index
    %gy = affine.apply affine_map<()[s0] -> (s0)>()[%y]
    %gx = affine.apply affine_map<()[s0] -> (s0)>()[%x]
    %ym1 = arith.subi %gy, %c1 : index
    %yl = arith.minsi %ym1, %c0 : index
    %ybase = arith.addi %yl, %c3 : index
    %yend = arith.subi %c54, %gy : index
    %yr = arith.minsi %yend, %c0 : index
    %ysum = arith.addi %ybase, %yr : index
    %yc = arith.maxsi %ysum, %c1 : index
    %xm1 = arith.subi %gx, %c1 : index
    %xl = arith.minsi %xm1, %c0 : index
    %xbase = arith.addi %xl, %c3 : index
    %xend = arith.subi %c54, %gx : index
    %xr = arith.minsi %xend, %c0 : index
    %xsum = arith.addi %xbase, %xr : index
    %xc = arith.maxsi %xsum, %c1 : index
    %count = arith.muli %yc, %xc : index
    %count32 = arith.index_cast %count : index to i32
    %countm1 = arith.subi %count32, %c1_i32 : i32
    %leading = math.ctlz %countm1 : i32
    %log = arith.subi %c32_i32, %leading : i32
    %log64 = arith.extui %log : i32 to i64
    %numerator = arith.shli %base, %log64 : i64
    %count64 = arith.extui %count32 : i32 to i64
    %quotient = arith.divui %numerator, %count64 : i64
    %multiplier = arith.trunci %quotient : i64 to i32
    %log8 = arith.trunci %log : i32 to i8
    %shift = arith.addi %log8, %c30_i8 : i8
    %scaled = tosa.apply_scale %in, %multiplier, %shift rounding_mode<SINGLE_ROUND> : (i32, i32, i8) -> i32
    %clamp_lo = arith.maxsi %scaled, %lo : i32
    %clamp_hi = arith.minsi %clamp_lo, %hi : i32
    %value = arith.trunci %clamp_hi : i32 to i8
    linalg.yield %value : i8
  } -> tensor<1x2x56x11xi8>
  return %result : tensor<1x2x56x11xi8>
}
