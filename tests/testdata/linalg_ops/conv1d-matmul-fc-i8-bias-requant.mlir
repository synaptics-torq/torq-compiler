// Regression for the integer bias / requant path in
// Conv1DMatmulToTorqHlFCPattern (compiler/torq/Conversions/LinalgToTorqHL/
// Conv1DMatmulPattern.cpp). The IR shape below mirrors the output of
// Conv1DNcwFcwToLinalgMatmulPattern for an int8 pointwise Conv1D:
//
//   matmul [Ow, K] x [K, F] -> [Ow, F]            (i8 x i8 -> i32)
//   linalg.transpose [1, 0]                       -> [F, Ow]
//   tensor.expand_shape [[0,1],[2]]               -> [1, F, Ow]
//   per-channel bias addi                         -> [1, F, Ow] i32
//   per-channel tosa.apply_scale + zp + clamp     -> [1, F, Ow] i8
//
// Exercises the isInt branch that builds an interleaved [2*F] bias/scale
// operand for torq_hl.fully_connected.
module {
  func.func @main(%im2col: tensor<4x3xi8>) -> tensor<1x2x4xi8> {
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c_min = arith.constant -128 : i32
    %c_max = arith.constant 127 : i32

    // Weights already canonicalized to [K=3, F=2] (matmul RHS layout).
    %weights = arith.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi8>
    %bias = arith.constant dense<[7, -9]> : tensor<2xi32>
    %multiplier = arith.constant dense<1073741824> : tensor<2xi32>
    %shift = arith.constant dense<30> : tensor<2xi8>

    // matmul [Ow=4, K=3] x [K=3, F=2] -> [Ow=4, F=2] in i32.
    %mm_init = tensor.empty() : tensor<4x2xi32>
    %mm_zero = linalg.fill ins(%c0_i32 : i32) outs(%mm_init : tensor<4x2xi32>) -> tensor<4x2xi32>
    %matmul = linalg.matmul
        ins(%im2col, %weights : tensor<4x3xi8>, tensor<3x2xi8>)
        outs(%mm_zero : tensor<4x2xi32>) -> tensor<4x2xi32>

    // Post-matmul layout chain: [Ow, F] -> [F, Ow] -> [1, F, Ow].
    %trans_init = tensor.empty() : tensor<2x4xi32>
    %trans = linalg.transpose
        ins(%matmul : tensor<4x2xi32>)
        outs(%trans_init : tensor<2x4xi32>) permutation = [1, 0]
    %expand = tensor.expand_shape %trans [[0, 1], [2]] output_shape [1, 2, 4]
        : tensor<2x4xi32> into tensor<1x2x4xi32>

    // Per-channel bias add along F (dim 1).
    %bias_empty = tensor.empty() : tensor<1x2x4xi32>
    %biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d1)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%expand, %bias : tensor<1x2x4xi32>, tensor<2xi32>)
        outs(%bias_empty : tensor<1x2x4xi32>) {
      ^bb0(%in: i32, %b: i32, %out: i32):
        %sum = arith.addi %in, %b : i32
        linalg.yield %sum : i32
    } -> tensor<1x2x4xi32>

    // Per-channel requant: apply_scale + output zp + clamp + trunc to i8.
    %req_empty = tensor.empty() : tensor<1x2x4xi8>
    %requant = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d1)>,
          affine_map<(d0, d1, d2) -> (d1)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%biased, %multiplier, %shift : tensor<1x2x4xi32>, tensor<2xi32>, tensor<2xi8>)
        outs(%req_empty : tensor<1x2x4xi8>) {
      ^bb0(%in: i32, %mul: i32, %sh: i8, %out: i8):
        %scaled = tosa.apply_scale %in, %mul, %sh {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
        %with_zp = arith.addi %scaled, %c3_i32 : i32
        %clamp_min = arith.maxsi %with_zp, %c_min : i32
        %clamp_max = arith.minsi %clamp_min, %c_max : i32
        %trunc = arith.trunci %clamp_max : i32 to i8
        linalg.yield %trunc : i8
    } -> tensor<1x2x4xi8>

    return %requant : tensor<1x2x4xi8>
  }
}
