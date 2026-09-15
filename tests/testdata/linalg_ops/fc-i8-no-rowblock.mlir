// Gate check: an int8 fully_connected must NOT take the row-blocked (fc-fast) kernel.
// Same IR shape as conv1d-matmul-fc-i8-bias-requant.mlir but with N=16 rows, K=64 and
// F=64, which is the size range where the fast kernel would be picked if the isFloat()
// gate in FCToHw.cpp were dropped. For int8 a per-item bias/scale narrows the ACT width
// from 16 to 4, and draining four rows of that overflows the S field of the reg-NDL
// descriptor (torq_api.c asserts dims.sn <= 16), so lifting the gate aborts the compile.
//
// The weight is an argument, not a constant, purely to keep this file small.
module {
  func.func @main(%im2col: tensor<16x64xi8>, %weights: tensor<64x64xi8>) -> tensor<1x64x16xi8> {
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c_min = arith.constant -128 : i32
    %c_max = arith.constant 127 : i32

    %bias = arith.constant dense<[1, -14, -37, 9, -12, -27, -14, 4, 2, -5, 39, -39, -16, -31, -29, -20, 35, -1, -7, -17, -35, -22, 21, -28, -33, 9, -8, -29, 32, 34, -12, -33, -32, -3, -39, -6, -24, 5, 6, 29, -18, -23, 7, -8, 7, 6, -19, 26, -26, -9, -19, -4, 8, -37, -12, -16, -12, 9, 6, -10, 20, -7, -40, -34]> : tensor<64xi32>
    %multiplier = arith.constant dense<[1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824, 1073741824]> : tensor<64xi32>
    %shift = arith.constant dense<[30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]> : tensor<64xi8>

    %mm_init = tensor.empty() : tensor<16x64xi32>
    %mm_zero = linalg.fill ins(%c0_i32 : i32) outs(%mm_init : tensor<16x64xi32>) -> tensor<16x64xi32>
    %matmul = linalg.matmul
        ins(%im2col, %weights : tensor<16x64xi8>, tensor<64x64xi8>)
        outs(%mm_zero : tensor<16x64xi32>) -> tensor<16x64xi32>

    %trans_init = tensor.empty() : tensor<64x16xi32>
    %trans = linalg.transpose
        ins(%matmul : tensor<16x64xi32>)
        outs(%trans_init : tensor<64x16xi32>) permutation = [1, 0]
    %expand = tensor.expand_shape %trans [[0, 1], [2]] output_shape [1, 64, 16]
        : tensor<64x16xi32> into tensor<1x64x16xi32>

    %bias_empty = tensor.empty() : tensor<1x64x16xi32>
    %biased = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d1)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%expand, %bias : tensor<1x64x16xi32>, tensor<64xi32>)
        outs(%bias_empty : tensor<1x64x16xi32>) {
      ^bb0(%in: i32, %b: i32, %out: i32):
        %sum = arith.addi %in, %b : i32
        linalg.yield %sum : i32
    } -> tensor<1x64x16xi32>

    %req_empty = tensor.empty() : tensor<1x64x16xi8>
    %requant = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d1)>,
          affine_map<(d0, d1, d2) -> (d1)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%biased, %multiplier, %shift : tensor<1x64x16xi32>, tensor<64xi32>, tensor<64xi8>)
        outs(%req_empty : tensor<1x64x16xi8>) {
      ^bb0(%in: i32, %mul: i32, %sh: i8, %out: i8):
        %scaled = tosa.apply_scale %in, %mul, %sh {rounding_mode = DOUBLE_ROUND} : (i32, i32, i8) -> i32
        %with_zp = arith.addi %scaled, %c3_i32 : i32
        %clamp_min = arith.maxsi %with_zp, %c_min : i32
        %clamp_max = arith.minsi %clamp_min, %c_max : i32
        %trunc = arith.trunci %clamp_max : i32 to i8
        linalg.yield %trunc : i8
    } -> tensor<1x64x16xi8>

    return %requant : tensor<1x64x16xi8>
  }
}
