func.func @pool(%input: tensor<1x4x3x1xi8>) -> tensor<1x2x1x1xi8> {
  %zero = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %step = arith.constant 1 : index
  %c17 = arith.constant 17 : index
  %c19 = arith.constant 19 : index
  %c3 = arith.constant 3 : index
  %c28 = arith.constant 28 : index
  %c55 = arith.constant 55 : index
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %base = arith.constant 1073741825 : i64
  %c30_i8 = arith.constant 30 : i8
  %lo = arith.constant -128 : i32
  %hi = arith.constant 127 : i32
  %kernel = tensor.empty() : tensor<3x3xi32>
  %pool_init = tensor.empty() : tensor<1x2x1x1xi32>
  %pool_init_zero = linalg.fill ins(%zero : i32) outs(%pool_init : tensor<1x2x1x1xi32>) -> tensor<1x2x1x1xi32>
  %out_init = tensor.empty() : tensor<1x2x1x1xi8>
  %result = scf.for %row = %c17 to %c19 step %step iter_args(%iter = %out_init) -> tensor<1x2x1x1xi8> {
    %sum = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %kernel : tensor<1x4x3x1xi8>, tensor<3x3xi32>) outs(%pool_init_zero : tensor<1x2x1x1xi32>) -> tensor<1x2x1x1xi32>
    %output = linalg.generic {
      indexing_maps = [affine_map<(n, h, w, c) -> (n, h, w, c)>, affine_map<(n, h, w, c) -> (n, h, w, c)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%sum : tensor<1x2x1x1xi32>) outs(%iter : tensor<1x2x1x1xi8>) {
    ^bb0(%in: i32, %out: i8):
      %y = linalg.index 1 : index
      %x = linalg.index 2 : index
      %gy = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%y)[%row]
      %gx = arith.addi %x, %c28 : index
      %ym1 = arith.subi %gy, %c1 : index
      %yl = arith.minsi %ym1, %c0 : index
      %ybase = arith.addi %yl, %c3 : index
      %yend = arith.subi %c55, %gy : index
      %yr = arith.minsi %yend, %c0 : index
      %ysum = arith.addi %ybase, %yr : index
      %yc = arith.maxsi %ysum, %c1 : index
      %xm1 = arith.subi %gx, %c1 : index
      %xl = arith.minsi %xm1, %c0 : index
      %xbase = arith.addi %xl, %c3 : index
      %xend = arith.subi %c55, %gx : index
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
    } -> tensor<1x2x1x1xi8>
    scf.yield %output : tensor<1x2x1x1xi8>
  }
  return %result : tensor<1x2x1x1xi8>
}
