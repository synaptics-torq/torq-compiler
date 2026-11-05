// Overlapping regions are rejected.
func.func @overlapping_regions(
    %init: tensor<1x3x3x1xi8>, %weights: tensor<1xi8>,
    %scale_bias: tensor<4xi32>, %input: tensor<1x4x4x1xi8>) {
  // expected-error @+1 {{'torq_hl.avg_pool2d' op scale regions must not overlap}}
  "torq_hl.avg_pool2d"(%init, %weights, %scale_bias, %input) {
    dilation = array<i64: 1, 1>, input_zp = 0 : i32,
    kernel = array<i64: 2, 2>, output_max = 127 : i32,
    output_min = -128 : i32, output_zp = 0 : i32,
    scale_regions = array<i64: 0, 0, 2, 2, 0, 1, 1, 2, 2, 0>,
    shift_factor = 0 : i32, sliding_pool = true,
    stride = array<i64: 1, 1>
  } : (tensor<1x3x3x1xi8>, tensor<1xi8>, tensor<4xi32>, tensor<1x4x4x1xi8>) -> ()
  return
}

// -----

// Disjoint regions that leave a gap are rejected.
func.func @incomplete_region_coverage(
    %init: tensor<1x3x3x1xi8>, %weights: tensor<1xi8>,
    %scale_bias: tensor<4xi32>, %input: tensor<1x4x4x1xi8>) {
  // expected-error @+1 {{'torq_hl.avg_pool2d' op scale regions must cover the output frame exactly}}
  "torq_hl.avg_pool2d"(%init, %weights, %scale_bias, %input) {
    dilation = array<i64: 1, 1>, input_zp = 0 : i32,
    kernel = array<i64: 2, 2>, output_max = 127 : i32,
    output_min = -128 : i32, output_zp = 0 : i32,
    scale_regions = array<i64: 0, 0, 1, 3, 0, 1, 0, 1, 3, 0>,
    shift_factor = 0 : i32, sliding_pool = true,
    stride = array<i64: 1, 1>
  } : (tensor<1x3x3x1xi8>, tensor<1xi8>, tensor<4xi32>, tensor<1x4x4x1xi8>) -> ()
  return
}

// -----

// Region shifts are range-checked independently of the global shift.
func.func @region_shift_out_of_range(
    %init: tensor<1x3x3x1xi8>, %weights: tensor<1xi8>,
    %scale_bias: tensor<2xi32>, %input: tensor<1x4x4x1xi8>) {
  // expected-error @+1 {{'torq_hl.avg_pool2d' op scale region shift must be between 0 and 62}}
  "torq_hl.avg_pool2d"(%init, %weights, %scale_bias, %input) {
    dilation = array<i64: 1, 1>, input_zp = 0 : i32,
    kernel = array<i64: 2, 2>, output_max = 127 : i32,
    output_min = -128 : i32, output_zp = 0 : i32,
    scale_regions = array<i64: 0, 0, 3, 3, 63>,
    shift_factor = 0 : i32, sliding_pool = true,
    stride = array<i64: 1, 1>
  } : (tensor<1x3x3x1xi8>, tensor<1xi8>, tensor<2xi32>, tensor<1x4x4x1xi8>) -> ()
  return
}

// -----

// The global shift is checked when no scale regions are provided.
func.func @global_shift_out_of_range(
    %init: tensor<1x3x3x1xi8>, %weights: tensor<1xi8>,
    %scale_bias: tensor<2xi32>, %input: tensor<1x4x4x1xi8>) {
  // expected-error @+1 {{'torq_hl.avg_pool2d' op sliding pooling shift must be between 0 and 62}}
  "torq_hl.avg_pool2d"(%init, %weights, %scale_bias, %input) {
    dilation = array<i64: 1, 1>, input_zp = 0 : i32,
    kernel = array<i64: 2, 2>, output_max = 127 : i32,
    output_min = -128 : i32, output_zp = 0 : i32,
    shift_factor = 63 : i32, sliding_pool = true,
    stride = array<i64: 1, 1>
  } : (tensor<1x3x3x1xi8>, tensor<1xi8>, tensor<2xi32>, tensor<1x4x4x1xi8>) -> ()
  return
}

// -----

// Output dimensions must match the valid window geometry.
func.func @bad_window_geometry(
    %init: tensor<1x2x3x1xi8>, %weights: tensor<1xi8>,
    %scale_bias: tensor<2xi32>, %input: tensor<1x4x4x1xi8>) {
  // expected-error @+1 {{'torq_hl.avg_pool2d' op invalid sliding pooling window or output shape}}
  "torq_hl.avg_pool2d"(%init, %weights, %scale_bias, %input) {
    dilation = array<i64: 1, 1>, input_zp = 0 : i32,
    kernel = array<i64: 2, 2>, output_max = 127 : i32,
    output_min = -128 : i32, output_zp = 0 : i32,
    shift_factor = 0 : i32, sliding_pool = true,
    stride = array<i64: 1, 1>
  } : (tensor<1x2x3x1xi8>, tensor<1xi8>, tensor<2xi32>, tensor<1x4x4x1xi8>) -> ()
  return
}
