module {
  func.func @conv_izp_i8(%arg0: !torch.vtensor<[1,4,1,16],f32>) -> !torch.vtensor<[1,4,1,16],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 19 : si64} {
    // Input activation quant: NONZERO zero-point (si8 -23, i.e. uint8 105).
    // This is the value that must survive as izp into conv_2d_nchw_fchw_q and be
    // corrected by computeInputZpCorrection on the NSS path.
    %in_scale = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0078125> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %in_zp = torch.operator "onnx.Constant"() {torch.onnx.value = dense<-23> : tensor<si8>} : () -> !torch.vtensor<[],si8>
    // Weight quant: symmetric (zp = 0).
    %w_scale = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0145466495> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %w_zp = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8>
    %w_i8 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<[[[[1, 2, 3]], [[2, 3, 4]], [[3, 4, 5]], [[4, 5, 6]]], [[[2, 3, 4]], [[3, 4, 5]], [[4, 5, 6]], [[5, 6, 7]]], [[[3, 4, 5]], [[4, 5, 6]], [[5, 6, 7]], [[6, 7, 8]]], [[[4, 5, 6]], [[5, 6, 7]], [[6, 7, 8]], [[7, 8, 9]]]]> : tensor<4x4x1x3xsi8>} : () -> !torch.vtensor<[4,4,1,3],si8>
    // Bias quant: scale = in_scale * w_scale, zp = 0.
    %bias_scale = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.13645700E-4> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %bias_zp = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si32>} : () -> !torch.vtensor<[],si32>
    %bias_i32 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<[1, 3, 5, 7]> : tensor<4xsi32>} : () -> !torch.vtensor<[4],si32>
    // Output activation quant: symmetric (zp = 0), scale sets the requant multiplier.
    %out_scale = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0145466495> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %out_zp = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8>

    %bias = torch.operator "onnx.DequantizeLinear"(%bias_i32, %bias_scale, %bias_zp) : (!torch.vtensor<[4],si32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si32>) -> !torch.vtensor<[4],f32>
    %w = torch.operator "onnx.DequantizeLinear"(%w_i8, %w_scale, %w_zp) : (!torch.vtensor<[4,4,1,3],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[4,4,1,3],f32>
    // Quantize the activation with the NONZERO zp, then dequantize -> the QDQ-full-integer
    // fold turns Q(izp=-23)->DQ + Conv into linalg.conv_2d_nchw_fchw_q(..., izp=-23, wzp=0).
    %q = torch.operator "onnx.QuantizeLinear"(%arg0, %in_scale, %in_zp) : (!torch.vtensor<[1,4,1,16],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,4,1,16],si8>
    %dq = torch.operator "onnx.DequantizeLinear"(%q, %in_scale, %in_zp) : (!torch.vtensor<[1,4,1,16],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,4,1,16],f32>
    %conv = torch.operator "onnx.Conv"(%dq, %w, %bias) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [1 : si64, 3 : si64], torch.onnx.pads = [0 : si64, 1 : si64, 0 : si64, 1 : si64], torch.onnx.strides = [1 : si64, 1 : si64]} : (!torch.vtensor<[1,4,1,16],f32>, !torch.vtensor<[4,4,1,3],f32>, !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4,1,16],f32>
    // Output requant Q/DQ pair: the QuantizeLinear supplies the integer conv's
    // per-channel multiplier (so the conv fuses onto the NSS conv path via
    // QConv2DPattern), and the trailing DequantizeLinear returns f32 so float
    // tolerance applies -- exactly like conv1d_307_qdq_tiled.
    %qo = torch.operator "onnx.QuantizeLinear"(%conv, %out_scale, %out_zp) : (!torch.vtensor<[1,4,1,16],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,4,1,16],si8>
    %dqo = torch.operator "onnx.DequantizeLinear"(%qo, %out_scale, %out_zp) : (!torch.vtensor<[1,4,1,16],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,4,1,16],f32>
    return %dqo : !torch.vtensor<[1,4,1,16],f32>
  }
}
