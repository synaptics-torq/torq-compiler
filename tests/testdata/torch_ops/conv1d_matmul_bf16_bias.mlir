// Standard ONNX Conv1D that lowers through Conv1D-as-matmul with per-channel bias.
module {
  func.func @main(%input: !torch.vtensor<[1,3,4],bf16>) -> !torch.vtensor<[1,2,4],bf16>
      attributes {torch.onnx_meta.ir_version = 11 : si64,
                  torch.onnx_meta.opset_version = 22 : si64,
                  torch.onnx_meta.producer_name = "test",
                  torch.onnx_meta.producer_version = "1.0"} {
    %weights = torch.operator "onnx.Constant"() {torch.onnx.value = dense<[[[1.000000e+00], [3.000000e+00], [5.000000e+00]], [[2.000000e+00], [4.000000e+00], [6.000000e+00]]]> : tensor<2x3x1xbf16>} : () -> !torch.vtensor<[2,3,1],bf16>
    %bias = torch.operator "onnx.Constant"() {torch.onnx.value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xbf16>} : () -> !torch.vtensor<[2],bf16>

    %output = torch.operator "onnx.Conv"(%input, %weights, %bias)
        {torch.onnx.auto_pad = "NOTSET",
         torch.onnx.dilations = [1 : si64],
         torch.onnx.group = 1 : si64,
         torch.onnx.kernel_shape = [1 : si64],
         torch.onnx.pads = [0 : si64, 0 : si64],
         torch.onnx.strides = [1 : si64]}
        : (!torch.vtensor<[1,3,4],bf16>, !torch.vtensor<[2,3,1],bf16>, !torch.vtensor<[2],bf16>) -> !torch.vtensor<[1,2,4],bf16>

    return %output : !torch.vtensor<[1,2,4],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
    }
  }
#-}
