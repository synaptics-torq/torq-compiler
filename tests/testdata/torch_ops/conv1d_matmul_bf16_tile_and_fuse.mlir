// TORQ_FP_MAX_TOL: 0.05
// Standard ONNX Conv1D that lowers through Conv1D-as-matmul and is large enough
// to exercise tile-and-fuse.
module {
  func.func @main(%input: !torch.vtensor<[1,288,207],bf16>) -> !torch.vtensor<[1,1152,207],bf16>
      attributes {torch.onnx_meta.ir_version = 11 : si64,
                  torch.onnx_meta.opset_version = 22 : si64,
                  torch.onnx_meta.producer_name = "test",
                  torch.onnx_meta.producer_version = "1.0"} {
    %weights = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.000000e-02> : tensor<1152x288x1xbf16>} : () -> !torch.vtensor<[1152,288,1],bf16>
    %output = torch.operator "onnx.Conv"(%input, %weights)
        {torch.onnx.auto_pad = "NOTSET",
         torch.onnx.dilations = [1 : si64],
         torch.onnx.group = 1 : si64,
         torch.onnx.kernel_shape = [1 : si64],
         torch.onnx.pads = [0 : si64, 0 : si64],
         torch.onnx.strides = [1 : si64]}
        : (!torch.vtensor<[1,288,207],bf16>, !torch.vtensor<[1152,288,1],bf16>) -> !torch.vtensor<[1,1152,207],bf16>

    return %output : !torch.vtensor<[1,1152,207],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
    }
  }
#-}
