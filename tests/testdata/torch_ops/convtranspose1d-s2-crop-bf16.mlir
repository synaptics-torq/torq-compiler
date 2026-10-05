// Regression for torq-mark-patterns-for-tile-and-fuse failing to converge, reduced
// from the NNNR3 trcnn upsampling block: a 1-channel 1D ConvTranspose (k=3, s=2)
// whose 257-wide output is cropped back to 256 by a Slice. Marking the fuse group
// through the rewriter requeued its shared producers every iteration, so the greedy
// driver never converged with host and CSS disabled.
module {
  func.func @main(%arg0: !torch.vtensor<[1,1,128],bf16>) -> !torch.vtensor<[1,1,256],bf16> attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_model.trcnn_1d_block6.1.weight_bf16> : tensor<1x1x3xbf16>} : () -> !torch.vtensor<[1,1,3],bf16> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_model.trcnn_1d_block6.1.bias_bf16> : tensor<1xbf16>} : () -> !torch.vtensor<[1],bf16> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__model_trcnn_1d_block6_trcnn_1d_block6.1_Pad_slice_starts> : tensor<3xsi64>} : () -> !torch.vtensor<[3],si64> 
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__model_trcnn_1d_block6_trcnn_1d_block6.1_Pad_slice_ends> : tensor<3xsi64>} : () -> !torch.vtensor<[3],si64> 
    %none = torch.constant.none
    %4 = torch.operator "onnx.ConvTranspose"(%arg0, %0, %1) {torch.onnx.dilations = [1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [3 : si64], torch.onnx.pads = [0 : si64, 0 : si64], torch.onnx.strides = [2 : si64]} : (!torch.vtensor<[1,1,128],bf16>, !torch.vtensor<[1,1,3],bf16>, !torch.vtensor<[1],bf16>) -> !torch.vtensor<[1,1,257],bf16> 
    %5 = torch.operator "onnx.Slice"(%4, %2, %3) : (!torch.vtensor<[1,1,257],bf16>, !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[1,1,256],bf16> 
    return %5 : !torch.vtensor<[1,1,256],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      _model.trcnn_1d_block6.1.weight_bf16: "0x0800000004402040063F",
      _model.trcnn_1d_block6.1.bias_bf16: "0x08000000D8C0",
      __model_trcnn_1d_block6_trcnn_1d_block6.1_Pad_slice_starts: "0x08000000000000000000000000000000000000000000000000000000",
      __model_trcnn_1d_block6_trcnn_1d_block6.1_Pad_slice_ends: "0x08000000010000000000000001000000000000000001000000000000"
    }
  }
#-}

