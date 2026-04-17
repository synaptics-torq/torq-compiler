module {
  func.func @main(%input: !torch.vtensor<[1,1,64],bf16>, %kv0:   !torch.vtensor<[1,4,8,64],bf16>, %kv1:   !torch.vtensor<[1,4,8,64],bf16>) -> (!torch.vtensor<[1,1,256],bf16>, !torch.vtensor<[1,4,8,64],bf16>, !torch.vtensor<[1,4,8,64],bf16>) attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "test", torch.onnx_meta.producer_version = "1.0"} {
    %weight = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.0> : tensor<64x256xbf16>} : () -> !torch.vtensor<[64,256],bf16>
    %logits = torch.operator "onnx.MatMul"(%input, %weight) : (!torch.vtensor<[1,1,64],bf16>, !torch.vtensor<[64,256],bf16>) -> !torch.vtensor<[1,1,256],bf16>
    %kv0_out = torch.operator "onnx.Identity"(%kv0) : (!torch.vtensor<[1,4,8,64],bf16>) -> !torch.vtensor<[1,4,8,64],bf16>
    %kv1_out = torch.operator "onnx.Identity"(%kv1) : (!torch.vtensor<[1,4,8,64],bf16>) -> !torch.vtensor<[1,4,8,64],bf16>
    return %logits, %kv0_out, %kv1_out : !torch.vtensor<[1,1,256],bf16>, !torch.vtensor<[1,4,8,64],bf16>, !torch.vtensor<[1,4,8,64],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
    }
  }
#-}
