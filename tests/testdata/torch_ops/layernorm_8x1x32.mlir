module {
  func.func @main_graph(%arg0: !torch.vtensor<[8,1,32],bf16>) -> !torch.vtensor<[8,1,32],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<scale> : tensor<32xbf16>} : () -> !torch.vtensor<[32],bf16>
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<bias> : tensor<32xbf16>} : () -> !torch.vtensor<[32],bf16>
    %none = torch.constant.none
    %2 = torch.operator "onnx.LayerNormalization"(%arg0, %0, %1) {torch.onnx.axis = -1 : si64, torch.onnx.epsilon = 9.99999974E-6 : f32, torch.onnx.stash_type = 16 : si64} : (!torch.vtensor<[8,1,32],bf16>, !torch.vtensor<[32],bf16>, !torch.vtensor<[32],bf16>) -> !torch.vtensor<[8,1,32],bf16>
    return %2 : !torch.vtensor<[8,1,32],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      scale: "0x080000008A3F6B3F643F853F3B3F823F803F8D3F793F3C3F763F913FAC3F963F533F623F993F7E3F7B3FB73F943F963F9E3F883F703F7E3F663F993F883F5C3F863F6B3F",
      bias: "0x0800000046401F402940304023C037405140334040C022C0294036C044C02F402BC027C04EC02F402A4044405BC03F4044C03DC022C037C03B4033C037C024C0414026C0"
    }
  }
#-}
