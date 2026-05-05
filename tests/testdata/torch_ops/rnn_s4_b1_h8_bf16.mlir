module {
  func.func @main_graph(%arg0: !torch.vtensor<[4,1,8],bf16>, %arg1: !torch.vtensor<[1,1,8],bf16>) -> (!torch.vtensor<[4,1,1,8],bf16>, !torch.vtensor<[1,1,8],bf16>) attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.9.1"} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_onnx__RNN_19> : tensor<1x8x8xbf16>} : () -> !torch.vtensor<[1,8,8],bf16> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_onnx__RNN_20> : tensor<1x8x8xbf16>} : () -> !torch.vtensor<[1,8,8],bf16> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_onnx__RNN_21> : tensor<1x16xbf16>} : () -> !torch.vtensor<[1,16],bf16> 
    %none = torch.constant.none
    %3:2 = torch.operator "onnx.RNN"(%arg0, %0, %1, %2, %none, %arg1) {torch.onnx.activations = ["Tanh"], torch.onnx.hidden_size = 8 : si64} : (!torch.vtensor<[4,1,8],bf16>, !torch.vtensor<[1,8,8],bf16>, !torch.vtensor<[1,8,8],bf16>, !torch.vtensor<[1,16],bf16>, !torch.none, !torch.vtensor<[1,1,8],bf16>) -> (!torch.vtensor<[4,1,1,8],bf16>, !torch.vtensor<[1,1,8],bf16>) 
    return %3#0, %3#1 : !torch.vtensor<[4,1,1,8],bf16>, !torch.vtensor<[1,1,8],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      _onnx__RNN_19: "0x080000002DBB423E95BE85BE0BBEC23DE6BB903E01BDC03DDBBD8EBDADBE70BE15BE573C0F3E593E75BE1EBE033E963E95BD873E69BD193DA43EA8BE64BEB7BD0DBE9C3E6BBE27BE7DBEAABE53BE9C3E223E2F3E983C3ABE753DA9BE83BE3BBE643E543E21BE51BC683EB43E103E443D733E55BE873D8CBE7BBE3BBE243E123E56BEDB3D",
      _onnx__RNN_20: "0x08000000473E37BD5D3CA83D613EAE3E8BBE05BE0E3E963E9E3EA03E903D9DBE053D62BEA9BEA13E8A3EB5BE883D74BD6EBD26BE0B3E56BE053E373E823E073EB3BE6BBE353E983D8DBE50BEAA3E743E1EBEB6BDACBED0BB88BE8CBEA0BC593D14BE573E5CBEA43E783E99BEB4BD833C533DAC3D0E3EAD3C31BE2B3EAEBE57BEB5BD30BE",
      _onnx__RNN_21: "0x08000000FDBD94BE9ABD9B3D6CBE95BC823E15BD213CFABC933D663EAB3E663EAC3ED1BC"
    }
  }
#-}

