module {
  func.func @subgraph_8_to_12(%arg0: !torch.vtensor<[1,3136,64],bf16>) -> !torch.vtensor<[1,3137,64],bf16> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__ConstantOfShape_output_0_subgraph_init0> : tensor<3xsi64>} : () -> !torch.vtensor<[3],si64> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__Constant_1_output_0_subgraph_init1> : tensor<3xsi64>} : () -> !torch.vtensor<[3],si64> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__Constant_3_output_0_subgraph_init2> : tensor<si64>} : () -> !torch.vtensor<[],si64> 
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<cls_token1_subgraph_init3> : tensor<1x1x64xbf16>} : () -> !torch.vtensor<[1,1,64],bf16> 
    %none = torch.constant.none
    %4 = torch.operator "onnx.Mul"(%0, %2) : (!torch.vtensor<[3],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[3],si64> 
    %5 = torch.operator "onnx.Equal"(%1, %4) : (!torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3],i1> 
    %6 = torch.operator "onnx.Where"(%5, %0, %1) : (!torch.vtensor<[3],i1>, !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3],si64> 
    %7 = torch.operator "onnx.Expand"(%3, %6) : (!torch.vtensor<[1,1,64],bf16>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[1,1,64],bf16> 
    %8 = torch.operator "onnx.Concat"(%7, %arg0) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[1,1,64],bf16>, !torch.vtensor<[1,3136,64],bf16>) -> !torch.vtensor<[1,3137,64],bf16> 
    return %8 : !torch.vtensor<[1,3137,64],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      __ConstantOfShape_output_0_subgraph_init0: "0x08000000010000000000000001000000000000000100000000000000",
      __Constant_1_output_0_subgraph_init1: "0x080000000100000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
      __Constant_3_output_0_subgraph_init2: "0x08000000FFFFFFFFFFFFFFFF",
      cls_token1_subgraph_init3: "0x08000000AC3B17BD4B3C533D9E3D153C77BD743BBD3CEE3C2A3C973AE83CF73B4EBD063CDFBD833C323C0E3C34BDE13CA53C5EBB193D62BD2ABE4E3C8C3C16BE1C3D0ABDFA3BC23C683DD8BDA33C023EBDBBA83CA93D5B3D0EBC23BC97BEDE3C233C853C1B3EA43DFFBC593DE337803D8E3DE23C4F3D963C2D3E7FBC143D89BD92BCC73D"
    }
  }
#-}

