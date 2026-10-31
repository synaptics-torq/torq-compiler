// A quantized conv1d with a kernel wider than the HW kernel border allows (more than 7
// columns), SAME padding. The frontend keeps the padding explicit and the conv runs on the
// EasyKernel 1D valid path (isWideKernel1DValidConv). k9 is a multiple of the ALU kernel width
// (3); k11 is not, so its weights get one zero column. Random weights: a few outputs differ from
// the float reference by one int8 step (0.0145), as they do with a 7-column kernel on the
// regular path. The gate allows 0.01 * max|expected| (about 0.017): one step, not two.
// TORQ_USE_ABS_TOL_GATE: 1
// TORQ_FP_ABS_TOL_FRAC: 0.01
module {
  func.func @Conv_node_conv1d_wide_conv2d_standalone(%arg0: !torch.vtensor<[1,8,100],f32>) -> !torch.vtensor<[1,8,100],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "onnx.quantize", torch.onnx_meta.producer_version = "0.1.0"} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<conv1d_wide_nchw_shape> : tensor<4xsi64>} : () -> !torch.vtensor<[4],si64> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<conv1d_wide_shape> : tensor<3xsi64>} : () -> !torch.vtensor<[3],si64> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.00784313586> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<-7> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %5 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0145266987> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %6 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %7 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<8.76599399E-4> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %8 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<conv1d_wide_weight_3d_quantized> : tensor<8x8x1x11xsi8>} : () -> !torch.vtensor<[8,8,1,11],si8> 
    %9 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<conv1d_wide_bias_bf16_quantized> : tensor<8xsi32>} : () -> !torch.vtensor<[8],si32> 
    %10 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<conv1d_wide_bias_bf16_quantized_scale> : tensor<1xf32>} : () -> !torch.vtensor<[1],f32> 
    %11 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si32>} : () -> !torch.vtensor<[],si32> 
    %none = torch.constant.none
    %12 = torch.operator "onnx.DequantizeLinear"(%9, %10, %11) : (!torch.vtensor<[8],si32>, !torch.vtensor<[1],f32>, !torch.vtensor<[],si32>) -> !torch.vtensor<[8],f32> 
    %13 = torch.operator "onnx.DequantizeLinear"(%8, %7, %6) : (!torch.vtensor<[8,8,1,11],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[8,8,1,11],f32> 
    %14 = torch.operator "onnx.Reshape"(%arg0, %0) : (!torch.vtensor<[1,8,100],f32>, !torch.vtensor<[4],si64>) -> !torch.vtensor<[1,8,1,100],f32> 
    %15 = torch.operator "onnx.QuantizeLinear"(%14, %3, %2) : (!torch.vtensor<[1,8,1,100],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,8,1,100],si8> 
    %16 = torch.operator "onnx.DequantizeLinear"(%15, %3, %2) : (!torch.vtensor<[1,8,1,100],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,8,1,100],f32> 
    %17 = torch.operator "onnx.Conv"(%16, %13, %12) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [1 : si64, 11 : si64], torch.onnx.pads = [0 : si64, 5 : si64, 0 : si64, 5 : si64], torch.onnx.strides = [1 : si64, 1 : si64]} : (!torch.vtensor<[1,8,1,100],f32>, !torch.vtensor<[8,8,1,11],f32>, !torch.vtensor<[8],f32>) -> !torch.vtensor<[1,8,1,100],f32> 
    %18 = torch.operator "onnx.QuantizeLinear"(%17, %5, %4) : (!torch.vtensor<[1,8,1,100],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,8,1,100],si8> 
    %19 = torch.operator "onnx.DequantizeLinear"(%18, %5, %4) : (!torch.vtensor<[1,8,1,100],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,8,1,100],f32> 
    %20 = torch.operator "onnx.Reshape"(%19, %1) : (!torch.vtensor<[1,8,1,100],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[1,8,100],f32> 
    return %20 : !torch.vtensor<[1,8,100],f32> 
  }
}

{-#
  dialect_resources: {
    builtin: {
      conv1d_wide_nchw_shape: "0x080000000100000000000000080000000000000001000000000000006400000000000000",
      conv1d_wide_shape: "0x08000000010000000000000008000000000000006400000000000000",
      conv1d_wide_weight_3d_quantized: "0x08000000C5D71484F8CF9BF4B76F47904730804B9E3225A9F133B5DEA168F4E2851F072FCC00FCAA7CA62061717A48E52E29A3FA379A953FAA6893E32EC5A27B945E605F1085F3232D424C1329C88D786ED68CE6FCB62AA63BF9AB617C088A3B70BE57AADA1F334A7017250D3F603DC82EBD3B120B635E3FF56B1F0BD933852371249AB3DF5C1FEF1433C86685B7F056681D5152AF803CE25906F1D19FB6C6804E06EA28AB178F457AF6B493B7439EC6D4290062AB517A72E5C1D410CDD61754E4208450E4F90013FDA69FEF19D4602A4207CDD5A1016D070132613C659A8F5D33F3CB290B8CE73B8344B13A4F8E0915146984A1BB15FDEADEBE5B6AC09504464D8AAAACBC2FAD12158A534C94B8CA42963AF47A189D5B249ACEA899D43732F6F2ACAF3FF53BFEDA139AAB4F55C02C212B65711FC50432C994E5FA6FD82ABC708555DC62B73A200EE7673CFECB836A156E4A3565EAC1B94D35F94BCFD8FDA5FFFF677004AEA2A4124B834FC296F0212B14217342149907E5A9524CEBBEC3112E27DA6994D5F6C6770A005D9A82AA21FC869BD0C4C41F53417A92AB1C12F6D548FB294DB4D212EEC5EA1833F14D0A1043A535B163C4FB381EEFAC3F9741C6963E6013C8E3BE61E9B6261614F8820D6E752FD79C3A4ADAD82B35D42032D44F0FE4DCD50FFEA68128B4243EB70FB0B25B0576BB24496A01683F0396BC0C77485FE839F4B084420E6AB9ABF29597A75E29349D50C04B4072A17C795E95BED617430AC9272543D799D548D898B52B7FE3BD1DC0D104D5A4E168BE96F12E5E378D394EE4CC5ED7DD597EE8AE48B5EC2CF7689600E5EC036F983A9A4FD9F12FFE76CF8F0B3D8A140083CBCAE33481B591642B1224869CAE3C7F5322D49490446B35D2CE8E95E2BE46503F3DC3CDEF4799B5F2D46FF4FAA2FC1EE39949FD1A6E0DB5F1C8052229CA03B8153B018A957448936135DEEBA9C456A9D7DE4BE54BA1426A5FE3B2C7DAFB",
      conv1d_wide_bias_bf16_quantized: "0x08000000BF000000A20B000053FBFFFF8FFAFFFF56020000D1FAFFFF0C070000000B0000",
      conv1d_wide_bias_bf16_quantized_scale: "0x0800000047B2E636"
    }
  }
#-}

