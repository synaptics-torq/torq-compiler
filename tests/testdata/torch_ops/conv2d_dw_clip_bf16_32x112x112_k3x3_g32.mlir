// Standalone test case for onnx.Conv (depthwise) + onnx.Clip
// Extracted from MobileNetV2 ONNX model (features.1.conv.0.0)
//   input  : [1, 32, 112, 112]  bf16  (N=1, C=32, H=112, W=112)
//   weight : [32, 1, 3, 3]      bf16  (C_out=32, C_in/group=1, kH=3, kW=3)
//   bias   : [32]                bf16
//   output : [1, 32, 112, 112]  bf16
//   attrs  : dilations=[1,1], group=32, kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1]
//   clip   : min=0.0, max=6.0 (ReLU6)
//
// Output shape derivation (ONNX Conv formula):
//   H_out = floor((H_in + pad_top + pad_bot - kH) / stride_H) + 1
//         = floor((112 + 1 + 1 - 3) / 1) + 1 = 112
//   W_out = floor((W_in + pad_left + pad_right - kW) / stride_W) + 1
//         = floor((112 + 1 + 1 - 3) / 1) + 1 = 112

module {
  func.func @main(%arg0: !torch.vtensor<[1,32,112,112],bf16>) -> !torch.vtensor<[1,32,112,112],bf16>
      attributes {torch.onnx_meta.ir_version = 10 : si64,
                  torch.onnx_meta.opset_version = 22 : si64,
                  torch.onnx_meta.producer_name = "pytorch",
                  torch.onnx_meta.producer_version = "2.9.1+cu128"} {
    %weight = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<dw_weight> : tensor<32x1x3x3xbf16>} : () -> !torch.vtensor<[32,1,3,3],bf16>
    %bias = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<dw_bias> : tensor<32xbf16>} : () -> !torch.vtensor<[32],bf16>
    %min_val = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<min_val_cast> : tensor<bf16>} : () -> !torch.vtensor<[],bf16>
    %max_val = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<max_val_cast> : tensor<bf16>} : () -> !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %conv = torch.operator "onnx.Conv"(%arg0, %weight, %bias) {torch.onnx.auto_pad = "NOTSET", torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 32 : si64, torch.onnx.pads = [1 : si64, 1 : si64, 1 : si64, 1 : si64], torch.onnx.strides = [1 : si64, 1 : si64]} : (!torch.vtensor<[1,32,112,112],bf16>, !torch.vtensor<[32,1,3,3],bf16>, !torch.vtensor<[32],bf16>) -> !torch.vtensor<[1,32,112,112],bf16>
    %out = torch.operator "onnx.Clip"(%conv, %min_val, %max_val) : (!torch.vtensor<[1,32,112,112],bf16>, !torch.vtensor<[],bf16>, !torch.vtensor<[],bf16>) -> !torch.vtensor<[1,32,112,112],bf16>
    return %out : !torch.vtensor<[1,32,112,112],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      dw_weight: "0x080000000E3ED13F75BEBA3C8C3FE9BE05BF57BFA6BD5A3E993D9FBE6F3FE03E6FBEF9BE30BE2EBE06BDA3BE07BE823DAB3FBABD24BD84BE04BE8B3DF13C3B3C463EBFBF05BE94BEB53F2A3E3EBD50BEE9BD4C3CF03F91BD89BD80BE11BEE2BD8CBDFD3DA64096C08CBE2FBE98BE9D3E833EAE3D7FBEDFBF92BDE43FC8BF26BED93FDA3DA8BED93D303DAAC08C3E25BEB540DCBE4DBF89BE323FC8400F3FCAC07640FFBC7EC028C052BD1D409AC0AA3CA940BBBF8B3D9F3F7A3EC33C52BEB8C0AF40B83E8EBD6B3E9EBEAA3F96BE0DC00340443F7ABF3ABEEABE8EBE273F5B3EABBEE83E8C3EC3BDEEBE18BEFDBDEE27C826AAA74328D22709A87A27F926ECA763BFD9BFF23C63BE15BE3E3E9A3FFD3F7FBE30BEF93F21BE79BD243FE1BD5FBE8FBF0EBD9B3C16BD8A3C753E7CBF833E3F3EA4BE653E23BE87BE86BDF63DA53F9ABC28BE69BE59BDA23DDE3D823D25BF7FBF723EB03E383E863DB8BC1F3D69BC4ABC4CBD903C723D82BF943CD93CA9BE333EA03DF3C0EE3E69BCFF402DBF023F923F04BE893E3D3F65BE4EBF0FBF0FBE87BDCBBDACBD4ABEBA3F78BE6FBD47BD78BD023F6C3F96BD4D3E363F83BE3ABFC0BE43BE993D703E8F3DF23D9FBFBC3CBF3D613E473D51BF753D283FA54094BE98C03C400ABE36C001BE4ABDAA3C993F25BE58BCFBBD6CBDAFBC06C0C7C080C0BB3EAA3F36BDE43F9B407D4023BE6DBE98BD0A3ECE3F99BD32BE9BBE8ABD4FA81828552888A84F29A52818A71BA8BBA873BFA03F53BF09BF8940F9BEA6BE8DBF1EBFC93D603DBF3D433EBEBF6E3EA53DD13DB03D",
      dw_bias: "0x08000000D7BDB83EB23C42401A3D3B4049405C402C40FA3F40400040D53E6CBF2D40343E7E40093E9040BC401940AD3D443D833D0E402340883F5F406B3E6EBF803FBD40",
      min_val_cast: "0x080000000000",
      max_val_cast: "0x08000000C040"
    }
  }
#-}
