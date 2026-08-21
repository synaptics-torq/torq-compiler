// TORQ_FP_MAX_TOL: 0.05
// TORQ_ALLOWED_WRONG: 0.03
// Regression: tiling an SE-block global-average-pool (keepdims ReduceMean).
//
// This is the PP-OCRv6-tiny squeeze-excite block: a keepdims ReduceMean reduces
// [1,32,80,80] -> [1,32,1,1] (the global avg-pool), a 1x1 Conv/Relu/Conv/HardSigmoid
// gate, then Mul back onto the input. The 80x80 reduction is large enough that
// tile-and-fuse must tile it, which exposed two problems (both handled in
// TileAndFusePass, with no change to the vendored upstream tiling interface):
//   * A reduction fuse-group member whose reduced dims don't appear in the
//     consumer's result tile made getIterationDomainTileFromResultTile fail; the old
//     code returned doNotFuse() for a member it could not skip, sending the SCF
//     tile-and-fuse driver into an infinite loop (asserts "Unable to fuse a pattern
//     fuse group member").
//   * The keepdims size-1 reduced dims map to a constant result-map position, which
//     the upstream linalg TilingInterface refuses to tile. Rather than patch that
//     interface, TileAndFusePass rewrites the reduction (only when its fuse-group is
//     about to be tiled) into a non-keepdims reduce + tensor.expand_shape, whose
//     projected-permutation map the interface accepts. The reduce accumulates in f32
//     (truncating back to bf16) so it keeps the accuracy of the pooling kernel the
//     keepdims form would otherwise have used.
//
// Before the fix: torq-compile aborts/hangs in TileAndFuse. After: compiles and
// matches the reference.
module {
  func.func @"Extracted from {PaddlePaddle Graph in PIR mode}"(%arg0: !torch.vtensor<[1,32,80,80],bf16>) -> !torch.vtensor<[1,32,80,80],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "onnx.utils.extract_model", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_v_414_folded> : tensor<2xsi64>} : () -> !torch.vtensor<[2],si64> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_conv2d_7.w_0_bf16> : tensor<8x32x1x1xbf16>} : () -> !torch.vtensor<[8,32,1,1],bf16> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_v_698_bf16> : tensor<8xbf16>} : () -> !torch.vtensor<[8],bf16> 
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_conv2d_8.w_0_bf16> : tensor<32x8x1x1xbf16>} : () -> !torch.vtensor<[32,8,1,1],bf16> 
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_v_701_bf16> : tensor<32xbf16>} : () -> !torch.vtensor<[32],bf16> 
    %none = torch.constant.none
    %5 = torch.operator "onnx.ReduceMean"(%arg0, %0) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[1,32,80,80],bf16>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[1,32,1,1],bf16> 
    %6 = torch.operator "onnx.Conv"(%5, %1, %2) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [1 : si64, 1 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], torch.onnx.strides = [1 : si64, 1 : si64]} : (!torch.vtensor<[1,32,1,1],bf16>, !torch.vtensor<[8,32,1,1],bf16>, !torch.vtensor<[8],bf16>) -> !torch.vtensor<[1,8,1,1],bf16> 
    %7 = torch.operator "onnx.Relu"(%6) : (!torch.vtensor<[1,8,1,1],bf16>) -> !torch.vtensor<[1,8,1,1],bf16> 
    %8 = torch.operator "onnx.Conv"(%7, %3, %4) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [1 : si64, 1 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], torch.onnx.strides = [1 : si64, 1 : si64]} : (!torch.vtensor<[1,8,1,1],bf16>, !torch.vtensor<[32,8,1,1],bf16>, !torch.vtensor<[32],bf16>) -> !torch.vtensor<[1,32,1,1],bf16> 
    %9 = torch.operator "onnx.HardSigmoid"(%8) {torch.onnx.alpha = 0.166666701 : f32, torch.onnx.beta = 5.000000e-01 : f32} : (!torch.vtensor<[1,32,1,1],bf16>) -> !torch.vtensor<[1,32,1,1],bf16> 
    %10 = torch.operator "onnx.Mul"(%arg0, %9) : (!torch.vtensor<[1,32,80,80],bf16>, !torch.vtensor<[1,32,1,1],bf16>) -> !torch.vtensor<[1,32,80,80],bf16> 
    return %10 : !torch.vtensor<[1,32,80,80],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      _v_414_folded: "0x0800000002000000000000000300000000000000",
      _conv2d_7.w_0_bf16: "0x0800000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000CDBECEBED7BD04BFF63DB9BD923E293F82BECCBEF63DDE3E4F3E2C3F033E58BE0CBEDB3EF7BDF4BE23BED33D563EA0BEB4BE9ABE503E4DBDD2BE5BBFB83EB33E60BCCDBEB3BCA23DF2BE09BFA93E93BF273F003E02BED63E54BFB8BEBFBE6EBF78BE86BE01C0753E833E9D3DEABB45BD04BE24BD1B3E213E94BEE83E5B3FF2BE1E3FB93EBC3D6FBD4DBEC1BE0B3D863ED93EE3BEBFBDF6BD4C3F3C3C153E7FBFF3BC8A3DCB3DEDBF8FBEC9BDF43D8D3F35BDBA3DEC3DDCBE1E3EBBBED23ECC3E273FD43DE03D0BBF5F3FE0BE2D3EDDBEB4BD163EAF3E14BE9C3EAC3EB63E663FB6BE27BF0A3FD23DA73E473E853E9DBDF0BE95BEE5BC093FD43E073E29BE993EE93EC1BEBE3ECBBE883D233E19BE49BFF23EFB3E91BEB7BD183F613E803DACBDA33EFBBE8F3DEDBE52BE89BD28BE073F54BD81BE9F3D43BF683E4A3D193F583DD13C033FFD3E753E3F3F853E30BD583F1A3FEFBE953E503DAC3E343F8B3EB03E95BE5CBF9CBEA1BCF03D41BE1F3D07BE793E5C3F5CBEA53E5ABEA9BED03E36BF4ABF7C3EF1BD443DCABC21BE923E6E3F85BF92BF393E2CBB45BFDA3DB7BE89BF7ABE333F853E703E313EFF3C233EA6BE5D3ECF3DA03D033F24BC93BE33BFA8BB",
      _v_698_bf16: "0x080000000000E03EB23E4E3E363F873E963D0CBD",
      _conv2d_8.w_0_bf16: "0x08000000000048BF5A3E923C20BFC83DABBEC63C0000F43D173F553FFD3E253E92BE703E0000DC3E023F3A3EAFBEDCBD39BE7B3F00000D3E8C3F0BBFE4BE83BE21BFB03E000048BD523F0DBF333F03BFD73D3C3F0000233E4B3F6E3EB0BE40BE14BF2C3F0000CF3E0B3D3EBD63BF74BE57BF843F0000AABD863FA7BE5BBD5F3E27BF493D0000BC3E65BC32BF2ABE5CBE21BF873F000032BEC43E713D2B3E37BC9BBE673F0000EF3E073F2FBFA0BE303E84BF223F0000AB3C983FC33EC1BBCDBC04BF89BE000015BD0C3F2D3F88BEA0BEDEBE343F00007B3E613F1F3D10BEC3BDDABE973E0000F33E1DBD533FDA3E083FC9BE3FBE0000053F3E3F00BFA6BEA7BEBBBE083F00004F3E063F2EBE86BE14BD22BF4B3F0000D3BC6F3F28BEAB3E3A3E07BE31BE0000563FBD3F62BFACBE72BDCFBD883E0000543F9FBE573F34BFFEBD1CBF383F0000673F013F32BF2FBF653E71BF913E00009C3E9D3E65BEEC3C5DBE02BF433F0000B53E2A3F3E3EF3BD03BE23BF303F0000F73B033EDE3E6BBE82BEF2BE203F0000F83E553F42BCE1BED2BEA8BE233F000091BE233F753D79BEA5BDDDBE713E00006C3EFBBE6B3E213FE3BC3DBF91BE00002D3F383E593FB1BE1A3D44BF983D0000FA3D913E3DBE5F3FEDBE1EBEE63E0000B03FB73E01BE9ABF6ABEA1BE683F0000A13D4E3F6D3F88BE363E33BF9F3D00002C3EDD3EE63E70BECD3E2BBF9A3E",
      _v_701_bf16: "0x08000000FC3EAFBECFBD6C3D29BE8B3E183D98BD53BCE3BD1F3E85BED3BD043E0F3FF1BE963E88BE88BE9FBE773E31BE2BBE213E653D17BD243DF93C393F133D5EBE95BE"
    }
  }
#-}

