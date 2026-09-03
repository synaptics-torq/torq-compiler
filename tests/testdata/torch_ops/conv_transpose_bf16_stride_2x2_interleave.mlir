// Regression: ConvTranspose with stride 2x2 (two strided dimensions).
//
// torch-mlir decomposes ConvTranspose k2x2/s2x2/pads0 into a zero-interleave of
// the input (tensor.insert_slice with strides [1,1,2,2]) followed by a plain
// stride-1 conv over flipped weights. convertToInterleaved (Conv2DPattern.cpp)
// used to pick only the FIRST strided dimension and lower the upsample to a
// single row-interleave (torq_hl.interleaved_insert doubles H only), silently
// dropping the W interleave: input pixels land at columns c+1 instead of
// 2c+1, the right half of the conv input stays zero and every output pixel is
// a wrong linear combination (~99% of elements wrong on sim).
//
// This is the PP-OCRv6-tiny detector-head deconv (surgered det_800x608:
// 40x Slice -> ConvTranspose k2x2/s2x2 -> Concat) reduced to one op.
// The fix keeps multi-axis-strided upsamples on the plain strided-insert path.
module {
  func.func @convtranspose_s2x2(%arg0: !torch.vtensor<[1,8,4,16],bf16>) -> !torch.vtensor<[1,8,8,32],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<w> : tensor<8x8x2x2xbf16>} : () -> !torch.vtensor<[8,8,2,2],bf16> 
    %none = torch.constant.none
    %1 = torch.operator "onnx.ConvTranspose"(%arg0, %0) {torch.onnx.kernel_shape = [2 : si64, 2 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,8,4,16],bf16>, !torch.vtensor<[8,8,2,2],bf16>) -> !torch.vtensor<[1,8,8,32],bf16> 
    return %1 : !torch.vtensor<[1,8,8,32],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      w: "0x08000000213A193E0CBEE4BE69BEFEBEF63C2C3F7CBE9FBE7B3E373E583DEEBE70BCB23E2CBF6ABE73BF25BF6CBFF1BD22BF0B3EA13DBFBDA1BF8ABEC7BC683D44BF75BEFBBECFBE083FCFBE85BCE23E95BE65BD623D033D1DBF1C3D2E3F46BFDC3E743DA4BE803FC33E1ABF193D943EC1BDAF3E08BDAB3E383FADBED03D6DBE823D18BF94BEC9BDE63E133F29BFCBBEA63E7FBF6DBE47BD213FB03E28BE3DBE00BE433F5BBE1BBE353E77BDCABD0FBFBDBB63BE153FA73E46BCAB3E2EBE073F31BB953E25BF323E58BF82BF1CBEE6BEA83D903FD5BEA0BED23D7C3EB5BDD3BDB43E853E04BF22BD913C07BF053EDCBEF93EC53D373D97BE73BD80BF11BF3A3E88BFD93E60BFC23ED8BEC73E863D45BF203F393F07BD0CBEA4BDFABE0D3F8BBED2BCCBBEA0BE24BF213F9EBDF73EDA3BB2BE27BE8FBE823B40BE1ABE30BFCFBE543FACBE07BF2D3E343F3ABFD6BDA2BE61BFBC3E40BC123DC1BE693E8ABE92BD0EBF1CBF2B3F82BE153E8ABC62BE82BEA13E1BBE9BBD363C173FAE3E443E90BE31BFF33EF73E90BD8B3EC83ED53EEC3E69BE423F20BFDD3E7D3EE03E713F3E3F13BF58BFD13E02BFCBBBD73E52BF87BF053EB63CFCBD9E3CDCBE42BFABBDF9BE52BF813EFBBC503EFDBEA8BE00BFE3BEC83DC8BE363E2E3E823F32BFE33E37BDE6BB3ABF6CBEBE3E29BD263D15BE143F30BC8DBFB1BE7CBFD0BF88BE2B3FC13C16BFF1BE"
    }
  }
#-}

