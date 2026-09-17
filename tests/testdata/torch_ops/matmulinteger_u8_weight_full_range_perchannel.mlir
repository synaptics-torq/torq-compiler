// TORQ_USE_ABS_TOL_GATE: 1
// TORQ_FP_ABS_TOL_FRAC: 0.01
// A full-range weight doubles max|W - w_zp| against the +/-127 weights the other
// matmulinteger fixtures use, so one LSB of DQL activation noise is worth twice as
// much in the output. W - w_zp also cannot fit int8 here, so the raise centres the
// weight and carries a residual correction term whose own bf16 rounding adds
// further error. Gate on absolute error at a wider fraction of the dynamic range
// than the other matmulinteger fixtures.
module {
  func.func @matmulinteger_u8_weight_full_range_perchannel(%arg0: !torch.vtensor<[4,16],f32>) -> !torch.vtensor<[4,32],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<W> : tensor<16x32xui8>} : () -> !torch.vtensor<[16,32],ui8> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<w_zp> : tensor<ui8>} : () -> !torch.vtensor<[],ui8> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<b_scale> : tensor<32xf32>} : () -> !torch.vtensor<[32],f32> 
    %none = torch.constant.none
    %3:3 = torch.operator "onnx.DynamicQuantizeLinear"(%arg0) : (!torch.vtensor<[4,16],f32>) -> (!torch.vtensor<[4,16],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) 
    %4 = torch.operator "onnx.MatMulInteger"(%3#0, %0, %3#2, %1) : (!torch.vtensor<[4,16],ui8>, !torch.vtensor<[16,32],ui8>, !torch.vtensor<[],ui8>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[4,32],si32> 
    %5 = torch.operator "onnx.Cast"(%4) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[4,32],si32>) -> !torch.vtensor<[4,32],f32> 
    %6 = torch.operator "onnx.Mul"(%3#1, %2) : (!torch.vtensor<[],f32>, !torch.vtensor<[32],f32>) -> !torch.vtensor<[32],f32> 
    %7 = torch.operator "onnx.Mul"(%5, %6) : (!torch.vtensor<[4,32],f32>, !torch.vtensor<[32],f32>) -> !torch.vtensor<[4,32],f32> 
    return %7 : !torch.vtensor<[4,32],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      W: "0x08000000D6421B4C69D073175599D0BAFE30E10E8E4633A84E8F4226BF6EADABF16C38A2EFF7DEAE61640A2F55589482B1E4E0C6FA51E8EC397891B1B31B7A1AF53372E285AE7FD998A474689784C89778DC36707CE4709D13D4717F18B1A256C4850E374219800985B3947444E552D57162B1F9C9971BC4C9685C32FB2B7E2EC99A281C1805DBD561194773897D9E9EA08172EFD1C00E93E19E31818CF68C3A1CB0FC8E180AC84B7BED40C81503084B7B029CD3811CD20E82FB5C7240513D0CBD63FF5DC886A001DE25F535C470C54D429D0049ACE877F64A0FAE35F990F6C5B010D62F16741CABFAE7EDDD3ECB400DC5FACF9DF5161B41129E0A62837298CDE3D2C08B97CA7867D4F9619A0BF7CF0B60E1A28F1DB6A0317A8C8E4A821B950005E8D2AB7A346E42307786D1891E1EF725F1A23E6EA4995C63B2A116B878D89688A062B14DE4B43DE127EB635E9105F5D9B519BD60F9C6443342956C734BDFA611F300272C84D3AD157E8FE939BC6FE1CF1C44322D10D7D46FF6C103ADCAAFBA8BE8F64545B63B3C9A1328BEA40AB9B38A48317C2DE4E53EFE77805AEB303B79792EEECCAA9DEB483BF90F5812ABE6BBD843180307E43AF4242B45278D6E4998D0FDDDF6D4FC573ED199966F693B2010302B545BBA162A26850BFB9F397E283E8A86EC6CDDA8DEEE3D5ACDA529384DF0A79ED128F841F228A52DD74EB0D25A75AB468A6B",
      w_zp: "0x0800000091",
      b_scale: "0x0800000096DB513C37BF583CFC4E5B3C62C9533C7EC3493C88125C3C2959553C78284D3C7449583C69A6523CCB0A4D3C1B67503CC7CC553C8642503CDD5A4C3CA6A64A3CA749513C5D39583CD279503CB1CD513C01B5593CC067563C43575D3CA4535A3C9B745C3C6FC8533C441C593C9EAA5D3CFAC0493CA4FB523C3D83583C502E4E3C"
    }
  }
#-}

