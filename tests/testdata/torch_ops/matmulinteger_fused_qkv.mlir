// TORQ_USE_ABS_TOL_GATE: 1
// TORQ_FP_ABS_TOL_FRAC: 0.015
// Both fixtures inherit the same DQL quantization noise as a single
// MatMulInteger chain, but three weight-column sets now draw on it instead
// of one, so the worst branch measures somewhat higher.
module {
  func.func @matmulinteger_fused_qkv(%arg0: !torch.vtensor<[1,16],f32>) -> (!torch.vtensor<[1,32],f32>, !torch.vtensor<[1,32],f32>, !torch.vtensor<[1,32],f32>) attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<W_qkv> : tensor<16x96xsi8>} : () -> !torch.vtensor<[16,96],si8> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<w_zp> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<s_q> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<e_q> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<ax_q> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %5 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<b_scale_q> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %6 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<s_k> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %7 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<e_k> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %8 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<ax_k> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %9 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<b_scale_k> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %10 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<s_v> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %11 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<e_v> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %12 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<ax_v> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %13 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<b_scale_v> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %none = torch.constant.none
    %14:3 = torch.operator "onnx.DynamicQuantizeLinear"(%arg0) : (!torch.vtensor<[1,16],f32>) -> (!torch.vtensor<[1,16],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) 
    %15 = torch.operator "onnx.MatMulInteger"(%14#0, %0, %14#2, %1) : (!torch.vtensor<[1,16],ui8>, !torch.vtensor<[16,96],si8>, !torch.vtensor<[],ui8>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,96],si32> 
    %16 = torch.operator "onnx.Slice"(%15, %2, %3, %4) : (!torch.vtensor<[1,96],si32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,32],si32> 
    %17 = torch.operator "onnx.Cast"(%16) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[1,32],si32>) -> !torch.vtensor<[1,32],f32> 
    %18 = torch.operator "onnx.Mul"(%14#1, %5) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %19 = torch.operator "onnx.Mul"(%17, %18) : (!torch.vtensor<[1,32],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[1,32],f32> 
    %20 = torch.operator "onnx.Slice"(%15, %6, %7, %8) : (!torch.vtensor<[1,96],si32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,32],si32> 
    %21 = torch.operator "onnx.Cast"(%20) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[1,32],si32>) -> !torch.vtensor<[1,32],f32> 
    %22 = torch.operator "onnx.Mul"(%14#1, %9) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %23 = torch.operator "onnx.Mul"(%21, %22) : (!torch.vtensor<[1,32],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[1,32],f32> 
    %24 = torch.operator "onnx.Slice"(%15, %10, %11, %12) : (!torch.vtensor<[1,96],si32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,32],si32> 
    %25 = torch.operator "onnx.Cast"(%24) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[1,32],si32>) -> !torch.vtensor<[1,32],f32> 
    %26 = torch.operator "onnx.Mul"(%14#1, %13) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %27 = torch.operator "onnx.Mul"(%25, %26) : (!torch.vtensor<[1,32],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[1,32],f32> 
    return %19, %23, %27 : !torch.vtensor<[1,32],f32>, !torch.vtensor<[1,32],f32>, !torch.vtensor<[1,32],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      W_qkv: "0x08000000F903407289A55271C0D05DECC653C2E8240C96885C40550950D4F349A0CEA0F479A3E2E766B400C3863F90C8FFFC9E7A3F759839CB0A6BC739A9D277EB04CB9EEC1FF446DD1C4569ED8B37075EF5DE90F5234459B7174DC3D85614022C027A408EA60B51922E41485EB10E4CDBB1FA95B85A2A5B565FCFF81DC66A825525C137E9557EC8F8B73023574D797564A78BFBD8644BEC13166387FD2CF46A7253F6619328C5BF2D4462B65E54D0904552F5AAA6E041D189303EAE0FE5008222C30DEB1B9BDD2143E187390127A9EE615DD121964EC5D7760B5FB2417E94BEA6C2D09365C26A42AC3245A1A1E092EBFA2912F4B815F456323912DDC8F212DEA69C8CB4E8C9ADD036D06A13A4776045A94932413618D76A7A30DB000F9497FD0FB7C3A26901C64874CB65444A0668A69776FDE73ACB22584DA0773B32B06BE470BB985615E47878831F4E31190508CFA5E5316F75B4887C2B414FDC5C2324E1D8E136008685B1FE197746C9033F61F136B6F566E28549CEF07EE3C3BC58B71ABD4D3721EDDD7242E487A1F2F5DF7AFA4AA1E0B9A50FA4E32F4A0F1ACB5BAF3B9719C0CAD84749C009946375DD0A09459207571C0E896BB0D92CF61116A984724EA82B023CA533377EC7EEA3848CF3AD3DB11F09D4F37E74C773B84BD82B2657E16F60869A9E25DCA798691989C327C443CA21994E3D0926911BF48988EE382FADA8C0E2A386DD95B3B7E2EA8EF66761A4D1278612524890C69800754B4040D679A2A3E3F6D7F85FF7EB2295636C861F7D9EA79D6AF774982F21091DA1891D4E6548F669892B03305DAA4687F591BF75E825737197D97D40EB9125AB67C7F20CFD0E4DFFD6EC4A13F676C1F57455138F0EE2760F121EE8C002E65371F725B215A891B58E5CB669A42B7A9D81C9DDFC8F9223978CE292A695FCC65413D54DB3C4A4C8E952DC3E57A1394DDA544DAEAA20EBB278BEACFEDD05D3FAF70A68B7EA4667C77268C70336CEB9ADF7FC38E0FD1F3DFFAF8A0054128E9B53CF4FAB6BAD29F2A9D6F1BAF09F2173E1F32C6EB4A8DA740B91ED78A0BB768E30085469DBCD711A4F8879EEB326FA3BE3CB1C6AC0219AA9FA902314E23F7B6BE81ACD974FE7F7E9C633C98671347554257CC7A9352FB87BD2790AAFE6A2DA0D7876AC2C1CE08A0D9850B6467CFD39E95D328DE12EFEF0F2EA0734C9CFFF03FDC336E41A084BA9A9C777EB88F9C74C86248A0F4C5D9FB3A99B87E5F5A3560E4A120AA2E436BF0E24EC7E6A1E5A4AB9BEAB9D69E6A8CB41DBD014DCEB0DDA6C31814FAA0937BFE566CA9F75E4C46C363675944219351538384DD7C5FA20D64C34622067EE662F99CEE03EF4DC6376EBBEC4C28611CA2047CC8613AA1FCF59082DDDEA6111B59C0F51468D6D335E7BA38D4A772C5AEC64879AABCD3FCF966AD06DC1313E1EDC2497DCDF5CD4AC37F9D25731CE097462503B1EE8AFFCECF8B05EACA4DCEC9F08BCEFA21829FFBAEA592F8FD41D1B873A92A22CD32D70D276157D488BD552476E57668336C02C4C375F13EF49B80078BA30987C67104011AEED51C5CD3E22E8DB4CB7D6ADCA93C79311928297A9543004DAA10606020A09FF9AB595EF175EE7E27801E16D22BC5339A6FB0A48EADC160A0FDEB35D60699221D27A243A1E53B764E1C6E57BA8E458FFB0AE71513CD68A305AB89CDAC1E2E7BD77893944EE727DBAB3AA9BD9B896092651DEC70F0467025B976BE86E9F160D00BF8AA39B1806E85AE0EE8E8171B6AF4254A90CB314C9D41B6E5585B86419C90778F188144F4F33B8FBFF3A999A039249FA7EBCF962CCC01421DF519E2B066E4B566387799DDEC1C09A9DC6EFBF4D87BC7758613506B44C216151646D82AA10515D4661BE53CB8C7410DCE0CA3D37DEA2D2FB4DDBFE0BA51AED233DEEDC624055396E5EF26F3A45EEE6C793267071E74D30C919BA684428346D5CF5A66A5B15EEB5C66DD8AF7D61730E966AD148377B89748AA88CB95D74D52FD1644A00D1A43C12DE63CC8BE301ABBA94755EFC5D64F6532F405CC3E40B3529403A94D7A47A3F02349EB02752324CE6D502D7B99D53F2E79C760B161FA21425933E1E5F40D9A55A18425118B2FF69B277A4351B5EC1C6B229C66CC98CC251AFBDA34C2A24D84CB1E6B4F3E66B0093BBA82278B668",
      w_zp: "0x0800000000",
      s_q: "0x080000000000000000000000",
      e_q: "0x080000002000000000000000",
      ax_q: "0x080000000100000000000000",
      b_scale_q: "0x08000000F085493C",
      s_k: "0x080000002000000000000000",
      e_k: "0x080000004000000000000000",
      ax_k: "0x080000000100000000000000",
      b_scale_k: "0x08000000EFAC5D3C",
      s_v: "0x080000004000000000000000",
      e_v: "0x080000006000000000000000",
      ax_v: "0x080000000100000000000000",
      b_scale_v: "0x08000000EDD3713C"
    }
  }
#-}

