// Nested N tiling (tile loop x slicing forall) with a non-splat zero-point; strict tolerance so a wrong bias column fails.
// TORQ_FP_MAX_TOL: 0.1
// TORQ_ALLOWED_WRONG: 0.1
module {
  func.func @main(%arg0: !torch.vtensor<[2048,128],si8>, %arg1: !torch.vtensor<[1,1,2048],bf16>) -> !torch.vtensor<[1,1,128],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.0> : tensor<2x128xbf16>} : () -> !torch.vtensor<[2,128],bf16> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<dequant_matmul_zp_nested_tile_zero_points> : tensor<2x128xsi8>} : () -> !torch.vtensor<[2,128],si8> 
    %none = torch.constant.none
    %2 = torch.operator "onnx.DequantizeLinear"(%arg0, %0, %1) {torch.onnx.axis = 0 : si64, torch.onnx.block_size = 1024 : si64} : (!torch.vtensor<[2048,128],si8>, !torch.vtensor<[2,128],bf16>, !torch.vtensor<[2,128],si8>) -> !torch.vtensor<[2048,128],bf16> 
    %3 = torch.operator "onnx.MatMul"(%arg1, %2) : (!torch.vtensor<[1,1,2048],bf16>, !torch.vtensor<[2048,128],bf16>) -> !torch.vtensor<[1,1,128],bf16> 
    return %3 : !torch.vtensor<[1,1,128],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      dequant_matmul_zp_nested_tile_zero_points: "0x0800000059192550103743C9A7D8D54B539D0041B63CB3FA40D8E0D32CCF63F5FC01110B02643E3B2819E062F9C745BC4817B3A4F5A3B8035FF93E54411AF403D1FFE8CD639EAFC25E274DC42CE6FE9C184221BB07D15D4CC10258462A1CA431FBAECD082E02174B25E41D14B1A720E914DCC9BAE840ECE80F60EA12F415F51C09235CBA0CF4E3CCA6EC42AFED5E5BC79D239ED8FF4BAE2107B64545FD5915515D0ED4B90DC23256CD0BA5C0E74D5E1C110EA8E761EEC6CC2BA35E4C0EFA350AFBDCF533A7A12CE63DA252B4EF5EB1202EF2AB05F34B25E1A712D42549E3510406353A5264BAEE57D99D0F33313E1DB746F0463F5F9E4A1AFD3B5B03AD2DCAC9C2C352E4"
    }
  }
#-}

