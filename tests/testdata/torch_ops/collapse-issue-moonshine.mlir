module {
  func.func @part73_graph(%arg0: !torch.vtensor<[1,1,8,36],bf16>, %arg1: !torch.vtensor<[1,1,8,36],bf16>, %arg2: !torch.vtensor<[1,1,8,36],bf16>, %arg3: !torch.vtensor<[1,1,1,32],bf16>, %arg4: !torch.vtensor<[1,1,32],bf16>) -> (!torch.vtensor<[1,1,8,32],bf16>, !torch.vtensor<[1,8,1,4],bf16>, !torch.vtensor<[1,8,1,32],bf16>, !torch.vtensor<[1,8,1,4],bf16>, !torch.vtensor<[1,8,1,32],bf16>, !torch.vtensor<[1,8,30,36],bf16>, !torch.vtensor<[1,1,1,32],bf16>, !torch.vtensor<[1,1,1,32],bf16>) attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.opset_versions = {ai.onnx.ml = 5 : si64, ai.onnx.preview.training = 1 : si64, ai.onnx.training = 1 : si64, com.microsoft = 1 : si64, com.microsoft.experimental = 1 : si64, com.microsoft.nchwc = 1 : si64, org.pytorch.aten = 1 : si64}, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__model_decoder_layers.0_self_attn_Constant_10_output_0_part73_init0> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__model_decoder_layers.0_self_attn_Constant_12_output_0_part73_init1> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__model_decoder_layers.0_self_attn_Constant_16_output_0_part73_init2> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__model_decoder_layers.0_self_attn_Transpose_1_output_0_bcast_shape_part73_init3> : tensor<4xsi64>} : () -> !torch.vtensor<[4],si64> 
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__model_decoder_rotary_emb_Constant_6_output_0_part73_init4> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %5 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<const_transpose_optimizer_part73_init5> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %6 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<const_transpose_optimizer_token_68_part73_init6> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %7 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_onnx__Unsqueeze_180_part73_init7> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %8 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__model_decoder_layers.0_self_attn_Constant_10_output_0_part73_init8> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %9 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__model_decoder_layers.0_self_attn_Constant_12_output_0_part73_init9> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %10 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__model_decoder_layers.0_self_attn_Constant_16_output_0_part73_init10> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %11 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__model_decoder_layers.0_self_attn_Transpose_1_output_0_bcast_shape_part73_init11> : tensor<4xsi64>} : () -> !torch.vtensor<[4],si64> 
    %12 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__model_decoder_rotary_emb_Constant_6_output_0_part73_init12> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %13 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<const_transpose_optimizer_part73_init13> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %14 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<const_transpose_optimizer_token_68_part73_init14> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %15 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_onnx__Unsqueeze_180_part73_init15> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %none = torch.constant.none
    %16 = torch.operator "onnx.Unsqueeze"(%arg4, %12) : (!torch.vtensor<[1,1,32],bf16>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,1,1,32],bf16> 
    %17 = torch.operator "onnx.Slice"(%arg2, %9, %10, %14, %12) : (!torch.vtensor<[1,1,8,36],bf16>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,1,8,4],bf16> 
    %18 = torch.operator "onnx.Slice"(%arg2, %15, %9, %13, %12) : (!torch.vtensor<[1,1,8,36],bf16>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,1,8,32],bf16> 
    %19 = torch.operator "onnx.Transpose"(%arg1) {torch.onnx.perm = [0 : si64, 2 : si64, 1 : si64, 3 : si64]} : (!torch.vtensor<[1,1,8,36],bf16>) -> !torch.vtensor<[1,8,1,36],bf16> 
    %20 = torch.operator "onnx.Transpose"(%arg0) {torch.onnx.perm = [0 : si64, 2 : si64, 1 : si64, 3 : si64]} : (!torch.vtensor<[1,1,8,36],bf16>) -> !torch.vtensor<[1,8,1,36],bf16> 
    %21 = torch.operator "onnx.Transpose"(%arg3) {torch.onnx.perm = [0 : si64, 2 : si64, 1 : si64, 3 : si64]} : (!torch.vtensor<[1,1,1,32],bf16>) -> !torch.vtensor<[1,1,1,32],bf16> 
    %22 = torch.operator "onnx.Transpose"(%17) {torch.onnx.perm = [0 : si64, 2 : si64, 1 : si64, 3 : si64]} : (!torch.vtensor<[1,1,8,4],bf16>) -> !torch.vtensor<[1,8,1,4],bf16> 
    %23 = torch.operator "onnx.Transpose"(%18) {torch.onnx.perm = [0 : si64, 2 : si64, 1 : si64, 3 : si64]} : (!torch.vtensor<[1,1,8,32],bf16>) -> !torch.vtensor<[1,8,1,32],bf16> 
    %24 = torch.operator "onnx.Slice"(%19, %15, %9, %8, %12) : (!torch.vtensor<[1,8,1,36],bf16>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,8,1,32],bf16> 
    %25 = torch.operator "onnx.Slice"(%19, %9, %10, %8, %12) : (!torch.vtensor<[1,8,1,36],bf16>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,8,1,4],bf16> 
    %26 = torch.operator "onnx.Expand"(%20, %11) : (!torch.vtensor<[1,8,1,36],bf16>, !torch.vtensor<[4],si64>) -> !torch.vtensor<[1,8,30,36],bf16> 
    %27 = torch.operator "onnx.Mul"(%18, %21) : (!torch.vtensor<[1,1,8,32],bf16>, !torch.vtensor<[1,1,1,32],bf16>) -> !torch.vtensor<[1,1,8,32],bf16> 
    return %27, %25, %23, %22, %24, %26, %16, %21 : !torch.vtensor<[1,1,8,32],bf16>, !torch.vtensor<[1,8,1,4],bf16>, !torch.vtensor<[1,8,1,32],bf16>, !torch.vtensor<[1,8,1,4],bf16>, !torch.vtensor<[1,8,1,32],bf16>, !torch.vtensor<[1,8,30,36],bf16>, !torch.vtensor<[1,1,1,32],bf16>, !torch.vtensor<[1,1,1,32],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      __model_decoder_layers.0_self_attn_Constant_10_output_0_part73_init0: "0x080000000300000000000000",
      __model_decoder_layers.0_self_attn_Constant_12_output_0_part73_init1: "0x080000002000000000000000",
      __model_decoder_layers.0_self_attn_Constant_16_output_0_part73_init2: "0x08000000FFFFFFFFFFFFFF7F",
      __model_decoder_layers.0_self_attn_Transpose_1_output_0_bcast_shape_part73_init3: "0x08000000010000000000000008000000000000001E000000000000002400000000000000",
      __model_decoder_rotary_emb_Constant_6_output_0_part73_init4: "0x080000000100000000000000",
      const_transpose_optimizer_part73_init5: "0x080000000300000000000000",
      const_transpose_optimizer_token_68_part73_init6: "0x080000000300000000000000",
      _onnx__Unsqueeze_180_part73_init7: "0x080000000000000000000000",
      __model_decoder_layers.0_self_attn_Constant_10_output_0_part73_init8: "0x080000000300000000000000",
      __model_decoder_layers.0_self_attn_Constant_12_output_0_part73_init9: "0x080000002000000000000000",
      __model_decoder_layers.0_self_attn_Constant_16_output_0_part73_init10: "0x08000000FFFFFFFFFFFFFF7F",
      __model_decoder_layers.0_self_attn_Transpose_1_output_0_bcast_shape_part73_init11: "0x08000000010000000000000008000000000000001E000000000000002400000000000000",
      __model_decoder_rotary_emb_Constant_6_output_0_part73_init12: "0x080000000100000000000000",
      const_transpose_optimizer_part73_init13: "0x080000000300000000000000",
      const_transpose_optimizer_token_68_part73_init14: "0x080000000300000000000000",
      _onnx__Unsqueeze_180_part73_init15: "0x080000000000000000000000"
    }
  }
#-}

