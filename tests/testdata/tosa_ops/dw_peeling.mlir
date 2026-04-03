module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<1x56x56x11xi8> {ml_program.identifier = "serving_default_input_0:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x56x56x11xi8> {ml_program.identifier = "StatefulPartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tosa.const"() <{values = dense<0> : tensor<11xi32>}> : () -> tensor<11xi32>
    %cst = arith.constant dense<-128> : tensor<1xi8>
    %cst_0 = arith.constant dense<0> : tensor<1xi8>
    %1 = "tosa.const"() <{values = dense<"0x98C9E2BA5A357F7FE6BD4C0C4A7AAE69F4664C4DD6B4D0418CBA26561EF7F59BD4A4645217234AA9EA774677B3B21E5E36CC8AD3337901630D776090F3A12AC4BE9A82BD4E90A4A1AE1B45CA27C8EE968B269120E74B6095D8E985DD475F7B92156CDC57C6CAC3CEBEC9F9FD0F6DFE181CBE467FB7E97F85C8E17FB8E40A7820CB374A6232DF55897F162BE2B2348BD18EFE6A41B1A38B9CDCD4D9F07F84E3B5170AE32EBB7F4FB2E75519B03B18813D121CD8DA9CDDC1AAA54583A804D6B08D97BA34A5B1A999ACA921DF62BF8A3C717F7D3E50E4D629076AE5F3B3903A4090B8A006051E6D23B42E438432C9FF53F7EDCAB7936FB3BDFE5D5DBB0564F546057F937D35B9ADE6D254B2CA0A761A5B1FA6AEC6"> : tensor<5x5x11x1xi8>}> : () -> tensor<5x5x11x1xi8>
    %2 = tosa.depthwise_conv2d %arg0, %1, %0, %cst, %cst_0 {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>} : (tensor<1x56x56x11xi8>, tensor<5x5x11x1xi8>, tensor<11xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x56x56x11xi32>
    %cst_1 = arith.constant dense<[1494728868, 1389415548, 1444318144, 1493774144, 1527403457, 1473327430, 1415957845, 1467841227, 1530001498, 1507764475, 1513213326]> : tensor<11xi32>
    %cst_2 = arith.constant dense<41> : tensor<11xi8>
    %cst_3 = arith.constant dense<0> : tensor<1xi32>
    %cst_4 = arith.constant dense<28> : tensor<1xi8>
    %3 = tosa.rescale %2, %cst_1, %cst_2, %cst_3, %cst_4 {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x56x56x11xi32>, tensor<11xi32>, tensor<11xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x56x56x11xi8>
    return %3 : tensor<1x56x56x11xi8>
  }
}

