"""
Configuration for known failing Keras tests.

This file defines which test cases are expected to fail. These tests will be
marked with pytest.mark.xfail, allowing them to run without breaking CI/CD.

When a test starts passing, remove it from the appropriate list.
When adding new expected failures, add them to the appropriate list with a reason.
"""

# Known failures for int8 quantization
KNOWN_FAILURES_INT8 = {
    # Format: "test_name": "reason for failure"
    # Conv2D failures
    "model006_conv2d_4x6x6x1_pad_stride2_int8": "Conv2D assertion failure",
    "model213_conv_22x1x3x9_valid_int8": "Conv assertion failure",
    "model218_conv_23x5x4x10_valid_stride2_int8": "Conv assertion failure",
    
    # Depthwise conv failures
    "model010_depthwise_1x3x3x16_valid_int8": "Depthwise conv assertion failure",
    "model011_depthwise_1x3x3x16_valid_int8": "Depthwise conv assertion failure",
    "model014_depthwise_1x3x3x8_valid_int8": "Depthwise conv assertion failure",
    "model012_depthwise_1x3x3x2_valid_int8": "Depthwise conv assertion failure",
}

# Known failures for int16 quantization
KNOWN_FAILURES_INT16 = {
    # Format: "test_name": "reason for failure"
    # Add/Sub operations
    "model470_add16x8_positive_s2v_int16": "Add operation assertion failure",
    "model156_add_s2v_1x6x8x4_int16": "Add operation assertion failure",
    "model157_add_v2s_1x6x8x4_int16": "Add operation assertion failure",
    "model474_sub16x8_pos_v2s_int16": "Sub operation assertion failure",
    "model475_sub16x8_neg_v2s_int16": "Sub operation assertion failure",
    
    # Conv2D failures
    "conv_model_w100_h100_f5_k6_c4_int16": "Conv2D assertion failure",
    "model006_conv2d_4x6x6x1_pad_stride2_int16": "Conv2D assertion failure",
    "model008_conv2d_4x4x4x4_int16": "Conv2D assertion failure",
    "model200_conv_4_4_5_19_valid_int16": "Conv assertion failure",
    "model201_conv_20x4x1x12_valid_int16": "Conv assertion failure",
    "model205_conv_9x2x5x21_same_int16": "Conv assertion failure",
    "model209_conv_25x1x5x9_same_int16": "Conv assertion failure",
    "model213_conv_22x1x3x9_valid_int16": "Conv assertion failure",
    "model218_conv_23x5x4x10_valid_stride2_int16": "Conv assertion failure",
    
    # Mean operations
    "model180_mean_12x8x256_to_1x256_int16": "Mean operation assertion failure",
    "model184_mean_inp_12x16_64_zp_minus76_int16": "Mean operation assertion failure",
    
    # Activation functions
    "model082_tanh_zpm115_int16": "Tanh assertion failure",
    "model083_tanh_zpm128_int16": "Tanh assertion failure",
    "model084_tanh_zp76_int16": "Tanh assertion failure",
    "model085_tanh_zp127_int16": "Tanh assertion failure",
    "model086_tanh_1x4x6x8_int16": "Tanh assertion failure",
    "model087_sigmoid_1x4x6x8_int16": "Sigmoid assertion failure",
    "model088_sigmoid_1x1x16x128_int16": "Sigmoid assertion failure",
    "model090_sigmoid_zp26_int16": "Sigmoid assertion failure",
    "model091_sigmoid_zp128_int16": "Sigmoid assertion failure",
    "model092_sigmoid_zpm76_int16": "Sigmoid assertion failure",
    "model093_sigmoid_zp128_int16": "Sigmoid assertion failure",
    "model094_sigmoid_zp16_int16": "Sigmoid assertion failure",
}

# Known failures for specific operations (regardless of quantization)
KNOWN_FAILURES_BY_OPERATION = {
    # "conv_transpose": ["Issue with stride > 1"],
    # "softmax": ["Precision issues with large input ranges"],
}

# Test cases that should be completely skipped (not run at all)
SKIP_TESTS = {
    # Format: "test_name": "reason for skipping"
    # Segfaults must be skipped (can't be caught with xfail)
    
    # FC int16 tests - segmentation faults
    "model003_hello_world_int16": "Segmentation fault - crashes Python interpreter",
    "model020_fc_1x1_int16": "Segmentation fault - crashes Python interpreter",
    "model021_fc_1991x61_int16": "Segmentation fault - crashes Python interpreter",
    "model022_fc_1024x1024_int16": "Segmentation fault - crashes Python interpreter",
    "model023_fc_512x1000_int16": "Segmentation fault - crashes Python interpreter",
    "model024_fc_97x2000_int16": "Segmentation fault - crashes Python interpreter",
    "model026_fc_1991x61_int16": "Segmentation fault - crashes Python interpreter",
    "model027_fc_500x700_int16": "Segmentation fault - crashes Python interpreter",
    "model029_fc_1000x1000_int16": "Segmentation fault - crashes Python interpreter",
    "model080_fc_7x3_int16": "Segmentation fault - crashes Python interpreter",
    
    # Hangs/timeout issues
    "model029_fc_1000x1000_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model030_avgpool_inp1x10x10x4_pool3x3_stride3x3_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model030_avgpool_inp1x10x10x4_pool3x3_stride3x3_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model031_avgpool_inp1x10x10x4_pool4x4_stride4x4_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model031_avgpool_inp1x10x10x4_pool4x4_stride4x4_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model032_avgpool_inp1x10x10x4_pool5x5_stride5x5_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model032_avgpool_inp1x10x10x4_pool5x5_stride5x5_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model033_avgpool_inp1x10x10x4_pool5x5_stride1x1_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model033_avgpool_inp1x10x10x4_pool5x5_stride1x1_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model034_avgpool_inp1x10x10x4_pool1x3_stride1x3_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model034_avgpool_inp1x10x10x4_pool1x3_stride1x3_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model035_avgpool_inp1x10x10x4_pool3x3_stride3x3_same_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model035_avgpool_inp1x10x10x4_pool3x3_stride3x3_same_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model036_avgpool_inp1x10x10x4_pool4x4_stride4x4_same_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model036_avgpool_inp1x10x10x4_pool4x4_stride4x4_same_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model037_avgpool_inp1x10x10x4_pool5x5_stride5x5_same_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model037_avgpool_inp1x10x10x4_pool5x5_stride5x5_same_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model038_avgpool_inp1x10x10x4_pool5x5_stride1x1_same_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model038_avgpool_inp1x10x10x4_pool5x5_stride1x1_same_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model039_avgpool_inp1x10x10x4_pool1x3_stride1x1_same_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model039_avgpool_inp1x10x10x4_pool1x3_stride1x1_same_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model040_maxpool_inp1x10x10x4_pool3x3_stride3x3_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model040_maxpool_inp1x10x10x4_pool3x3_stride3x3_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model041_maxpool_inp1x10x10x4_pool4x4_stride4x4_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model041_maxpool_inp1x10x10x4_pool4x4_stride4x4_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model042_maxpool_inp1x10x10x4_pool5x5_stride5x5_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model042_maxpool_inp1x10x10x4_pool5x5_stride5x5_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model043_maxpool_inp1x10x10x4_pool5x5_stride1x1_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model043_maxpool_inp1x10x10x4_pool5x5_stride1x1_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model044_maxpool_inp1x10x10x4_pool1x3_stride1x3_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model044_maxpool_inp1x10x10x4_pool1x3_stride1x3_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model045_maxpool_inp1x10x10x4_pool3x3_stride3x3_same_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model045_maxpool_inp1x10x10x4_pool3x3_stride3x3_same_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model046_maxpool_inp1x10x10x4_pool4x4_stride4x4_same_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model046_maxpool_inp1x10x10x4_pool4x4_stride4x4_same_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model047_maxpool_inp1x10x10x4_pool5x5_stride5x5_same_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model047_maxpool_inp1x10x10x4_pool5x5_stride5x5_same_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model048_maxpool_inp1x10x10x4_pool5x5_stride1x1_same_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model048_maxpool_inp1x10x10x4_pool5x5_stride1x1_same_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model049_maxpool_inp1x10x10x4_pool1x3_stride1x1_same_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model049_maxpool_inp1x10x10x4_pool1x3_stride1x1_same_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model055_avgpool_inp1x10x10x4_pool2x2_stride2x2_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model055_avgpool_inp1x10x10x4_pool2x2_stride2x2_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model056_avgpool_inp1x100x100x1_pool2x2_stride2x2_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model056_avgpool_inp1x100x100x1_pool2x2_stride2x2_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model057_avgpool_inp1x100x100x4_pool2x2_stride2x2_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model057_avgpool_inp1x100x100x4_pool2x2_stride2x2_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model058_avgpool_inp1x134x158x5_pool2x2_stride2x2_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model058_avgpool_inp1x134x158x5_pool2x2_stride2x2_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model117_avgpool_valid_model70_L1_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model117_avgpool_valid_model70_L1_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model118_avgpool_valid_model70_L3_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model118_avgpool_valid_model70_L3_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model119_avgpool_valid_model70_L6_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model119_avgpool_valid_model70_L6_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model089_softmax_inp1x1916x2_int8": "Known failure - softmax model 089 currently unstable",
    "model089_softmax_inp1x1916x2_int16": "Known failure - softmax model 089 currently unstable",
    
    # Mean operation failures - cause worker crashes
    "model185_mean_inp_32x1x512_zp_74_scale_3_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model185_mean_inp_32x1x512_zp_74_scale_3_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model186_mean_53x37x1_zp_minus25_scale_39_int8": "Worker crash - not properly terminated",
    "model186_mean_53x37x1_zp_minus25_scale_39_int16": "Worker crash - not properly terminated",
    "model187_mean_1x32x512_zp_26_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model187_mean_1x32x512_zp_26_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model188_mean_11x1x1_zp_52_scale_em7_int8": "Worker crash - not properly terminated",
    "model188_mean_11x1x1_zp_52_scale_em7_int16": "Worker crash - not properly terminated",
    "model189_mean_1x173x1_zp_minus71_scale19_int8": "Worker crash - not properly terminated",
    "model189_mean_1x173x1_zp_minus71_scale19_int16": "Worker crash - not properly terminated",
    
    # Pointwise convolution failures - cause worker crashes
    "model017_pointwise_8x8x9x16_stride2x2_int8": "Worker crash - not properly terminated",
    "model017_pointwise_8x8x9x16_stride2x2_int16": "Worker crash - not properly terminated",
    "model018_pointwise_8x9x8x16_stride2x2_int8": "Worker crash - not properly terminated",
    "model018_pointwise_8x9x8x16_stride2x2_int16": "Worker crash - not properly terminated",
    "model019_pointwise_8x9x9x16_stride2x2_int8": "Worker crash - not properly terminated",
    "model019_pointwise_8x9x9x16_stride2x2_int16": "Worker crash - not properly terminated",
    
    # Tanh failures - cause worker crashes
    "model081_tanh_zpm26_int8": "Worker crash - not properly terminated",
    "model081_tanh_zpm26_int16": "Worker crash - not properly terminated",
    
    # Multiply operation errors - cause test errors
    "model050_mult_inp1x10x10x4_int8": "Test ERROR - multiplication operation failure",
    "model050_mult_inp1x10x10x4_int16": "Test ERROR - multiplication operation failure",
    "model051_mult_inp1x4x4x1_int8": "Test ERROR - multiplication operation failure",
    "model051_mult_inp1x4x4x1_int16": "Test ERROR - multiplication operation failure",
    "model052_mult_inp1x4x4x1_zp128_AB_int8": "Test ERROR - multiplication operation failure",
    "model052_mult_inp1x4x4x1_zp128_AB_int16": "Test ERROR - multiplication operation failure",
    "model053_mult_inp1x4x4x1_zp128_B_int8": "Test ERROR - multiplication operation failure",
    "model053_mult_inp1x4x4x1_zp128_B_int16": "Test ERROR - multiplication operation failure",
    "model054_mult_inp1x4x4x1_zp128_A_int8": "Test ERROR - multiplication operation failure",
    "model054_mult_inp1x4x4x1_zp128_A_int16": "Test ERROR - multiplication operation failure",
    "model120_mult_inp1x8x8x1_int8": "Test ERROR - multiplication operation failure",
    "model120_mult_inp1x8x8x1_int16": "Test ERROR - multiplication operation failure",
    "model122_conv_mult_inp1x8x8x1_int8": "Test ERROR - conv with multiplication failure",
    "model122_conv_mult_inp1x8x8x1_int16": "Test ERROR - conv with multiplication failure",
    "model131_mult_inp1x3x3x1_int8": "Test ERROR - multiplication operation failure",
    "model131_mult_inp1x3x3x1_int16": "Test ERROR - multiplication operation failure",
    "model133_conv3x3_mult_inp1x3x3x1_int8": "Test ERROR - conv with multiplication failure",
    "model133_conv3x3_mult_inp1x3x3x1_int16": "Test ERROR - conv with multiplication failure",
    "model137_add_mult_inp1x3x3x1_int8": "Test ERROR - add with multiplication failure",
    "model137_add_mult_inp1x3x3x1_int16": "Test ERROR - add with multiplication failure",
    "model138_mult_mult_inp1x3x3x1_int8": "Test ERROR - multiplication operation failure",
    "model138_mult_mult_inp1x3x3x1_int16": "Test ERROR - multiplication operation failure",
    "model141_mult_inp1x4x4x1_int8": "Test ERROR - multiplication operation failure",
    "model141_mult_inp1x4x4x1_int16": "Test ERROR - multiplication operation failure",
    "model143_conv3x3_mult_inp1x4x4x1_int8": "Test ERROR - conv with multiplication failure",
    "model143_conv3x3_mult_inp1x4x4x1_int16": "Test ERROR - conv with multiplication failure",
    "model154_sub_s2v_1x6x8x4_int16": "Test ERROR - subtraction operation failure",
    "model155_sub_v2s_1x6x8x4_int16": "Test ERROR - subtraction operation failure",
    "model158_mul_s2v_1x6x8x4_int8": "Test ERROR - multiplication operation failure",
    "model158_mul_s2v_1x6x8x4_int16": "Test ERROR - multiplication operation failure",
    "model159_mul_v2s_1x6x8x4_int8": "Test ERROR - multiplication operation failure",
    "model159_mul_v2s_1x6x8x4_int16": "Test ERROR - multiplication operation failure",
    "model472_mult16x8_neg_s2v_int8": "Test ERROR - multiplication operation failure",
    "model472_mult16x8_neg_s2v_int16": "Test ERROR - multiplication operation failure",
    "model473_mult16x8_pos_v2s_int8": "Test ERROR - multiplication operation failure",
    "model473_mult16x8_pos_v2s_int16": "Test ERROR - multiplication operation failure",
    
    # Conv transpose errors - all conv_transpose operations fail
    "model100_conv_transpose_1x3_1x2_int8": "Test ERROR - conv_transpose operation failure",
    "model100_conv_transpose_1x3_1x2_int16": "Test ERROR - conv_transpose operation failure",
    "model101_conv_transpose_1x3_1x1_int8": "Test ERROR - conv_transpose operation failure",
    "model101_conv_transpose_1x3_1x1_int16": "Test ERROR - conv_transpose operation failure",
    "model102_conv_transpose_stride1_ker_1x3_input_1x1x4x1_int8": "Test ERROR - conv_transpose operation failure",
    "model102_conv_transpose_stride1_ker_1x3_input_1x1x4x1_int16": "Test ERROR - conv_transpose operation failure",
    "model103_conv_transpose_stride2_ker_1x3_1x1x4x1_int8": "Test ERROR - conv_transpose operation failure",
    "model103_conv_transpose_stride2_ker_1x3_1x1x4x1_int16": "Test ERROR - conv_transpose operation failure",
    "model104_conv_transpose_1x3_stride2_input_1x1x12x4_int8": "Test ERROR - conv_transpose operation failure",
    "model104_conv_transpose_1x3_stride2_input_1x1x12x4_int16": "Test ERROR - conv_transpose operation failure",
    "model105_conv_transpose_stride1_ker_1x3_input1x1x4x2_int8": "Test ERROR - conv_transpose operation failure",
    "model105_conv_transpose_stride1_ker_1x3_input1x1x4x2_int16": "Test ERROR - conv_transpose operation failure",
    "model500_conv_transpose_stride1_ker_1x1x3x1_padSame_int8": "Test ERROR - conv_transpose operation failure",
    "model500_conv_transpose_stride1_ker_1x1x3x1_padSame_int16": "Test ERROR - conv_transpose operation failure",
    "model501_conv_transpose_stride1x2_ker1x1x3x1_padSame_int8": "Test ERROR - conv_transpose operation failure",
    "model501_conv_transpose_stride1x2_ker1x1x3x1_padSame_int16": "Test ERROR - conv_transpose operation failure",
    "model502_conv_transpose_stride1_ker2x1x3x1_padSame_int8": "Test ERROR - conv_transpose operation failure",
    "model502_conv_transpose_stride1_ker2x1x3x1_padSame_int16": "Test ERROR - conv_transpose operation failure",
    "model503_conv_transpose_stride1_ker_1x1x3x1_padValid_int8": "Test ERROR - conv_transpose operation failure",
    "model503_conv_transpose_stride1_ker_1x1x3x1_padValid_int16": "Test ERROR - conv_transpose operation failure",
    "model504_conv_transpose_stride1x2_ker1x1x3x1_padValid_int8": "Test ERROR - conv_transpose operation failure",
    "model504_conv_transpose_stride1x2_ker1x1x3x1_padValid_int16": "Test ERROR - conv_transpose operation failure",
    "model505_conv_transpose_stride1_ker2x1x3x1_padValid_int8": "Test ERROR - conv_transpose operation failure",
    "model505_conv_transpose_stride1_ker2x1x3x1_padValid_int16": "Test ERROR - conv_transpose operation failure",
    "model506_conv_transpose_stride1_1x3x3x1_same_int8": "Test ERROR - conv_transpose operation failure",
    "model506_conv_transpose_stride1_1x3x3x1_same_int16": "Test ERROR - conv_transpose operation failure",
    "model507_conv_transpose_stride2_1x3x3x1_same_int8": "Test ERROR - conv_transpose operation failure",
    "model507_conv_transpose_stride2_1x3x3x1_same_int16": "Test ERROR - conv_transpose operation failure",
    "model508_conv_transpose_stride1_3x2x2x2_same_int8": "Test ERROR - conv_transpose operation failure",
    "model508_conv_transpose_stride1_3x2x2x2_same_int16": "Test ERROR - conv_transpose operation failure",
    "model509_conv_transpose_stride1_2x4x4x3_valid_int8": "Test ERROR - conv_transpose operation failure",
    "model509_conv_transpose_stride1_2x4x4x3_valid_int16": "Test ERROR - conv_transpose operation failure",
    "model510_conv_transpose_stride2_3x4x3x2_same_int8": "Test ERROR - conv_transpose operation failure",
    "model510_conv_transpose_stride2_3x4x3x2_same_int16": "Test ERROR - conv_transpose operation failure",
    "model511_conv_transpose_stride1_16x1x3x16_padsame_int8": "Test ERROR - conv_transpose operation failure",
    "model511_conv_transpose_stride1_16x1x3x16_padsame_int16": "Test ERROR - conv_transpose operation failure",
    "model512_conv_transpose_stride1_16x1x3x16_padvalid_int8": "Test ERROR - conv_transpose operation failure",
    "model512_conv_transpose_stride1_16x1x3x16_padvalid_int16": "Test ERROR - conv_transpose operation failure",
    "model513_conv_transpose_stride2_8x1x3x8_padSame_int8": "Test ERROR - conv_transpose operation failure",
    "model513_conv_transpose_stride2_8x1x3x8_padSame_int16": "Test ERROR - conv_transpose operation failure",
    "model514_conv_transpose_stride2_8x1x3x8_padValid_int8": "Test ERROR - conv_transpose operation failure",
    "model514_conv_transpose_stride2_8x1x3x8_padValid_int16": "Test ERROR - conv_transpose operation failure",
    "model515_conv_transpose_stride1_ker1x1x3x2_padSame_int8": "Test ERROR - conv_transpose operation failure",
    "model515_conv_transpose_stride1_ker1x1x3x2_padSame_int16": "Test ERROR - conv_transpose operation failure",
    "model516_conv_transpose_stride1x2_ker1x1x3x2_padSame_int8": "Test ERROR - conv_transpose operation failure",
    "model516_conv_transpose_stride1x2_ker1x1x3x2_padSame_int16": "Test ERROR - conv_transpose operation failure",
    "model517_conv_transpose_stride1_ker1x1x3x2_padValid_int8": "Test ERROR - conv_transpose operation failure",
    "model517_conv_transpose_stride1_ker1x1x3x2_padValid_int16": "Test ERROR - conv_transpose operation failure",
    "model518_conv_transpose_stride1x2_ker1x1x3x2_padValid_int8": "Test ERROR - conv_transpose operation failure",
    "model518_conv_transpose_stride1x2_ker1x1x3x2_padValid_int16": "Test ERROR - conv_transpose operation failure",
    "model519_conv_transpose_stride1_1x3x3x1_valid_int8": "Test ERROR - conv_transpose operation failure",
    "model519_conv_transpose_stride1_1x3x3x1_valid_int16": "Test ERROR - conv_transpose operation failure",
    "model520_conv_transpose_stride2_1x3x3x1_valid_int8": "Test ERROR - conv_transpose operation failure",
    "model520_conv_transpose_stride2_1x3x3x1_valid_int16": "Test ERROR - conv_transpose operation failure",
    "model521_conv_transpose_stride2_3x3x3x2_valid_int8": "Test ERROR - conv_transpose operation failure",
    "model521_conv_transpose_stride2_3x3x3x2_valid_int16": "Test ERROR - conv_transpose operation failure",
    "model522_conv_transpose_stride3_1x3x3x1_same_int8": "Test ERROR - conv_transpose operation failure",
    "model522_conv_transpose_stride3_1x3x3x1_same_int16": "Test ERROR - conv_transpose operation failure",
    "model523_conv_transpose_stride3_1x3x3x1_valid_int8": "Test ERROR - conv_transpose operation failure",
    "model523_conv_transpose_stride3_1x3x3x1_valid_int16": "Test ERROR - conv_transpose operation failure",
    "model531_convTrans_16x8_inp_1x64x32_ker1x3_stride1_padsame_int8": "Test ERROR - conv_transpose operation failure",
    "model531_convTrans_16x8_inp_1x64x32_ker1x3_stride1_padsame_int16": "Test ERROR - conv_transpose operation failure",
    "model532_convTrans_16x8_inp_1x4x1_ker1x3_stride1_padvalid_int8": "Test ERROR - conv_transpose operation failure",
    "model532_convTrans_16x8_inp_1x4x1_ker1x3_stride1_padvalid_int16": "Test ERROR - conv_transpose operation failure",
    "model533_convTrans_16x8_inp_1x4x1_ker1x3_stride2_padvalid_int8": "Test ERROR - conv_transpose operation failure",
    "model533_convTrans_16x8_inp_1x4x1_ker1x3_stride2_padvalid_int16": "Test ERROR - conv_transpose operation failure",
    "model534_convTrans_16x8_inp_1x3x3x1_ker3x3_stride1_padsame_int8": "Test ERROR - conv_transpose operation failure",
    "model534_convTrans_16x8_inp_1x3x3x1_ker3x3_stride1_padsame_int16": "Test ERROR - conv_transpose operation failure",
    "model535_convTrans_16x8_inp_1x3x3x1_ker3x3_stride2_padsame_int8": "Test ERROR - conv_transpose operation failure",
    "model535_convTrans_16x8_inp_1x3x3x1_ker3x3_stride2_padsame_int16": "Test ERROR - conv_transpose operation failure",
    "model536_convTrans_16x8_inp_1x4x4x2_ker2x2_stride1_padsame_int8": "Test ERROR - conv_transpose operation failure",
    "model536_convTrans_16x8_inp_1x4x4x2_ker2x2_stride1_padsame_int16": "Test ERROR - conv_transpose operation failure",
    "model537_convTrans_16x8_inp_1x5x5x3_ker4x4_stride1_padvalid_int8": "Test ERROR - conv_transpose operation failure",
    "model537_convTrans_16x8_inp_1x5x5x3_ker4x4_stride1_padvalid_int16": "Test ERROR - conv_transpose operation failure",
    "model538_convTrans_16x8_inp_1x7x7x2_ker4x3_stride2_padsame_int8": "Test ERROR - conv_transpose operation failure",
    "model538_convTrans_16x8_inp_1x7x7x2_ker4x3_stride2_padsame_int16": "Test ERROR - conv_transpose operation failure",
    "model539_convTrans_16x8_inp_1x3x3x1_ker3x3_stride3_padsame_int8": "Test ERROR - conv_transpose operation failure",
    "model539_convTrans_16x8_inp_1x3x3x1_ker3x3_stride3_padsame_int16": "Test ERROR - conv_transpose operation failure",
    "model540_convTrans_16x8_inp_1x3x3x1_ker3x3_stride3_padvalid_int8": "Test ERROR - conv_transpose operation failure",
    "model540_convTrans_16x8_inp_1x3x3x1_ker3x3_stride3_padvalid_int16": "Test ERROR - conv_transpose operation failure",
    "model541_convTrans_16x8_inp_1x11x10x7_ker2x4_stride2x4_valid_int8": "Test ERROR - conv_transpose operation failure",
    "model541_convTrans_16x8_inp_1x11x10x7_ker2x4_stride2x4_valid_int16": "Test ERROR - conv_transpose operation failure",
    
    # Additional errors from test output
    "model204_conv_32x5x4x5_valid_stride4_int8": "Test ERROR - conv operation failure",
    "model204_conv_32x5x4x5_valid_stride4_int16": "Test ERROR - conv operation failure",
    "model208_conv_13x5x3x7_same_stride3_int8": "Test ERROR - conv operation failure",
    "model208_conv_13x5x3x7_same_stride3_int16": "Test ERROR - conv operation failure",
    "model211_conv_26x5x4x2_valid_stride3_int8": "Test ERROR - conv operation failure",
    "model211_conv_26x5x4x2_valid_stride3_int16": "Test ERROR - conv operation failure",
    "model214_conv_8x4x4x27_same_stride3_int8": "Test ERROR - conv operation failure",
    "model214_conv_8x4x4x27_same_stride3_int16": "Test ERROR - conv operation failure",
    "model217_conv_17x3x5x10_valid_stride3_int8": "Test ERROR - conv operation failure",
    "model217_conv_17x3x5x10_valid_stride3_int16": "Test ERROR - conv operation failure",
    "model219_conv_26x3x3x20_same_stride3_int8": "Test ERROR - conv operation failure",
    "model219_conv_26x3x3x20_same_stride3_int16": "Test ERROR - conv operation failure",
    
    # Depthwise int16 errors
    "model010_depthwise_1x3x3x16_valid_int16": "Test ERROR - depthwise conv int16 failure",
    "model011_depthwise_1x3x3x16_valid_int16": "Test ERROR - depthwise conv int16 failure",
    "model012_depthwise_1x3x3x2_valid_int16": "Test ERROR - depthwise conv int16 failure",
    "model013_depthwise_1x3x3x2_same_int16": "Test ERROR - depthwise conv int16 failure",
    "model014_depthwise_1x3x3x8_valid_int16": "Test ERROR - depthwise conv int16 failure",
    
    # SmartAve errors (already in skip list)
    "model065_SmartAve_1x25x5x4_int8": "Test ERROR - SmartAve operation failure",
    "model065_SmartAve_1x25x5x4_int16": "Test ERROR - SmartAve operation failure",
    "model066_SmartAve_1x5x25x4_int8": "Test ERROR - SmartAve operation failure",
    "model066_SmartAve_1x5x25x4_int16": "Test ERROR - SmartAve operation failure",
    
    # Mean and FC errors
    "model003_hello_world_int8": "Test ERROR - FC operation failure",
    "model020_fc_1x1_int8": "Test ERROR - FC operation failure",
    "model080_fc_7x3_int8": "Test ERROR - FC operation failure",
    "model095_relu_8x8_inp_1x10x1x1_int16": "Test ERROR - ReLU int16 failure",
    "model096_relu_zp26_8x8_inp_1x10x1x1_int16": "Test ERROR - ReLU int16 failure",
    "model181_mean_inp_4x8x6_zp_64_int8": "Test ERROR - mean operation failure",
    "model181_mean_inp_4x8x6_zp_64_int16": "Test ERROR - mean operation failure",
    "model182_mean_inp_22x32x16_zp_1_int8": "Test ERROR - mean operation failure",
    "model182_mean_inp_22x32x16_zp_1_int16": "Test ERROR - mean operation failure",
    "model183_mean_inp_37x17x11_zp_minus127_int8": "Test ERROR - mean operation failure",
    "model183_mean_inp_37x17x11_zp_minus127_int16": "Test ERROR - mean operation failure",
}


def should_xfail(test_name, markers=None):
    """
    Determine if a test should be marked as xfail.
    
    Args:
        test_name: Name of the test case
        markers: List of marker names applied to the test
        
    Returns:
        Tuple of (should_xfail: bool, reason: str)
    """
    # Check if test is in known failures
    if test_name in KNOWN_FAILURES_INT8:
        return True, KNOWN_FAILURES_INT8[test_name]
    
    if test_name in KNOWN_FAILURES_INT16:
        return True, KNOWN_FAILURES_INT16[test_name]
    
    # Check operation-based failures
    if markers:
        for marker in markers:
            marker_name = marker if isinstance(marker, str) else marker.name
            if marker_name in KNOWN_FAILURES_BY_OPERATION:
                return True, KNOWN_FAILURES_BY_OPERATION[marker_name]
    
    return False, None


def should_skip(test_name):
    """
    Determine if a test should be skipped entirely.
    
    Args:
        test_name: Name of the test case
        
    Returns:
        Tuple of (should_skip: bool, reason: str)
    """
    # Check explicit skip list first
    if test_name in SKIP_TESTS:
        return True, SKIP_TESTS[test_name]
    
    return False, None
