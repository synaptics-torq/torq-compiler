"""
Configuration for known failing Keras tests.

When a test starts passing, remove it from the appropriate list.
When adding new expected failures, add them to the appropriate list with a reason.
"""

# Test cases that should be completely skipped (not run at all)
SKIP_TESTS = {
    # Format: "test_name": "reason for skipping"
    
    # Multiply/Sub operation errors - Only tests that actually fail
    "model051_mult_inp1x4x4x1_int8": "FAILED - AssertionError: Number of differences: 1 out of 16",
    "model133_conv3x3_mult_inp1x3x3x1_int8": "FAILED - AssertionError: Number of differences: 1 out of 9",
    "model138_mult_mult_inp1x3x3x1_int8": "ERROR - AttributeError: 'Functional' object has no attribute '_get_save_spec'",
    "model138_mult_mult_inp1x3x3x1_int16": "ERROR - AttributeError: 'Functional' object has no attribute '_get_save_spec'",
    "model154_sub_s2v_1x6x8x4_int16": "ERROR - subprocess.CalledProcessError in torq-compile",
    "model155_sub_v2s_1x6x8x4_int16": "ERROR - subprocess.CalledProcessError in torq-compile",
    "model158_mul_s2v_1x6x8x4_int16": "FAILED - AssertionError: Output is 0 always",
    "model159_mul_v2s_1x6x8x4_int16": "FAILED - AssertionError: Output is 0 always",
    "model472_mult16x8_neg_s2v_int16": "FAILED - AssertionError: Output is 0 always",
    "model473_mult16x8_pos_v2s_int16": "FAILED - AssertionError: Output is 0 always",
    
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
    
    # Hangs/timeout issues - avgpool/maxpool
    "model030_avgpool_inp1x10x10x4_pool3x3_stride3x3_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model030_avgpool_inp1x10x10x4_pool3x3_stride3x3_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model031_avgpool_inp1x10x10x4_pool4x4_stride4x4_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model032_avgpool_inp1x10x10x4_pool5x5_stride5x5_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model032_avgpool_inp1x10x10x4_pool5x5_stride5x5_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model033_avgpool_inp1x10x10x4_pool5x5_stride1x1_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model033_avgpool_inp1x10x10x4_pool5x5_stride1x1_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model034_avgpool_inp1x10x10x4_pool1x3_stride1x3_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model034_avgpool_inp1x10x10x4_pool1x3_stride1x3_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model037_avgpool_inp1x10x10x4_pool5x5_stride5x5_same_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model037_avgpool_inp1x10x10x4_pool5x5_stride5x5_same_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model038_avgpool_inp1x10x10x4_pool5x5_stride1x1_same_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model038_avgpool_inp1x10x10x4_pool5x5_stride1x1_same_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model044_maxpool_inp1x10x10x4_pool1x3_stride1x3_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model044_maxpool_inp1x10x10x4_pool1x3_stride1x3_valid_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model045_maxpool_inp1x10x10x4_pool3x3_stride3x3_same_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model045_maxpool_inp1x10x10x4_pool3x3_stride3x3_same_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model049_maxpool_inp1x10x10x4_pool1x3_stride1x1_same_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model049_maxpool_inp1x10x10x4_pool1x3_stride1x1_same_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model055_avgpool_inp1x10x10x4_pool2x2_stride2x2_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model056_avgpool_inp1x100x100x1_pool2x2_stride2x2_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model057_avgpool_inp1x100x100x4_pool2x2_stride2x2_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model058_avgpool_inp1x134x158x5_pool2x2_stride2x2_valid_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model117_avgpool_valid_model70_L1_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model118_avgpool_valid_model70_L3_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model119_avgpool_valid_model70_L6_int8": "Test hangs indefinitely - timeout/infinite loop",
    
    # Wrong result on FPGA - maxpool
    "model040_maxpool_inp1x10x10x4_pool3x3_stride3x3_valid_int8": "Test fails on FPGA",
    "model040_maxpool_inp1x10x10x4_pool3x3_stride3x3_valid_int16": "Test fails on FPGA",
    "model041_maxpool_inp1x10x10x4_pool4x4_stride4x4_valid_int8": "Test fails on FPGA",
    "model041_maxpool_inp1x10x10x4_pool4x4_stride4x4_valid_int16": "Test fails on FPGA",
    "model042_maxpool_inp1x10x10x4_pool5x5_stride5x5_valid_int8": "Test fails on FPGA",
    "model042_maxpool_inp1x10x10x4_pool5x5_stride5x5_valid_int16": "Test fails on FPGA",
    "model043_maxpool_inp1x10x10x4_pool5x5_stride1x1_valid_int8": "Test fails on FPGA",
    "model043_maxpool_inp1x10x10x4_pool5x5_stride1x1_valid_int16": "Test fails on FPGA",
    "model046_maxpool_inp1x10x10x4_pool4x4_stride4x4_same_int8": "Test fails on FPGA",
    "model046_maxpool_inp1x10x10x4_pool4x4_stride4x4_same_int16": "Test fails on FPGA",
    "model047_maxpool_inp1x10x10x4_pool5x5_stride5x5_same_int8": "Test fails on FPGA",
    "model047_maxpool_inp1x10x10x4_pool5x5_stride5x5_same_int16": "Test fails on FPGA",
    "model048_maxpool_inp1x10x10x4_pool5x5_stride1x1_same_int8": "Test fails on FPGA",
    "model048_maxpool_inp1x10x10x4_pool5x5_stride1x1_same_int16": "Test fails on FPGA",
    
    # Tanh/softmax failures
    "model081_tanh_zpm26_int16": "Worker crash - not properly terminated",
    "model089_softmax_inp1x1916x2_int16": "Known failure - softmax model 089 currently unstable",
    
    # Mean operation failures
    "model185_mean_inp_32x1x512_zp_74_scale_3_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model185_mean_inp_32x1x512_zp_74_scale_3_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model187_mean_1x32x512_zp_26_int8": "Test hangs indefinitely - timeout/infinite loop",
    "model187_mean_1x32x512_zp_26_int16": "Test hangs indefinitely - timeout/infinite loop",
    "model188_mean_11x1x1_zp_52_scale_em7_int8": "Worker crash - not properly terminated",
    "model189_mean_1x173x1_zp_minus71_scale19_int16": "Worker crash - not properly terminated",
    
    # Conv transpose errors
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
    "model503_conv_transpose_stride1_ker_1x1x3x1_padValid_int16": "Test ERROR - conv_transpose operation failure",
    "model504_conv_transpose_stride1x2_ker1x1x3x1_padValid_int8": "Test ERROR - conv_transpose operation failure",
    "model504_conv_transpose_stride1x2_ker1x1x3x1_padValid_int16": "Test ERROR - conv_transpose operation failure",
    "model505_conv_transpose_stride1_ker2x1x3x1_padValid_int8": "Test ERROR - conv_transpose operation failure",
    "model505_conv_transpose_stride1_ker2x1x3x1_padValid_int16": "Test ERROR - conv_transpose operation failure",
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
    "model532_convTrans_16x8_inp_1x4x1_ker1x3_stride1_padvalid_int16": "Test ERROR - conv_transpose operation failure",
    "model533_convTrans_16x8_inp_1x4x1_ker1x3_stride2_padvalid_int8": "Test ERROR - conv_transpose operation failure",
    "model533_convTrans_16x8_inp_1x4x1_ker1x3_stride2_padvalid_int16": "Test ERROR - conv_transpose operation failure",
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
    
    # Conv stride errors
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
}

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
