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
    "model472_mult16x8_neg_s2v_int16": "FAILED - AssertionError: Output is 0 always",
    "model473_mult16x8_pos_v2s_int16": "FAILED - AssertionError: Output is 0 always",
    
    # FC int16 tests - segmentation faults
    "fc_in1_out16_relu_out16_relu_out1_int16": "Segmentation fault - crashes Python interpreter",
    "fc_in1_out1_relu6_int16": "Segmentation fault - crashes Python interpreter",
    "fc_in61_out1991_relu_int16": "Segmentation fault - crashes Python interpreter",
    "fc_in1024_out1024_int16": "Segmentation fault - crashes Python interpreter",
    "fc_in1000_out512_relu6_int16": "Segmentation fault - crashes Python interpreter",
    "fc_in2000_out97_relu6_int16": "Segmentation fault - crashes Python interpreter",
    "fc_in700_out500_sigmoid_int16": "Segmentation fault - crashes Python interpreter",
    "fc_in1000_out1000_softmax_int16": "Segmentation fault - crashes Python interpreter",
    "fc_in3_out7_relu_int16": "Segmentation fault - crashes Python interpreter",
    
    # Hangs/timeout issues - avgpool/maxpool
    "avgpool_k3x3_s3x3_valid_10x10x4_int8": "Test hangs indefinitely - timeout/infinite loop",
    "avgpool_k3x3_s3x3_valid_10x10x4_int16": "Test hangs indefinitely - timeout/infinite loop",
    "avgpool_k4x4_s4x4_valid_10x10x4_int8": "Test hangs indefinitely - timeout/infinite loop",
    "avgpool_k5x5_s5x5_valid_10x10x4_int8": "Test hangs indefinitely - timeout/infinite loop",
    "avgpool_k5x5_s5x5_valid_10x10x4_int16": "Test hangs indefinitely - timeout/infinite loop",
    "avgpool_k1x3_s1x3_valid_10x10x4_int8": "Test hangs indefinitely - timeout/infinite loop",
    "avgpool_k1x3_s1x3_valid_10x10x4_int16": "Test hangs indefinitely - timeout/infinite loop",
    "avgpool_k5x5_s5x5_same_10x10x4_int8": "Test hangs indefinitely - timeout/infinite loop",
    "avgpool_k5x5_s5x5_same_10x10x4_int16": "Test hangs indefinitely - timeout/infinite loop",
    "maxpool_k1x3_s1x3_valid_10x10x4_int8": "Test hangs indefinitely - timeout/infinite loop",
    "maxpool_k1x3_s1x3_valid_10x10x4_int16": "Test hangs indefinitely - timeout/infinite loop",
    "maxpool_k3x3_s3x3_same_10x10x4_int8": "Test hangs indefinitely - timeout/infinite loop",
    "maxpool_k3x3_s3x3_same_10x10x4_int16": "Test hangs indefinitely - timeout/infinite loop",
    "maxpool_k1x3_s1x3_same_10x10x4_int8": "Test hangs indefinitely - timeout/infinite loop",
    "maxpool_k1x3_s1x3_same_10x10x4_int16": "Test hangs indefinitely - timeout/infinite loop",
    "avgpool_k2x2_s2x2_valid_10x10x4_int8": "Test hangs indefinitely - timeout/infinite loop",
    "avgpool_k2x2_s2x2_valid_100x100x1_int8": "Test hangs indefinitely - timeout/infinite loop",
    "avgpool_k2x2_s2x2_valid_100x100x4_int8": "Test hangs indefinitely - timeout/infinite loop",
    "avgpool_k2x2_s2x2_valid_134x158x5_int8": "Test hangs indefinitely - timeout/infinite loop",
    "avgpool_k2x2_s2x2_valid_65x77x10_int8": "Test hangs indefinitely - timeout/infinite loop",
    "avgpool_k2x2_s2x2_valid_30x36x15_int8": "Test hangs indefinitely - timeout/infinite loop",
    
    # Wrong result on FPGA - maxpool
    "maxpool_k3x3_s3x3_valid_10x10x4_int8": "Test fails on FPGA",
    "maxpool_k3x3_s3x3_valid_10x10x4_int16": "Test fails on FPGA",
    "maxpool_k4x4_s4x4_valid_10x10x4_int8": "Test fails on FPGA",
    "maxpool_k4x4_s4x4_valid_10x10x4_int16": "Test fails on FPGA",
    "maxpool_k5x5_s5x5_valid_10x10x4_int8": "Test fails on FPGA",
    "maxpool_k5x5_s5x5_valid_10x10x4_int16": "Test fails on FPGA",
    "maxpool_k4x4_s4x4_same_10x10x4_int8": "Test fails on FPGA",
    "maxpool_k4x4_s4x4_same_10x10x4_int16": "Test fails on FPGA",
    "maxpool_k5x5_s5x5_same_10x10x4_int8": "Test fails on FPGA",
    "maxpool_k5x5_s5x5_same_10x10x4_int16": "Test fails on FPGA",
    
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
    "conv2d_transpose_k1x3_s1x2_same_1x32x48_oc48_nobias_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x2_same_1x32x48_oc48_nobias_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x64x32_oc32_nobias_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x64x32_oc32_nobias_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x4x1_oc1_nobias_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x4x1_oc1_nobias_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x2_same_1x4x1_oc1_nobias_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x2_same_1x4x1_oc1_nobias_int16": "Test ERROR - conv_transpose operation failure",
    "conv1d_transpose_k3_s2_same_12x4_oc4_nobias_int8": "Test ERROR - conv_transpose operation failure",
    "conv1d_transpose_k3_s2_same_12x4_oc4_nobias_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x4x2_oc2_nobias_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x4x2_oc2_nobias_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x4x1_oc1_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x4x1_oc1_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x2_same_1x4x1_oc1_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x2_same_1x4x1_oc1_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x4x1_oc2_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x4x1_oc2_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_valid_1x4x1_oc1_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x2_valid_1x4x1_oc1_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x2_valid_1x4x1_oc1_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_valid_1x4x1_oc2_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_valid_1x4x1_oc2_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k3x3_s1x1_same_3x3x1_oc1_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k3x3_s2x2_same_3x3x1_oc1_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k3x3_s2x2_same_3x3x1_oc1_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k2x2_s1x1_same_4x4x2_oc3_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k2x2_s1x1_same_4x4x2_oc3_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k4x4_s1x1_valid_5x5x3_oc2_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k4x4_s1x1_valid_5x5x3_oc2_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k4x3_s2x2_same_7x7x2_oc3_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k4x3_s2x2_same_7x7x2_oc3_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x4x16_oc16_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x4x16_oc16_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_valid_1x4x16_oc16_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_valid_1x4x16_oc16_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x2_same_1x4x8_oc8_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x2_same_1x4x8_oc8_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x2_valid_1x4x8_oc8_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x2_valid_1x4x8_oc8_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x4x2_oc1_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x4x2_oc1_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x2_same_1x4x2_oc1_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x2_same_1x4x2_oc1_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_valid_1x4x2_oc1_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_valid_1x4x2_oc1_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x2_valid_1x4x2_oc1_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x2_valid_1x4x2_oc1_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k3x3_s1x1_valid_3x3x1_oc1_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k3x3_s1x1_valid_3x3x1_oc1_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k3x3_s2x2_valid_3x3x1_oc1_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k3x3_s2x2_valid_3x3x1_oc1_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k3x3_s2x2_valid_4x4x2_oc3_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k3x3_s2x2_valid_4x4x2_oc3_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k3x3_s3x3_same_3x3x1_oc1_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k3x3_s3x3_same_3x3x1_oc1_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k3x3_s3x3_valid_3x3x1_oc1_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k3x3_s3x3_valid_3x3x1_oc1_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x64x32_oc32_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k1x3_s1x1_same_1x64x32_oc32_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k4x4_s1x1_valid_5x5x3_oc23_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k4x4_s1x1_valid_5x5x3_oc23_int16": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k2x4_s2x4_valid_11x10x7_oc11_int8": "Test ERROR - conv_transpose operation failure",
    "conv2d_transpose_k2x4_s2x4_valid_11x10x7_oc11_int16": "Test ERROR - conv_transpose operation failure",
    
    # Conv stride errors
    "conv2d_k5x4_s4x4_valid_41x19x5_oc32_int8": "Test ERROR - conv operation failure",
    "conv2d_k5x4_s4x4_valid_41x19x5_oc32_int16": "Test ERROR - conv operation failure",
    "conv2d_k5x3_s3x3_same_16x55x7_oc13_int8": "Test ERROR - conv operation failure",
    "conv2d_k5x3_s3x3_same_16x55x7_oc13_int16": "Test ERROR - conv operation failure",
    "conv2d_k5x4_s3x3_valid_34x6x2_oc26_int8": "Test ERROR - conv operation failure",
    "conv2d_k5x4_s3x3_valid_34x6x2_oc26_int16": "Test ERROR - conv operation failure",
    "conv2d_k4x4_s3x3_same_45x22x27_oc8_int8": "Test ERROR - conv operation failure",
    "conv2d_k4x4_s3x3_same_45x22x27_oc8_int16": "Test ERROR - conv operation failure",
    "conv2d_k3x5_s3x3_valid_15x13x10_oc17_int8": "Test ERROR - conv operation failure",
    "conv2d_k3x5_s3x3_valid_15x13x10_oc17_int16": "Test ERROR - conv operation failure",
    "conv2d_k3x3_s3x3_same_43x61x20_oc26_int8": "Test ERROR - conv operation failure",
    "conv2d_k3x3_s3x3_same_43x61x20_oc26_int16": "Test ERROR - conv operation failure",
    
    # Depthwise int16 errors
    "dw_k3x3_s1x1_valid_6x42x16_relu_int16": "Test ERROR - depthwise conv int16 failure",
    "dw_k3x3_s1x1_valid_6x4x16_relu_int16": "Test ERROR - depthwise conv int16 failure",
    "dw_k3x3_s1x1_valid_6x6x2_relu_int16": "Test ERROR - depthwise conv int16 failure",
    "dw_k3x3_s1x1_same_6x6x2_relu_int16": "Test ERROR - depthwise conv int16 failure",
    "dw_k3x3_s1x1_valid_13x15x8_relu_int16": "Test ERROR - depthwise conv int16 failure",
    "dw_k12x12_s1x1_valid_12x12x45_int16": "Test ERROR - depthwise conv int16 failure",
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
