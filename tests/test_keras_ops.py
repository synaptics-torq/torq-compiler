import pytest

from .models.keras_models import *
from .models.keras_add_mul_sub import *
from .models.keras_conv import *
from .models.keras_conv_transpose import *
from .models.keras_depthwise import *
from .models.keras_fc import *
from .models.keras_mean import *
from .models.keras_pointwise import *
from .models.keras_pooling import *
from .models.keras_sig_tanh_relu import *
from .models.keras_softmax import *
from .models.keras_transpose import *
from torq.testing.comparison import compare_test_results
from torq.testing.cases import Case
from .keras_known_failures import should_xfail, should_skip



def _add_quantization_and_markers(cases, marker_name):
    """Helper to add quantization variants with markers to test cases."""
    result = []
    for case in cases:
        for quantize_to_int16 in [False, True]:
            quant_suffix = "_int16" if quantize_to_int16 else "_int8"
            new_data = case.data.copy()
            new_data["quantize_to_int16"] = quantize_to_int16
            
            # Create marks
            quant_mark = pytest.mark.int16 if quantize_to_int16 else pytest.mark.int8
            op_mark = getattr(pytest.mark, marker_name)
            
            # Create Case and wrap with pytest.param including marks
            new_case = Case(case.name + quant_suffix, new_data)
            
            # Check if this test should be xfailed or skipped
            marks = [op_mark, quant_mark]
            test_name = case.name + quant_suffix
            
            # Check for skip first
            skip, skip_reason = should_skip(test_name)
            if skip:
                marks.append(pytest.mark.skip(reason=skip_reason))
            else:
                # Check for xfail
                xfail, xfail_reason = should_xfail(test_name, [marker_name, quant_mark.name])
                if xfail:
                    marks.append(pytest.mark.xfail(reason=xfail_reason, strict=False))
            
            result.append(pytest.param(new_case, marks=marks))
    
    return result


def get_test_cases():
    test_cases = []

    for params in conv_model_params:
        test_cases.append(Case("conv_model_" + params.idfn(), {
            "keras_model_name": "conv_model",
            "keras_model_params": params
        }))

    for quantize_mode in [False, True]:
        test_cases.append(Case("transpose_conv_model_quantized_" + ("i16" if quantize_mode else "i8"), {
            "keras_model_name": "transpose_conv_model",
            "quantize_to_int16": quantize_mode
        }))

    # Integrate all Keras test cases from different modules
    test_cases.extend(_add_quantization_and_markers(get_keras_add_mul_sub_test_cases(), "add_mul_sub"))
    test_cases.extend(_add_quantization_and_markers(get_keras_conv_transpose_test_cases(), "conv_transpose"))
    test_cases.extend(_add_quantization_and_markers(get_keras_conv_test_cases(), "conv"))
    test_cases.extend(_add_quantization_and_markers(get_keras_depthwise_test_cases(), "depthwise"))
    test_cases.extend(_add_quantization_and_markers(get_keras_fc_test_cases(), "fc"))
    test_cases.extend(_add_quantization_and_markers(get_keras_mean_test_cases(), "mean"))
    test_cases.extend(_add_quantization_and_markers(get_keras_pointwise_test_cases(), "pointwise"))
    test_cases.extend(_add_quantization_and_markers(get_keras_pooling_test_cases(), "pooling"))
    test_cases.extend(_add_quantization_and_markers(get_keras_sig_tanh_relu_test_cases(), "activation"))
    test_cases.extend(_add_quantization_and_markers(get_keras_softmax_test_cases(), "softmax"))
    test_cases.extend(_add_quantization_and_markers(get_keras_transpose_test_cases(), "transpose"))

    return test_cases


def get_keras_add_mul_sub_test_cases():
    test_cases = []

    # Add model050_mult_inp1x10x10x4 test case
    test_cases.append(Case("model050_mult_inp1x10x10x4", {
        "keras_model_name": "model050_mult_inp1x10x10x4"
    }))

    # Add model051_mult_inp1x4x4x1 test case
    test_cases.append(Case("model051_mult_inp1x4x4x1", {
        "keras_model_name": "model051_mult_inp1x4x4x1"
    }))

    # Add model052_mult_inp1x4x4x1_zp128_AB test case
    test_cases.append(Case("model052_mult_inp1x4x4x1_zp128_AB", {
        "keras_model_name": "model052_mult_inp1x4x4x1_zp128_AB"
    }))

    # Add model053_mult_inp1x4x4x1_zp128_B test case
    test_cases.append(Case("model053_mult_inp1x4x4x1_zp128_B", {
        "keras_model_name": "model053_mult_inp1x4x4x1_zp128_B"
    }))

    # Add model054_mult_inp1x4x4x1_zp128_A test case
    test_cases.append(Case("model054_mult_inp1x4x4x1_zp128_A", {
        "keras_model_name": "model054_mult_inp1x4x4x1_zp128_A"
    }))

    # Add model060_add_inp1x4x4x1 test case
    test_cases.append(Case("model060_add_inp1x4x4x1", {
        "keras_model_name": "model060_add_inp1x4x4x1"
    }))

    # Add model061_add_inp1x4x4x1_zp128_x1 test case
    test_cases.append(Case("model061_add_inp1x4x4x1_zp128_x1", {
        "keras_model_name": "model061_add_inp1x4x4x1_zp128_x1"
    }))

    # Add model062_add_inp1x4x4x1_zp128_x2 test case
    test_cases.append(Case("model062_add_inp1x4x4x1_zp128_x2", {
        "keras_model_name": "model062_add_inp1x4x4x1_zp128_x2"
    }))

    # Add model120_mult_inp1x8x8x1 test case
    test_cases.append(Case("model120_mult_inp1x8x8x1", {
        "keras_model_name": "model120_mult_inp1x8x8x1"
    }))

    # Add model121_add_inp1x8x8x1 test case
    test_cases.append(Case("model121_add_inp1x8x8x1", {
        "keras_model_name": "model121_add_inp1x8x8x1"
    }))

    # Add model122_conv_mult_inp1x8x8x1 test case
    test_cases.append(Case("model122_conv_mult_inp1x8x8x1", {
        "keras_model_name": "model122_conv_mult_inp1x8x8x1"
    }))

    # Add model123_conv_add_inp1x8x8x1 test case
    test_cases.append(Case("model123_conv_add_inp1x8x8x1", {
        "keras_model_name": "model123_conv_add_inp1x8x8x1"
    }))

    # Add model130_conv3x3_inp1x3x3x1 test case
    test_cases.append(Case("model130_conv3x3_inp1x3x3x1", {
        "keras_model_name": "model130_conv3x3_inp1x3x3x1"
    }))

    # Add model131_mult_inp1x3x3x1 test case
    test_cases.append(Case("model131_mult_inp1x3x3x1", {
        "keras_model_name": "model131_mult_inp1x3x3x1"
    }))

    # Add model132_add_inp1x3x3x1 test case
    test_cases.append(Case("model132_add_inp1x3x3x1", {
        "keras_model_name": "model132_add_inp1x3x3x1"
    }))

    # Add model133_conv3x3_mult_inp1x3x3x1 test case
    test_cases.append(Case("model133_conv3x3_mult_inp1x3x3x1", {
        "keras_model_name": "model133_conv3x3_mult_inp1x3x3x1"
    }))

    # Add model134_conv3x3_add_inp1x3x3x1 test case
    test_cases.append(Case("model134_conv3x3_add_inp1x3x3x1", {
        "keras_model_name": "model134_conv3x3_add_inp1x3x3x1"
    }))

    # Add model135_conv3x3_conv2x2_inp1x3x3x1 test case
    test_cases.append(Case("model135_conv3x3_conv2x2_inp1x3x3x1", {
        "keras_model_name": "model135_conv3x3_conv2x2_inp1x3x3x1"
    }))

    # Add model136_conv2x2_inp1x3x3x1 test case
    test_cases.append(Case("model136_conv2x2_inp1x3x3x1", {
        "keras_model_name": "model136_conv2x2_inp1x3x3x1"
    }))

    # Add model137_add_mult_inp1x3x3x1 test case
    test_cases.append(Case("model137_add_mult_inp1x3x3x1", {
        "keras_model_name": "model137_add_mult_inp1x3x3x1"
    }))

    # Add model138_mult_mult_inp1x3x3x1 test case
    test_cases.append(Case("model138_mult_mult_inp1x3x3x1", {
        "keras_model_name": "model138_mult_mult_inp1x3x3x1"
    }))

    # Add model140_conv3x3_inp1x4x4x1 test case
    test_cases.append(Case("model140_conv3x3_inp1x4x4x1", {
        "keras_model_name": "model140_conv3x3_inp1x4x4x1"
    }))

    # Add model141_mult_inp1x4x4x1 test case
    test_cases.append(Case("model141_mult_inp1x4x4x1", {
        "keras_model_name": "model141_mult_inp1x4x4x1"
    }))

    # Add model142_add3x3_inp1x4x4x1 test case
    test_cases.append(Case("model142_add3x3_inp1x4x4x1", {
        "keras_model_name": "model142_add3x3_inp1x4x4x1"
    }))

    # Add model143_conv3x3_mult_inp1x4x4x1 test case
    test_cases.append(Case("model143_conv3x3_mult_inp1x4x4x1", {
        "keras_model_name": "model143_conv3x3_mult_inp1x4x4x1"
    }))

    # Add model144_conv3x3_add_inp1x4x4x1 test case
    test_cases.append(Case("model144_conv3x3_add_inp1x4x4x1", {
        "keras_model_name": "model144_conv3x3_add_inp1x4x4x1"
    }))

    # Add model145_conv3x3_conv2x2_inp1x4x4x1 test case
    test_cases.append(Case("model145_conv3x3_conv2x2_inp1x4x4x1", {
        "keras_model_name": "model145_conv3x3_conv2x2_inp1x4x4x1"
    }))

    # Add model146_conv2x2_inp1x4x4x1 test case
    test_cases.append(Case("model146_conv2x2_inp1x4x4x1", {
        "keras_model_name": "model146_conv2x2_inp1x4x4x1"
    }))

    # Add model147_conv_inp1x32x32x16_16x3x3_same_stride1x1 test case
    test_cases.append(Case("model147_conv_inp1x32x32x16_16x3x3_same_stride1x1", {
        "keras_model_name": "model147_conv_inp1x32x32x16_16x3x3_same_stride1x1"
    }))

    # Add model150_sub_1x6x8x4_zp128_B test case
    test_cases.append(Case("model150_sub_1x6x8x4_zp128_B", {
        "keras_model_name": "model150_sub_1x6x8x4_zp128_B"
    }))

    # Add model151_sub_1x6x8x4_zp128_A test case
    test_cases.append(Case("model151_sub_1x6x8x4_zp128_A", {
        "keras_model_name": "model151_sub_1x6x8x4_zp128_A"
    }))

    # Add model152_sub_1x6x8x4_zp128_AB test case
    test_cases.append(Case("model152_sub_1x6x8x4_zp128_AB", {
        "keras_model_name": "model152_sub_1x6x8x4_zp128_AB"
    }))

    # Add model153_sub_1x6x8x4_zp128_none test case
    test_cases.append(Case("model153_sub_1x6x8x4_zp128_none", {
        "keras_model_name": "model153_sub_1x6x8x4_zp128_none"
    }))

    # Add model154_sub_s2v_1x6x8x4 test case
    test_cases.append(Case("model154_sub_s2v_1x6x8x4", {
        "keras_model_name": "model154_sub_s2v_1x6x8x4"
    }))

    # Add model155_sub_v2s_1x6x8x4 test case
    test_cases.append(Case("model155_sub_v2s_1x6x8x4", {
        "keras_model_name": "model155_sub_v2s_1x6x8x4"
    }))

    # Add model156_add_s2v_1x6x8x4 test case
    test_cases.append(Case("model156_add_s2v_1x6x8x4", {
        "keras_model_name": "model156_add_s2v_1x6x8x4"
    }))

    # Add model157_add_v2s_1x6x8x4 test case
    test_cases.append(Case("model157_add_v2s_1x6x8x4", {
        "keras_model_name": "model157_add_v2s_1x6x8x4"
    }))

    # Add model158_mul_s2v_1x6x8x4 test case
    test_cases.append(Case("model158_mul_s2v_1x6x8x4", {
        "keras_model_name": "model158_mul_s2v_1x6x8x4"
    }))

    # Add model159_mul_v2s_1x6x8x4 test case
    test_cases.append(Case("model159_mul_v2s_1x6x8x4", {
        "keras_model_name": "model159_mul_v2s_1x6x8x4"
    }))

    # Add model470_add16x8_positive_s2v test case
    test_cases.append(Case("model470_add16x8_positive_s2v", {
        "keras_model_name": "model470_add16x8_positive_s2v"
    }))

    # Add model471_add16x8_negative_s2v test case
    test_cases.append(Case("model471_add16x8_negative_s2v", {
        "keras_model_name": "model471_add16x8_negative_s2v"
    }))

    # Add model472_mult16x8_neg_s2v test case
    test_cases.append(Case("model472_mult16x8_neg_s2v", {
        "keras_model_name": "model472_mult16x8_neg_s2v"
    }))

    # Add model473_mult16x8_pos_v2s test case
    test_cases.append(Case("model473_mult16x8_pos_v2s", {
        "keras_model_name": "model473_mult16x8_pos_v2s"
    }))

    # Add model474_sub16x8_pos_v2s test case
    test_cases.append(Case("model474_sub16x8_pos_v2s", {
        "keras_model_name": "model474_sub16x8_pos_v2s"
    }))

    # Add model475_sub16x8_neg_v2s test case
    test_cases.append(Case("model475_sub16x8_neg_v2s", {
        "keras_model_name": "model475_sub16x8_neg_v2s"
    }))

    return test_cases


def get_keras_conv_transpose_test_cases():
    test_cases = []

    # Add model100_conv_transpose_1x3_1x2 test case
    test_cases.append(Case("model100_conv_transpose_1x3_1x2", {
        "keras_model_name": "model100_conv_transpose_1x3_1x2"
    }))

    # Add model101_conv_transpose_1x3_1x1 test case
    test_cases.append(Case("model101_conv_transpose_1x3_1x1", {
        "keras_model_name": "model101_conv_transpose_1x3_1x1"
    }))

    # Add model102_conv_transpose_stride1_ker_1x3_input_1x1x4x1 test case
    test_cases.append(Case("model102_conv_transpose_stride1_ker_1x3_input_1x1x4x1", {
        "keras_model_name": "model102_conv_transpose_stride1_ker_1x3_input_1x1x4x1"
    }))

    # Add model103_conv_transpose_stride2_ker_1x3_1x1x4x1 test case
    test_cases.append(Case("model103_conv_transpose_stride2_ker_1x3_1x1x4x1", {
        "keras_model_name": "model103_conv_transpose_stride2_ker_1x3_1x1x4x1"
    }))

    # Add model104_conv_transpose_1x3_stride2_input_1x1x12x4 test case
    test_cases.append(Case("model104_conv_transpose_1x3_stride2_input_1x1x12x4", {
        "keras_model_name": "model104_conv_transpose_1x3_stride2_input_1x1x12x4"
    }))

    # Add model105_conv_transpose_stride1_ker_1x3_input1x1x4x2 test case
    test_cases.append(Case("model105_conv_transpose_stride1_ker_1x3_input1x1x4x2", {
        "keras_model_name": "model105_conv_transpose_stride1_ker_1x3_input1x1x4x2"
    }))

    # Add model500_conv_transpose_stride1_ker_1x1x3x1_padSame test case
    test_cases.append(Case("model500_conv_transpose_stride1_ker_1x1x3x1_padSame", {
        "keras_model_name": "model500_conv_transpose_stride1_ker_1x1x3x1_padSame"
    }))

    # Add model501_conv_transpose_stride1x2_ker1x1x3x1_padSame test case
    test_cases.append(Case("model501_conv_transpose_stride1x2_ker1x1x3x1_padSame", {
        "keras_model_name": "model501_conv_transpose_stride1x2_ker1x1x3x1_padSame"
    }))

    # Add model502_conv_transpose_stride1_ker2x1x3x1_padSame test case
    test_cases.append(Case("model502_conv_transpose_stride1_ker2x1x3x1_padSame", {
        "keras_model_name": "model502_conv_transpose_stride1_ker2x1x3x1_padSame"
    }))

    # Add model503_conv_transpose_stride1_ker_1x1x3x1_padValid test case
    test_cases.append(Case("model503_conv_transpose_stride1_ker_1x1x3x1_padValid", {
        "keras_model_name": "model503_conv_transpose_stride1_ker_1x1x3x1_padValid"
    }))

    # Add model504_conv_transpose_stride1x2_ker1x1x3x1_padValid test case
    test_cases.append(Case("model504_conv_transpose_stride1x2_ker1x1x3x1_padValid", {
        "keras_model_name": "model504_conv_transpose_stride1x2_ker1x1x3x1_padValid"
    }))

    # Add model505_conv_transpose_stride1_ker2x1x3x1_padValid test case
    test_cases.append(Case("model505_conv_transpose_stride1_ker2x1x3x1_padValid", {
        "keras_model_name": "model505_conv_transpose_stride1_ker2x1x3x1_padValid"
    }))

    # Add model506_conv_transpose_stride1_1x3x3x1_same test case
    test_cases.append(Case("model506_conv_transpose_stride1_1x3x3x1_same", {
        "keras_model_name": "model506_conv_transpose_stride1_1x3x3x1_same"
    }))

    # Add model507_conv_transpose_stride2_1x3x3x1_same test case
    test_cases.append(Case("model507_conv_transpose_stride2_1x3x3x1_same", {
        "keras_model_name": "model507_conv_transpose_stride2_1x3x3x1_same"
    }))

    # Add model508_conv_transpose_stride1_3x2x2x2_same test case
    test_cases.append(Case("model508_conv_transpose_stride1_3x2x2x2_same", {
        "keras_model_name": "model508_conv_transpose_stride1_3x2x2x2_same"
    }))

    # Add model509_conv_transpose_stride1_2x4x4x3_valid test case
    test_cases.append(Case("model509_conv_transpose_stride1_2x4x4x3_valid", {
        "keras_model_name": "model509_conv_transpose_stride1_2x4x4x3_valid"
    }))

    # Add model510_conv_transpose_stride2_3x4x3x2_same test case
    test_cases.append(Case("model510_conv_transpose_stride2_3x4x3x2_same", {
        "keras_model_name": "model510_conv_transpose_stride2_3x4x3x2_same"
    }))

    # Add model511_conv_transpose_stride1_16x1x3x16_padsame test case
    test_cases.append(Case("model511_conv_transpose_stride1_16x1x3x16_padsame", {
        "keras_model_name": "model511_conv_transpose_stride1_16x1x3x16_padsame"
    }))

    # Add model512_conv_transpose_stride1_16x1x3x16_padvalid test case
    test_cases.append(Case("model512_conv_transpose_stride1_16x1x3x16_padvalid", {
        "keras_model_name": "model512_conv_transpose_stride1_16x1x3x16_padvalid"
    }))

    # Add model513_conv_transpose_stride2_8x1x3x8_padSame test case
    test_cases.append(Case("model513_conv_transpose_stride2_8x1x3x8_padSame", {
        "keras_model_name": "model513_conv_transpose_stride2_8x1x3x8_padSame"
    }))

    # Add model514_conv_transpose_stride2_8x1x3x8_padValid test case
    test_cases.append(Case("model514_conv_transpose_stride2_8x1x3x8_padValid", {
        "keras_model_name": "model514_conv_transpose_stride2_8x1x3x8_padValid"
    }))

    # Add model515_conv_transpose_stride1_ker1x1x3x2_padSame test case
    test_cases.append(Case("model515_conv_transpose_stride1_ker1x1x3x2_padSame", {
        "keras_model_name": "model515_conv_transpose_stride1_ker1x1x3x2_padSame"
    }))

    # Add model516_conv_transpose_stride1x2_ker1x1x3x2_padSame test case
    test_cases.append(Case("model516_conv_transpose_stride1x2_ker1x1x3x2_padSame", {
        "keras_model_name": "model516_conv_transpose_stride1x2_ker1x1x3x2_padSame"
    }))

    # Add model517_conv_transpose_stride1_ker1x1x3x2_padValid test case
    test_cases.append(Case("model517_conv_transpose_stride1_ker1x1x3x2_padValid", {
        "keras_model_name": "model517_conv_transpose_stride1_ker1x1x3x2_padValid"
    }))

    # Add model518_conv_transpose_stride1x2_ker1x1x3x2_padValid test case
    test_cases.append(Case("model518_conv_transpose_stride1x2_ker1x1x3x2_padValid", {
        "keras_model_name": "model518_conv_transpose_stride1x2_ker1x1x3x2_padValid"
    }))

    # Add model519_conv_transpose_stride1_1x3x3x1_valid test case
    test_cases.append(Case("model519_conv_transpose_stride1_1x3x3x1_valid", {
        "keras_model_name": "model519_conv_transpose_stride1_1x3x3x1_valid"
    }))

    # Add model520_conv_transpose_stride2_1x3x3x1_valid test case
    test_cases.append(Case("model520_conv_transpose_stride2_1x3x3x1_valid", {
        "keras_model_name": "model520_conv_transpose_stride2_1x3x3x1_valid"
    }))

    # Add model521_conv_transpose_stride2_3x3x3x2_valid test case
    test_cases.append(Case("model521_conv_transpose_stride2_3x3x3x2_valid", {
        "keras_model_name": "model521_conv_transpose_stride2_3x3x3x2_valid"
    }))

    # Add model522_conv_transpose_stride3_1x3x3x1_same test case
    test_cases.append(Case("model522_conv_transpose_stride3_1x3x3x1_same", {
        "keras_model_name": "model522_conv_transpose_stride3_1x3x3x1_same"
    }))

    # Add model523_conv_transpose_stride3_1x3x3x1_valid test case
    test_cases.append(Case("model523_conv_transpose_stride3_1x3x3x1_valid", {
        "keras_model_name": "model523_conv_transpose_stride3_1x3x3x1_valid"
    }))

    # Add model531_convTrans_16x8_inp_1x64x32_ker1x3_stride1_padsame test case
    test_cases.append(Case("model531_convTrans_16x8_inp_1x64x32_ker1x3_stride1_padsame", {
        "keras_model_name": "model531_convTrans_16x8_inp_1x64x32_ker1x3_stride1_padsame"
    }))

    # Add model532_convTrans_16x8_inp_1x4x1_ker1x3_stride1_padvalid test case
    test_cases.append(Case("model532_convTrans_16x8_inp_1x4x1_ker1x3_stride1_padvalid", {
        "keras_model_name": "model532_convTrans_16x8_inp_1x4x1_ker1x3_stride1_padvalid"
    }))

    # Add model533_convTrans_16x8_inp_1x4x1_ker1x3_stride2_padvalid test case
    test_cases.append(Case("model533_convTrans_16x8_inp_1x4x1_ker1x3_stride2_padvalid", {
        "keras_model_name": "model533_convTrans_16x8_inp_1x4x1_ker1x3_stride2_padvalid"
    }))

    # Add model534_convTrans_16x8_inp_1x3x3x1_ker3x3_stride1_padsame test case
    test_cases.append(Case("model534_convTrans_16x8_inp_1x3x3x1_ker3x3_stride1_padsame", {
        "keras_model_name": "model534_convTrans_16x8_inp_1x3x3x1_ker3x3_stride1_padsame"
    }))

    # Add model535_convTrans_16x8_inp_1x3x3x1_ker3x3_stride2_padsame test case
    test_cases.append(Case("model535_convTrans_16x8_inp_1x3x3x1_ker3x3_stride2_padsame", {
        "keras_model_name": "model535_convTrans_16x8_inp_1x3x3x1_ker3x3_stride2_padsame"
    }))

    # Add model536_convTrans_16x8_inp_1x4x4x2_ker2x2_stride1_padsame test case
    test_cases.append(Case("model536_convTrans_16x8_inp_1x4x4x2_ker2x2_stride1_padsame", {
        "keras_model_name": "model536_convTrans_16x8_inp_1x4x4x2_ker2x2_stride1_padsame"
    }))

    # Add model537_convTrans_16x8_inp_1x5x5x3_ker4x4_stride1_padvalid test case
    test_cases.append(Case("model537_convTrans_16x8_inp_1x5x5x3_ker4x4_stride1_padvalid", {
        "keras_model_name": "model537_convTrans_16x8_inp_1x5x5x3_ker4x4_stride1_padvalid"
    }))

    # Add model538_convTrans_16x8_inp_1x7x7x2_ker4x3_stride2_padsame test case
    test_cases.append(Case("model538_convTrans_16x8_inp_1x7x7x2_ker4x3_stride2_padsame", {
        "keras_model_name": "model538_convTrans_16x8_inp_1x7x7x2_ker4x3_stride2_padsame"
    }))

    # Add model539_convTrans_16x8_inp_1x3x3x1_ker3x3_stride3_padsame test case
    test_cases.append(Case("model539_convTrans_16x8_inp_1x3x3x1_ker3x3_stride3_padsame", {
        "keras_model_name": "model539_convTrans_16x8_inp_1x3x3x1_ker3x3_stride3_padsame"
    }))

    # Add model540_convTrans_16x8_inp_1x3x3x1_ker3x3_stride3_padvalid test case
    test_cases.append(Case("model540_convTrans_16x8_inp_1x3x3x1_ker3x3_stride3_padvalid", {
        "keras_model_name": "model540_convTrans_16x8_inp_1x3x3x1_ker3x3_stride3_padvalid"
    }))

    # Add model541_convTrans_16x8_inp_1x11x10x7_ker2x4_stride2x4_valid test case
    test_cases.append(Case("model541_convTrans_16x8_inp_1x11x10x7_ker2x4_stride2x4_valid", {
        "keras_model_name": "model541_convTrans_16x8_inp_1x11x10x7_ker2x4_stride2x4_valid"
    }))

    return test_cases


def get_keras_conv_test_cases():
    test_cases = []

    # Add model001_conv_3X3_small test case
    test_cases.append(Case("model001_conv_3X3_small", {
        "keras_model_name": "model001_conv_3X3_small"
    }))

    # Add model002_conv_2x3_small_valid test case
    test_cases.append(Case("model002_conv_2x3_small_valid", {
        "keras_model_name": "model002_conv_2x3_small_valid"
    }))

    # Add model004_conv_3x3_valid_bias test case
    test_cases.append(Case("model004_conv_3x3_valid_bias", {
        "keras_model_name": "model004_conv_3x3_valid_bias"
    }))

    # Add model005_conv2d_4x3x3x3_padding test case
    test_cases.append(Case("model005_conv2d_4x3x3x3_padding", {
        "keras_model_name": "model005_conv2d_4x3x3x3_padding"
    }))

    # Add model006_conv2d_4x6x6x1_pad_stride2 test case
    test_cases.append(Case("model006_conv2d_4x6x6x1_pad_stride2", {
        "keras_model_name": "model006_conv2d_4x6x6x1_pad_stride2"
    }))

    # Add model007_conv2d_8x3x3x4 test case
    test_cases.append(Case("model007_conv2d_8x3x3x4", {
        "keras_model_name": "model007_conv2d_8x3x3x4"
    }))

    # Add model008_conv2d_4x4x4x4 test case
    test_cases.append(Case("model008_conv2d_4x4x4x4", {
        "keras_model_name": "model008_conv2d_4x4x4x4"
    }))

    # Add model009_conv_3x3_same_bias test case
    test_cases.append(Case("model009_conv_3x3_same_bias", {
        "keras_model_name": "model009_conv_3x3_same_bias"
    }))

    # Add model200_conv_4_4_5_19_valid test case
    test_cases.append(Case("model200_conv_4_4_5_19_valid", {
        "keras_model_name": "model200_conv_4_4_5_19_valid"
    }))

    # Add model201_conv_20x4x1x12_valid test case
    test_cases.append(Case("model201_conv_20x4x1x12_valid", {
        "keras_model_name": "model201_conv_20x4x1x12_valid"
    }))

    # Add model202_conv_20x1x2x14_same test case
    test_cases.append(Case("model202_conv_20x1x2x14_same", {
        "keras_model_name": "model202_conv_20x1x2x14_same"
    }))

    # Add model203_conv_19x5x2x24_same test case
    test_cases.append(Case("model203_conv_19x5x2x24_same", {
        "keras_model_name": "model203_conv_19x5x2x24_same"
    }))

    # Add model204_conv_32x5x4x5_valid_stride4 test case
    test_cases.append(Case("model204_conv_32x5x4x5_valid_stride4", {
        "keras_model_name": "model204_conv_32x5x4x5_valid_stride4"
    }))

    # Add model205_conv_9x2x5x21_same test case
    test_cases.append(Case("model205_conv_9x2x5x21_same", {
        "keras_model_name": "model205_conv_9x2x5x21_same"
    }))

    # Add model206_conv_17x1x1x19_valid test case
    test_cases.append(Case("model206_conv_17x1x1x19_valid", {
        "keras_model_name": "model206_conv_17x1x1x19_valid"
    }))

    # Add model207_conv_25x2x1x26_same test case
    test_cases.append(Case("model207_conv_25x2x1x26_same", {
        "keras_model_name": "model207_conv_25x2x1x26_same"
    }))

    # Add model208_conv_13x5x3x7_same_stride3 test case
    test_cases.append(Case("model208_conv_13x5x3x7_same_stride3", {
        "keras_model_name": "model208_conv_13x5x3x7_same_stride3"
    }))

    # Add model209_conv_25x1x5x9_same test case
    test_cases.append(Case("model209_conv_25x1x5x9_same", {
        "keras_model_name": "model209_conv_25x1x5x9_same"
    }))

    # Add model210_conv_12x3x3x24_same test case
    test_cases.append(Case("model210_conv_12x3x3x24_same", {
        "keras_model_name": "model210_conv_12x3x3x24_same"
    }))

    # Add model211_conv_26x5x4x2_valid_stride3 test case
    test_cases.append(Case("model211_conv_26x5x4x2_valid_stride3", {
        "keras_model_name": "model211_conv_26x5x4x2_valid_stride3"
    }))

    # Add model212_conv_15x4x2x29_valid test case
    test_cases.append(Case("model212_conv_15x4x2x29_valid", {
        "keras_model_name": "model212_conv_15x4x2x29_valid"
    }))

    # Add model213_conv_22x1x3x9_valid test case
    test_cases.append(Case("model213_conv_22x1x3x9_valid", {
        "keras_model_name": "model213_conv_22x1x3x9_valid"
    }))

    # Add model214_conv_8x4x4x27_same_stride3 test case
    test_cases.append(Case("model214_conv_8x4x4x27_same_stride3", {
        "keras_model_name": "model214_conv_8x4x4x27_same_stride3"
    }))

    # Add model215_conv_16x5x2x7_valid test case
    test_cases.append(Case("model215_conv_16x5x2x7_valid", {
        "keras_model_name": "model215_conv_16x5x2x7_valid"
    }))

    # Add model216_conv_9x4x3x11_same test case
    test_cases.append(Case("model216_conv_9x4x3x11_same", {
        "keras_model_name": "model216_conv_9x4x3x11_same"
    }))

    # Add model217_conv_17x3x5x10_valid_stride3 test case
    test_cases.append(Case("model217_conv_17x3x5x10_valid_stride3", {
        "keras_model_name": "model217_conv_17x3x5x10_valid_stride3"
    }))

    # Add model218_conv_23x5x4x10_valid_stride2 test case
    test_cases.append(Case("model218_conv_23x5x4x10_valid_stride2", {
        "keras_model_name": "model218_conv_23x5x4x10_valid_stride2"
    }))

    # Add model219_conv_26x3x3x20_same_stride3 test case
    test_cases.append(Case("model219_conv_26x3x3x20_same_stride3", {
        "keras_model_name": "model219_conv_26x3x3x20_same_stride3"
    }))

    # Add model220_conv_inp1x4x4x2_ker3x3_same_ker2x2_same test case
    test_cases.append(Case("model220_conv_inp1x4x4x2_ker3x3_same_ker2x2_same", {
        "keras_model_name": "model220_conv_inp1x4x4x2_ker3x3_same_ker2x2_same"
    }))

    # Add model221_conv_inp1x4x4x2_ker3x3_same_ker3x3_same test case
    test_cases.append(Case("model221_conv_inp1x4x4x2_ker3x3_same_ker3x3_same", {
        "keras_model_name": "model221_conv_inp1x4x4x2_ker3x3_same_ker3x3_same"
    }))

    # Add model222_conv_inp1x4x4x2_ker2x2_same_ker3x3_same test case
    test_cases.append(Case("model222_conv_inp1x4x4x2_ker2x2_same_ker3x3_same", {
        "keras_model_name": "model222_conv_inp1x4x4x2_ker2x2_same_ker3x3_same"
    }))

    # Add model237_conv_inp1x18x76x24_1x5x2_same_stride1x1 test case
    test_cases.append(Case("model237_conv_inp1x18x76x24_1x5x2_same_stride1x1", {
        "keras_model_name": "model237_conv_inp1x18x76x24_1x5x2_same_stride1x1"
    }))

    # Add model238_conv_inp1x58x26x24_1x5x2_same_stride1x1 test case
    test_cases.append(Case("model238_conv_inp1x58x26x24_1x5x2_same_stride1x1", {
        "keras_model_name": "model238_conv_inp1x58x26x24_1x5x2_same_stride1x1"
    }))

    return test_cases


def get_keras_depthwise_test_cases():
    test_cases = []

    # Add model010_depthwise_1x3x3x16_valid test case
    test_cases.append(Case("model010_depthwise_1x3x3x16_valid", {
        "keras_model_name": "model010_depthwise_1x3x3x16_valid"
    }))

    # Add model011_depthwise_1x3x3x16_valid test case
    test_cases.append(Case("model011_depthwise_1x3x3x16_valid", {
        "keras_model_name": "model011_depthwise_1x3x3x16_valid"
    }))

    # Add model012_depthwise_1x3x3x2_valid test case
    test_cases.append(Case("model012_depthwise_1x3x3x2_valid", {
        "keras_model_name": "model012_depthwise_1x3x3x2_valid"
    }))

    # Add model013_depthwise_1x3x3x2_same test case
    test_cases.append(Case("model013_depthwise_1x3x3x2_same", {
        "keras_model_name": "model013_depthwise_1x3x3x2_same"
    }))

    # Add model014_depthwise_1x3x3x8_valid test case
    test_cases.append(Case("model014_depthwise_1x3x3x8_valid", {
        "keras_model_name": "model014_depthwise_1x3x3x8_valid"
    }))

    return test_cases


def get_keras_fc_test_cases():
    test_cases = []

    # Add model003_hello_world test case
    test_cases.append(Case("model003_hello_world", {
        "keras_model_name": "model003_hello_world"
    }))

    # Add model020_fc_1x1 test case
    test_cases.append(Case("model020_fc_1x1", {
        "keras_model_name": "model020_fc_1x1"
    }))

    # Add model021_fc_1991x61 test case
    test_cases.append(Case("model021_fc_1991x61", {
        "keras_model_name": "model021_fc_1991x61"
    }))

    # Add model022_fc_1024x1024 test case
    test_cases.append(Case("model022_fc_1024x1024", {
        "keras_model_name": "model022_fc_1024x1024"
    }))

    # Add model023_fc_512x1000 test case
    test_cases.append(Case("model023_fc_512x1000", {
        "keras_model_name": "model023_fc_512x1000"
    }))

    # Add model024_fc_97x2000 test case
    test_cases.append(Case("model024_fc_97x2000", {
        "keras_model_name": "model024_fc_97x2000"
    }))

    # Add model026_fc_1991x61 test case
    test_cases.append(Case("model026_fc_1991x61", {
        "keras_model_name": "model026_fc_1991x61"
    }))

    # Add model027_fc_500x700 test case
    test_cases.append(Case("model027_fc_500x700", {
        "keras_model_name": "model027_fc_500x700"
    }))

    # Add model029_fc_1000x1000 test case
    test_cases.append(Case("model029_fc_1000x1000", {
        "keras_model_name": "model029_fc_1000x1000"
    }))

    # Add model080_fc_7x3 test case
    test_cases.append(Case("model080_fc_7x3", {
        "keras_model_name": "model080_fc_7x3"
    }))

    return test_cases


def get_keras_mean_test_cases():
    test_cases = []

    # Add model180_mean_12x8x256_to_1x256 test case
    test_cases.append(Case("model180_mean_12x8x256_to_1x256", {
        "keras_model_name": "model180_mean_12x8x256_to_1x256"
    }))

    # Add model181_mean_inp_4x8x6_zp_64 test case
    test_cases.append(Case("model181_mean_inp_4x8x6_zp_64", {
        "keras_model_name": "model181_mean_inp_4x8x6_zp_64"
    }))

    # Add model182_mean_inp_22x32x16_zp_1 test case
    test_cases.append(Case("model182_mean_inp_22x32x16_zp_1", {
        "keras_model_name": "model182_mean_inp_22x32x16_zp_1"
    }))

    # Add model183_mean_inp_37x17x11_zp_minus127 test case
    test_cases.append(Case("model183_mean_inp_37x17x11_zp_minus127", {
        "keras_model_name": "model183_mean_inp_37x17x11_zp_minus127"
    }))

    # Add model184_mean_inp_12x16_64_zp_minus76 test case
    test_cases.append(Case("model184_mean_inp_12x16_64_zp_minus76", {
        "keras_model_name": "model184_mean_inp_12x16_64_zp_minus76"
    }))

    # Add model185_mean_inp_32x1x512_zp_74_scale_3 test case
    test_cases.append(Case("model185_mean_inp_32x1x512_zp_74_scale_3", {
        "keras_model_name": "model185_mean_inp_32x1x512_zp_74_scale_3"
    }))

    # Add model186_mean_53x37x1_zp_minus25_scale_39 test case
    test_cases.append(Case("model186_mean_53x37x1_zp_minus25_scale_39", {
        "keras_model_name": "model186_mean_53x37x1_zp_minus25_scale_39"
    }))

    # Add model187_mean_1x32x512_zp_26 test case
    test_cases.append(Case("model187_mean_1x32x512_zp_26", {
        "keras_model_name": "model187_mean_1x32x512_zp_26"
    }))

    # Add model188_mean_11x1x1_zp_52_scale_em7 test case
    test_cases.append(Case("model188_mean_11x1x1_zp_52_scale_em7", {
        "keras_model_name": "model188_mean_11x1x1_zp_52_scale_em7"
    }))

    # Add model189_mean_1x173x1_zp_minus71_scale19 test case
    test_cases.append(Case("model189_mean_1x173x1_zp_minus71_scale19", {
        "keras_model_name": "model189_mean_1x173x1_zp_minus71_scale19"
    }))

    return test_cases


def get_keras_pointwise_test_cases():
    test_cases = []

    # Add model016_pointwise_8x8x8x16_stride2x2 test case
    test_cases.append(Case("model016_pointwise_8x8x8x16_stride2x2", {
        "keras_model_name": "model016_pointwise_8x8x8x16_stride2x2"
    }))

    # Add model017_pointwise_8x8x9x16_stride2x2 test case
    test_cases.append(Case("model017_pointwise_8x8x9x16_stride2x2", {
        "keras_model_name": "model017_pointwise_8x8x9x16_stride2x2"
    }))

    # Add model018_pointwise_8x9x8x16_stride2x2 test case
    test_cases.append(Case("model018_pointwise_8x9x8x16_stride2x2", {
        "keras_model_name": "model018_pointwise_8x9x8x16_stride2x2"
    }))

    # Add model019_pointwise_8x9x9x16_stride2x2 test case
    test_cases.append(Case("model019_pointwise_8x9x9x16_stride2x2", {
        "keras_model_name": "model019_pointwise_8x9x9x16_stride2x2"
    }))

    # Add model113_pointwise_1x1x1024 test case
    test_cases.append(Case("model113_pointwise_1x1x1024", {
        "keras_model_name": "model113_pointwise_1x1x1024"
    }))

    # Add model240_pointwise_inp1x4x4x16_8x1x1_valid_stride1x1 test case
    test_cases.append(Case("model240_pointwise_inp1x4x4x16_8x1x1_valid_stride1x1", {
        "keras_model_name": "model240_pointwise_inp1x4x4x16_8x1x1_valid_stride1x1"
    }))

    return test_cases


def get_keras_pooling_test_cases():
    test_cases = []

    # Add model030_avgpool_inp1x10x10x4_pool3x3_stride3x3_valid test case
    test_cases.append(Case("model030_avgpool_inp1x10x10x4_pool3x3_stride3x3_valid", {
        "keras_model_name": "model030_avgpool_inp1x10x10x4_pool3x3_stride3x3_valid"
    }))

    # Add model031_avgpool_inp1x10x10x4_pool4x4_stride4x4_valid test case
    test_cases.append(Case("model031_avgpool_inp1x10x10x4_pool4x4_stride4x4_valid", {
        "keras_model_name": "model031_avgpool_inp1x10x10x4_pool4x4_stride4x4_valid"
    }))

    # Add model032_avgpool_inp1x10x10x4_pool5x5_stride5x5_valid test case
    test_cases.append(Case("model032_avgpool_inp1x10x10x4_pool5x5_stride5x5_valid", {
        "keras_model_name": "model032_avgpool_inp1x10x10x4_pool5x5_stride5x5_valid"
    }))

    # Add model033_avgpool_inp1x10x10x4_pool5x5_stride1x1_valid test case
    test_cases.append(Case("model033_avgpool_inp1x10x10x4_pool5x5_stride1x1_valid", {
        "keras_model_name": "model033_avgpool_inp1x10x10x4_pool5x5_stride1x1_valid"
    }))

    # Add model034_avgpool_inp1x10x10x4_pool1x3_stride1x3_valid test case
    test_cases.append(Case("model034_avgpool_inp1x10x10x4_pool1x3_stride1x3_valid", {
        "keras_model_name": "model034_avgpool_inp1x10x10x4_pool1x3_stride1x3_valid"
    }))

    # Add model035_avgpool_inp1x10x10x4_pool3x3_stride3x3_same test case
    test_cases.append(Case("model035_avgpool_inp1x10x10x4_pool3x3_stride3x3_same", {
        "keras_model_name": "model035_avgpool_inp1x10x10x4_pool3x3_stride3x3_same"
    }))

    # Add model036_avgpool_inp1x10x10x4_pool4x4_stride4x4_same test case
    test_cases.append(Case("model036_avgpool_inp1x10x10x4_pool4x4_stride4x4_same", {
        "keras_model_name": "model036_avgpool_inp1x10x10x4_pool4x4_stride4x4_same"
    }))

    # Add model037_avgpool_inp1x10x10x4_pool5x5_stride5x5_same test case
    test_cases.append(Case("model037_avgpool_inp1x10x10x4_pool5x5_stride5x5_same", {
        "keras_model_name": "model037_avgpool_inp1x10x10x4_pool5x5_stride5x5_same"
    }))

    # Add model038_avgpool_inp1x10x10x4_pool5x5_stride1x1_same test case
    test_cases.append(Case("model038_avgpool_inp1x10x10x4_pool5x5_stride1x1_same", {
        "keras_model_name": "model038_avgpool_inp1x10x10x4_pool5x5_stride1x1_same"
    }))

    # Add model039_avgpool_inp1x10x10x4_pool1x3_stride1x1_same test case
    test_cases.append(Case("model039_avgpool_inp1x10x10x4_pool1x3_stride1x1_same", {
        "keras_model_name": "model039_avgpool_inp1x10x10x4_pool1x3_stride1x1_same"
    }))

    # Add model040_maxpool_inp1x10x10x4_pool3x3_stride3x3_valid test case
    test_cases.append(Case("model040_maxpool_inp1x10x10x4_pool3x3_stride3x3_valid", {
        "keras_model_name": "model040_maxpool_inp1x10x10x4_pool3x3_stride3x3_valid"
    }))

    # Add model041_maxpool_inp1x10x10x4_pool4x4_stride4x4_valid test case
    test_cases.append(Case("model041_maxpool_inp1x10x10x4_pool4x4_stride4x4_valid", {
        "keras_model_name": "model041_maxpool_inp1x10x10x4_pool4x4_stride4x4_valid"
    }))

    # Add model042_maxpool_inp1x10x10x4_pool5x5_stride5x5_valid test case
    test_cases.append(Case("model042_maxpool_inp1x10x10x4_pool5x5_stride5x5_valid", {
        "keras_model_name": "model042_maxpool_inp1x10x10x4_pool5x5_stride5x5_valid"
    }))

    # Add model043_maxpool_inp1x10x10x4_pool5x5_stride1x1_valid test case
    test_cases.append(Case("model043_maxpool_inp1x10x10x4_pool5x5_stride1x1_valid", {
        "keras_model_name": "model043_maxpool_inp1x10x10x4_pool5x5_stride1x1_valid"
    }))

    # Add model044_maxpool_inp1x10x10x4_pool1x3_stride1x3_valid test case
    test_cases.append(Case("model044_maxpool_inp1x10x10x4_pool1x3_stride1x3_valid", {
        "keras_model_name": "model044_maxpool_inp1x10x10x4_pool1x3_stride1x3_valid"
    }))

    # Add model045_maxpool_inp1x10x10x4_pool3x3_stride3x3_same test case
    test_cases.append(Case("model045_maxpool_inp1x10x10x4_pool3x3_stride3x3_same", {
        "keras_model_name": "model045_maxpool_inp1x10x10x4_pool3x3_stride3x3_same"
    }))

    # Add model046_maxpool_inp1x10x10x4_pool4x4_stride4x4_same test case
    test_cases.append(Case("model046_maxpool_inp1x10x10x4_pool4x4_stride4x4_same", {
        "keras_model_name": "model046_maxpool_inp1x10x10x4_pool4x4_stride4x4_same"
    }))

    # Add model047_maxpool_inp1x10x10x4_pool5x5_stride5x5_same test case
    test_cases.append(Case("model047_maxpool_inp1x10x10x4_pool5x5_stride5x5_same", {
        "keras_model_name": "model047_maxpool_inp1x10x10x4_pool5x5_stride5x5_same"
    }))

    # Add model048_maxpool_inp1x10x10x4_pool5x5_stride1x1_same test case
    test_cases.append(Case("model048_maxpool_inp1x10x10x4_pool5x5_stride1x1_same", {
        "keras_model_name": "model048_maxpool_inp1x10x10x4_pool5x5_stride1x1_same"
    }))

    # Add model049_maxpool_inp1x10x10x4_pool1x3_stride1x1_same test case
    test_cases.append(Case("model049_maxpool_inp1x10x10x4_pool1x3_stride1x1_same", {
        "keras_model_name": "model049_maxpool_inp1x10x10x4_pool1x3_stride1x1_same"
    }))

    # Add model055_avgpool_inp1x10x10x4_pool2x2_stride2x2_valid test case
    test_cases.append(Case("model055_avgpool_inp1x10x10x4_pool2x2_stride2x2_valid", {
        "keras_model_name": "model055_avgpool_inp1x10x10x4_pool2x2_stride2x2_valid"
    }))

    # Add model056_avgpool_inp1x100x100x1_pool2x2_stride2x2_valid test case
    test_cases.append(Case("model056_avgpool_inp1x100x100x1_pool2x2_stride2x2_valid", {
        "keras_model_name": "model056_avgpool_inp1x100x100x1_pool2x2_stride2x2_valid"
    }))

    # Add model057_avgpool_inp1x100x100x4_pool2x2_stride2x2_valid test case
    test_cases.append(Case("model057_avgpool_inp1x100x100x4_pool2x2_stride2x2_valid", {
        "keras_model_name": "model057_avgpool_inp1x100x100x4_pool2x2_stride2x2_valid"
    }))

    # Add model058_avgpool_inp1x134x158x5_pool2x2_stride2x2_valid test case
    test_cases.append(Case("model058_avgpool_inp1x134x158x5_pool2x2_stride2x2_valid", {
        "keras_model_name": "model058_avgpool_inp1x134x158x5_pool2x2_stride2x2_valid"
    }))

    # Add model065_SmartAve_1x25x5x4 test case
    test_cases.append(Case("model065_SmartAve_1x25x5x4", {
        "keras_model_name": "model065_SmartAve_1x25x5x4"
    }))

    # Add model066_SmartAve_1x5x25x4 test case
    test_cases.append(Case("model066_SmartAve_1x5x25x4", {
        "keras_model_name": "model066_SmartAve_1x5x25x4"
    }))

    # Add model117_avgpool_valid_model70_L1 test case
    test_cases.append(Case("model117_avgpool_valid_model70_L1", {
        "keras_model_name": "model117_avgpool_valid_model70_L1"
    }))

    # Add model118_avgpool_valid_model70_L3 test case
    test_cases.append(Case("model118_avgpool_valid_model70_L3", {
        "keras_model_name": "model118_avgpool_valid_model70_L3"
    }))

    # Add model119_avgpool_valid_model70_L6 test case
    test_cases.append(Case("model119_avgpool_valid_model70_L6", {
        "keras_model_name": "model119_avgpool_valid_model70_L6"
    }))

    return test_cases


def get_keras_sig_tanh_relu_test_cases():
    test_cases = []

    # Add model081_tanh_zpm26 test case
    test_cases.append(Case("model081_tanh_zpm26", {
        "keras_model_name": "model081_tanh_zpm26"
    }))

    # Add model082_tanh_zpm115 test case
    test_cases.append(Case("model082_tanh_zpm115", {
        "keras_model_name": "model082_tanh_zpm115"
    }))

    # Add model083_tanh_zpm128 test case
    test_cases.append(Case("model083_tanh_zpm128", {
        "keras_model_name": "model083_tanh_zpm128"
    }))

    # Add model084_tanh_zp76 test case
    test_cases.append(Case("model084_tanh_zp76", {
        "keras_model_name": "model084_tanh_zp76"
    }))

    # Add model085_tanh_zp127 test case
    test_cases.append(Case("model085_tanh_zp127", {
        "keras_model_name": "model085_tanh_zp127"
    }))

    # Add model086_tanh_1x4x6x8 test case
    test_cases.append(Case("model086_tanh_1x4x6x8", {
        "keras_model_name": "model086_tanh_1x4x6x8"
    }))

    # Add model087_sigmoid_1x4x6x8 test case
    test_cases.append(Case("model087_sigmoid_1x4x6x8", {
        "keras_model_name": "model087_sigmoid_1x4x6x8"
    }))

    # Add model088_sigmoid_1x1x16x128 test case
    test_cases.append(Case("model088_sigmoid_1x1x16x128", {
        "keras_model_name": "model088_sigmoid_1x1x16x128"
    }))

    # Add model090_sigmoid_zp26 test case
    test_cases.append(Case("model090_sigmoid_zp26", {
        "keras_model_name": "model090_sigmoid_zp26"
    }))

    # Add model091_sigmoid_zp128 test case
    test_cases.append(Case("model091_sigmoid_zp128", {
        "keras_model_name": "model091_sigmoid_zp128"
    }))

    # Add model092_sigmoid_zpm76 test case
    test_cases.append(Case("model092_sigmoid_zpm76", {
        "keras_model_name": "model092_sigmoid_zpm76"
    }))

    # Add model093_sigmoid_zp128 test case
    test_cases.append(Case("model093_sigmoid_zp128", {
        "keras_model_name": "model093_sigmoid_zp128"
    }))

    # Add model094_sigmoid_zp16 test case
    test_cases.append(Case("model094_sigmoid_zp16", {
        "keras_model_name": "model094_sigmoid_zp16"
    }))

    # Add model095_relu_8x8_inp_1x10x1x1 test case
    test_cases.append(Case("model095_relu_8x8_inp_1x10x1x1", {
        "keras_model_name": "model095_relu_8x8_inp_1x10x1x1"
    }))

    # Add model096_relu_zp26_8x8_inp_1x10x1x1 test case
    test_cases.append(Case("model096_relu_zp26_8x8_inp_1x10x1x1", {
        "keras_model_name": "model096_relu_zp26_8x8_inp_1x10x1x1"
    }))

    return test_cases


def get_keras_softmax_test_cases():
    test_cases = []

    # Add model089_softmax_inp1x1916x2 test case
    test_cases.append(Case("model089_softmax_inp1x1916x2", {
        "keras_model_name": "model089_softmax_inp1x1916x2"
    }))

    return test_cases


def get_keras_transpose_test_cases():
    test_cases = []

    # Add model110_transpose_last2first test case
    test_cases.append(Case("model110_transpose_last2first", {
        "keras_model_name": "model110_transpose_last2first"
    }))

    # Add model111_transpose_first2last test case
    test_cases.append(Case("model111_transpose_first2last", {
        "keras_model_name": "model111_transpose_first2last"
    }))

    # Add model112_transpose_2d test case
    test_cases.append(Case("model112_transpose_2d", {
        "keras_model_name": "model112_transpose_2d"
    }))

    # Add model160_transpose_ChLastToFirst_1x8x10x4 test case
    test_cases.append(Case("model160_transpose_ChLastToFirst_1x8x10x4", {
        "keras_model_name": "model160_transpose_ChLastToFirst_1x8x10x4"
    }))

    # Add model161_transpose_ChFirstToLast_1x8x10x4 test case
    test_cases.append(Case("model161_transpose_ChFirstToLast_1x8x10x4", {
        "keras_model_name": "model161_transpose_ChFirstToLast_1x8x10x4"
    }))

    # Add model162_transpose_2d_matrix_1x4x6x2 test case
    test_cases.append(Case("model162_transpose_2d_matrix_1x4x6x2", {
        "keras_model_name": "model162_transpose_2d_matrix_1x4x6x2"
    }))

    return test_cases


@pytest.fixture(params=get_test_cases())
def case_config(request, chip_config):
    # Extract the case from pytest.param if needed
    case = request.param

    # Next chip failures
    next_chip_failed_tc = [
        # FIXME: it should be model024_fc_97x2000_int8
        'model024_fc_97x2000' #'Failed to allocate LRAM addresses"
    ]
    tc = request.param.data['keras_model_name']
    if chip_config.data['target'] != "SL2610" and any(s in tc for s in next_chip_failed_tc):
        pytest.xfail("output mismatch or error on next chip")

    return {
        "keras_model": case.data['keras_model_name'],
        "keras_model_params": case.data.get('keras_model_params', {}),
        "mlir_model_file": "tflite_mlir_model_file",
        "tflite_model_file": "quantized_tflite_model_file",
        "input_data": "tweaked_random_input_data",
        "quantize_to_int16": case.data.get("quantize_to_int16", False)
    }


@pytest.mark.ci
@pytest.mark.fpga_ci
def test_keras_model(request, torq_results, tflite_reference_results, case_config):
    compare_test_results(request, torq_results, tflite_reference_results, case_config)
