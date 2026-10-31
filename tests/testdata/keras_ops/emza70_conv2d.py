"""Keras test models: Conv2D layers of the EMZA 70 model."""

from torq.testing.keras_builder import conv2d

CASES = {
    "conv2d_test1_int8_inp_136x160x1_k3x3_oc5_s1x1_valid": conv2d((136, 160, 1), 5, (3, 3)),
    "conv2d_test2_int8_inp_67x79x5_k3x3_oc10_s1x1_valid": conv2d((67, 79, 5), 10, (3, 3)),
    "conv2d_test3_int8_inp_30x36x10_k1x1_oc15_s1x1_valid": conv2d((30, 36, 10), 15, (1, 1)),
    "conv2d_test4_int8_inp_15x18x15_k4x3_oc64_s1x1_valid": conv2d((15, 18, 15), 64, (4, 3)),
    "conv2d_test5_int8_inp_12x16x64_k1x1_oc32_s1x1_valid": conv2d((12, 16, 64), 32, (1, 1)),
    "conv2d_test6_int8_inp_12x16x32_k1x1_oc4_s1x1_valid": conv2d((12, 16, 32), 4, (1, 1)),
    "conv2d_test7_int8_inp_12x16x32_k1x1_oc1_s1x1_valid": conv2d((12, 16, 32), 1, (1, 1)),
}
