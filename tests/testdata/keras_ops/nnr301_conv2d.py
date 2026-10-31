"""Keras test models: Conv2D layers of the NNR 301 model."""

from torq.testing.keras_builder import conv2d

CASES = {
    "conv2d_test1_int8_inp_256x1x1_k3x1_oc32_s2x1_same": conv2d(
        (256, 1, 1), 32, (3, 1), (2, 1), "same",
    ),
    "conv2d_test2_int8_inp_128x1x32_k1x1_oc32_s1x1_valid": conv2d((128, 1, 32), 32, (1, 1)),
    "conv2d_test3_int8_inp_64x1x32_k1x1_oc64_s1x1_valid": conv2d((64, 1, 32), 64, (1, 1)),
    "conv2d_test4_int8_inp_64x1x64_k1x1_oc64_s1x1_valid": conv2d((64, 1, 64), 64, (1, 1)),
    "conv2d_test5_int8_inp_32x1x64_k1x1_oc64_s1x1_valid": conv2d((32, 1, 64), 64, (1, 1)),
    "conv2d_test6_int8_inp_16x1x64_k1x1_oc64_s1x1_valid": conv2d((16, 1, 64), 64, (1, 1)),
    "conv2d_test7_int8_inp_16x1x64_k1x1_oc32_s1x1_same": conv2d((16, 1, 64), 32, (1, 1), "same"),
    "conv2d_test8_int8_inp_16x1x32_k1x1_oc48_s1x1_valid": conv2d((16, 1, 32), 48, (1, 1)),
    "conv2d_test9_int8_inp_16x1x48_k2x1_oc48_s1x1_same": conv2d((16, 1, 48), 48, (2, 1), "same"),
    "conv2d_test10_int8_inp_16x1x48_k1x1_oc48_s1x1_valid": conv2d((16, 1, 48), 48, (1, 1)),
    "conv2d_test11_int8_inp_32x1x48_k1x1_oc48_s1x1_valid": conv2d((32, 1, 48), 48, (1, 1)),
    "conv2d_test12_int8_inp_32x1x48_k2x1_oc48_s1x1_same": conv2d((32, 1, 48), 48, (2, 1), "same"),
    "conv2d_test13_int8_inp_64x1x48_k1x1_oc32_s1x1_valid": conv2d((64, 1, 48), 32, (1, 1)),
    "conv2d_test14_int8_inp_64x1x32_k3x1_oc32_s1x1_same": conv2d((64, 1, 32), 32, (3, 1), "same"),
    "conv2d_test15_int8_inp_64x1x32_k1x1_oc32_s1x1_valid": conv2d((64, 1, 32), 32, (1, 1)),
    "conv2d_test16_int8_inp_64x1x32_k2x1_oc32_s1x1_same": conv2d((64, 1, 32), 32, (2, 1), "same"),
    "conv2d_test17_int8_inp_128x1x32_k3x1_oc32_s1x1_same": conv2d((128, 1, 32), 32, (3, 1), "same"),
    "conv2d_test18_int8_inp_128x1x32_k1x1_oc1_s1x1_valid": conv2d((128, 1, 32), 1, (1, 1)),
    "conv2d_test19_int8_inp_128x1x1_k2x1_oc1_s1x1_same": conv2d((128, 1, 1), 1, (2, 1), "same"),
    "conv2d_test20_int8_inp_128x1x1_k1x1_oc1_s1x1_valid": conv2d((128, 1, 1), 1, (1, 1)),
}
