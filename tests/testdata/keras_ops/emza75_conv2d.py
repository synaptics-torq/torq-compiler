"""Keras test models: Conv2D layers of the EMZA 75 model."""

from torq.testing.keras_builder import conv2d

CASES = {
    "conv2d_test1_int8_inp_242x137x3_k3x3_oc16_s2x2_valid": conv2d(
        (242, 137, 3), 16, (3, 3), (2, 2),
    ),
    "conv2d_test2_int8_inp_120x68x16_k1x1_oc32_s1x1_valid": conv2d((120, 68, 16), 32, (1, 1)),
    "conv2d_test3_int8_inp_60x34x32_k1x1_oc32_s1x1_valid": conv2d((60, 34, 32), 32, (1, 1)),
    "conv2d_test4_int8_inp_60x34x32_k1x1_oc32_s1x1_valid": conv2d((60, 34, 32), 32, (1, 1)),
    "conv2d_test5_int8_inp_30x17x32_k1x1_oc64_s1x1_valid": conv2d((30, 17, 32), 64, (1, 1)),
    "conv2d_test6_int8_inp_30x17x64_k1x1_oc64_s1x1_valid": conv2d((30, 17, 64), 64, (1, 1)),
    "conv2d_test7_int8_inp_30x17x64_k1x1_oc64_s1x1_valid": conv2d((30, 17, 64), 64, (1, 1)),
    "conv2d_test8_int8_inp_30x17x64_k1x1_oc64_s1x1_valid": conv2d((30, 17, 64), 64, (1, 1)),
    "conv2d_test9_int8_inp_30x17x64_k1x1_oc64_s1x1_valid": conv2d((30, 17, 64), 64, (1, 1)),
    "conv2d_test10_int8_inp_15x9x64_k1x1_oc128_s1x1_valid": conv2d((15, 9, 64), 128, (1, 1)),
    "conv2d_test11_int8_inp_15x9x128_k1x1_oc128_s1x1_valid": conv2d((15, 9, 128), 128, (1, 1)),
    "conv2d_test12_int8_inp_15x9x128_k1x1_oc128_s1x1_valid": conv2d((15, 9, 128), 128, (1, 1)),
    "conv2d_test13_int8_inp_8x5x128_k1x1_oc256_s1x1_valid": conv2d((8, 5, 128), 256, (1, 1)),
    "conv2d_test14_int8_inp_8x5x256_k1x1_oc256_s1x1_valid": conv2d((8, 5, 256), 256, (1, 1)),
    "conv2d_test15_int8_inp_8x5x256_k1x1_oc4_s1x1_valid": conv2d((8, 5, 256), 4, (1, 1)),
    "conv2d_test16_int8_inp_8x5x256_k1x1_oc64_s1x1_same": conv2d((8, 5, 256), 64, (1, 1), "same"),
    "conv2d_test17_int8_inp_4x3x64_k1x1_oc256_s1x1_valid": conv2d((4, 3, 64), 256, (1, 1)),
    "conv2d_test18_int8_inp_4x3x256_k3x3_oc6_s1x1_same": conv2d((4, 3, 256), 6, (3, 3), "same"),
    "conv2d_test19_int8_inp_4x3x256_k3x3_oc12_s1x1_same": conv2d((4, 3, 256), 12, (3, 3), "same"),
    "conv2d_test20_int8_inp_4x3x256_k3x3_oc9_s1x1_same": conv2d((4, 3, 256), 9, (3, 3), "same"),
    "conv2d_test21_int8_inp_8x5x256_k1x1_oc8_s1x1_valid": conv2d((8, 5, 256), 8, (1, 1)),
    "conv2d_test22_int8_inp_8x5x256_k1x1_oc6_s1x1_valid": conv2d((8, 5, 256), 6, (1, 1)),
    "conv2d_test23_int8_inp_15x9x128_k1x1_oc4_s1x1_valid": conv2d((15, 9, 128), 4, (1, 1)),
    "conv2d_test24_int8_inp_15x9x128_k1x1_oc8_s1x1_valid": conv2d((15, 9, 128), 8, (1, 1)),
    "conv2d_test25_int8_inp_15x9x128_k1x1_oc6_s1x1_valid": conv2d((15, 9, 128), 6, (1, 1)),
    "conv2d_test26_int8_inp_30x17x64_k1x1_oc6_s1x1_valid": conv2d((30, 17, 64), 6, (1, 1)),
    "conv2d_test27_int8_inp_30x17x64_k1x1_oc12_s1x1_valid": conv2d((30, 17, 64), 12, (1, 1)),
    "conv2d_test28_int8_inp_30x17x64_k1x1_oc9_s1x1_valid": conv2d((30, 17, 64), 9, (1, 1)),
}

CONFIG = {
    # FIXME: fails with slicing.
    "conv2d_test1_int8_inp_242x137x3_k3x3_oc16_s2x2_valid": {
        "compiler_options": ["--torq-disable-slicing"],
    },
}
