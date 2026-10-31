"""Keras test models: Conv2DTranspose layers of the NNR 301 model."""

from torq.testing.keras_builder import conv2d_transpose

CASES = {
    "convtranspose_test1_int8_nnr301_inp_16x1x48_k3x1_oc48_s2x1_same": conv2d_transpose(
        (16, 1, 48), 48, (3, 1), (2, 1), "same",
    ),
    "convtranspose_test2_int8_nnr301_inp_32x1x48_k3x1_oc48_s2x1_same": conv2d_transpose(
        (32, 1, 48), 48, (3, 1), (2, 1), "same",
    ),
    "convtranspose_test3_int8_nnr301_inp_64x1x32_k3x1_oc32_s1x1_same": conv2d_transpose(
        (64, 1, 32), 32, (3, 1), "same",
    ),
    "convtranspose_test4_int8_nnr301_inp_64x1x32_k3x1_oc32_s2x1_same": conv2d_transpose(
        (64, 1, 32), 32, (3, 1), (2, 1), "same",
    ),
    "convtranspose_test5_int8_nnr301_inp_128x1x32_k3x1_oc32_s1x1_same": conv2d_transpose(
        (128, 1, 32), 32, (3, 1), "same",
    ),
    "convtranspose_test6_int8_nnr301_inp_128x1x1_k3x1_oc1_s2x1_same": conv2d_transpose(
        (128, 1, 1), 1, (3, 1), (2, 1), "same",
    ),
}
