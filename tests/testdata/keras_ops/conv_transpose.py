"""Keras test models: Conv2DTranspose and Conv1DTranspose."""

from torq.testing.keras_builder import layers, sequential, conv2d_transpose

CASES = {
    "model100_conv_transpose_1x3_1x2": sequential(
        (1, 32, 48),
        layers.Conv2DTranspose(
            filters=48, kernel_size=(1, 3), strides=(1, 2), padding="same", use_bias=False,
        ),
    ),
    "model101_conv_transpose_1x3_1x1": sequential(
        (1, 64, 32),
        layers.Conv2DTranspose(filters=32, kernel_size=(1, 3), padding="same", use_bias=False),
    ),
    "model102_conv_transpose_stride1_ker_1x3_input_1x1x4x1": sequential(
        (1, 4, 1),
        layers.Conv2DTranspose(filters=1, kernel_size=(1, 3), padding="same", use_bias=False),
    ),
    "model103_conv_transpose_stride2_ker_1x3_1x1x4x1": sequential(
        (1, 4, 1),
        layers.Conv2DTranspose(
            filters=1, kernel_size=(1, 3), strides=(1, 2), padding="same", use_bias=False,
        ),
    ),
    "model104_conv_transpose_1x3_stride2_input_1x1x12x4": sequential(
        (12, 4),
        layers.Conv1DTranspose(filters=4, kernel_size=3, strides=2, padding="same", use_bias=False),
    ),
    "model105_conv_transpose_stride1_ker_1x3_input1x1x4x2": sequential(
        (1, 4, 2),
        layers.Conv2DTranspose(filters=2, kernel_size=(1, 3), padding="same", use_bias=False),
    ),
    "model500_conv_transpose_stride1_ker_1x1x3x1_padSame": conv2d_transpose(
        (1, 4, 1), 1, (1, 3), (1, 1), "same",
    ),
    "model501_conv_transpose_stride1x2_ker1x1x3x1_padSame": conv2d_transpose(
        (1, 4, 1), 1, (1, 3), (1, 2), "same",
    ),
    "model502_conv_transpose_stride1_ker2x1x3x1_padSame": conv2d_transpose(
        (1, 4, 1), 2, (1, 3), (1, 1), "same",
    ),
    "model503_conv_transpose_stride1_ker_1x1x3x1_padValid": conv2d_transpose((1, 4, 1), 1, (1, 3)),
    "model504_conv_transpose_stride1x2_ker1x1x3x1_padValid": conv2d_transpose(
        (1, 4, 1), 1, (1, 3), (1, 2),
    ),
    "model505_conv_transpose_stride1_ker2x1x3x1_padValid": conv2d_transpose((1, 4, 1), 2, (1, 3)),
    "model506_conv_transpose_stride1_1x3x3x1_same": conv2d_transpose(
        (3, 3, 1), 1, (3, 3), (1, 1), "same",
    ),
    "model507_conv_transpose_stride2_1x3x3x1_same": conv2d_transpose(
        (3, 3, 1), 1, (3, 3), (2, 2), "same",
    ),
    "model508_conv_transpose_stride1_3x2x2x2_same": conv2d_transpose(
        (4, 4, 2), 3, (2, 2), (1, 1), "same",
    ),
    "model509_conv_transpose_stride1_2x4x4x3_valid": conv2d_transpose((5, 5, 3), 2, (4, 4)),
    "model510_conv_transpose_stride2_3x4x3x2_same": conv2d_transpose(
        (7, 7, 2), 3, (4, 3), (2, 2), "same",
    ),
    "model511_conv_transpose_stride1_16x1x3x16_padsame": conv2d_transpose(
        (1, 4, 16), 16, (1, 3), (1, 1), "same",
    ),
    "model512_conv_transpose_stride1_16x1x3x16_padvalid": conv2d_transpose((1, 4, 16), 16, (1, 3)),
    "model513_conv_transpose_stride2_8x1x3x8_padSame": conv2d_transpose(
        (1, 4, 8), 8, (1, 3), (1, 2), "same",
    ),
    "model514_conv_transpose_stride2_8x1x3x8_padValid": conv2d_transpose(
        (1, 4, 8), 8, (1, 3), (1, 2),
    ),
    "model515_conv_transpose_stride1_ker1x1x3x2_padSame": conv2d_transpose(
        (1, 4, 2), 1, (1, 3), (1, 1), "same",
    ),
    "model516_conv_transpose_stride1x2_ker1x1x3x2_padSame": conv2d_transpose(
        (1, 4, 2), 1, (1, 3), (1, 2), "same",
    ),
    "model517_conv_transpose_stride1_ker1x1x3x2_padValid": conv2d_transpose((1, 4, 2), 1, (1, 3)),
    "model518_conv_transpose_stride1x2_ker1x1x3x2_padValid": conv2d_transpose(
        (1, 4, 2), 1, (1, 3), (1, 2),
    ),
    "model519_conv_transpose_stride1_1x3x3x1_valid": conv2d_transpose((3, 3, 1), 1, (3, 3)),
    "model520_conv_transpose_stride2_1x3x3x1_valid": conv2d_transpose((3, 3, 1), 1, (3, 3), (2, 2)),
    "model521_conv_transpose_stride2_3x3x3x2_valid": conv2d_transpose((4, 4, 2), 3, (3, 3), (2, 2)),
    "model522_conv_transpose_stride3_1x3x3x1_same": conv2d_transpose(
        (3, 3, 1), 1, (3, 3), (3, 3), "same",
    ),
    "model523_conv_transpose_stride3_1x3x3x1_valid": conv2d_transpose((3, 3, 1), 1, (3, 3), (3, 3)),
    "model531_convTrans_16x8_inp_1x64x32_ker1x3_stride1_padsame": conv2d_transpose(
        (1, 64, 32), 32, (1, 3), (1, 1), "same",
    ),
    "model532_convTrans_16x8_inp_1x4x1_ker1x3_stride1_padvalid": conv2d_transpose(
        (1, 4, 1), 1, (1, 3),
    ),
    "model533_convTrans_16x8_inp_1x4x1_ker1x3_stride2_padvalid": conv2d_transpose(
        (1, 4, 1), 1, (1, 3), (1, 2),
    ),
    "model534_convTrans_16x8_inp_1x3x3x1_ker3x3_stride1_padsame": conv2d_transpose(
        (3, 3, 1), 1, (3, 3), (1, 1), "same",
    ),
    "model535_convTrans_16x8_inp_1x3x3x1_ker3x3_stride2_padsame": conv2d_transpose(
        (3, 3, 1), 1, (3, 3), (2, 2), "same",
    ),
    "model536_convTrans_16x8_inp_1x4x4x2_ker2x2_stride1_padsame": conv2d_transpose(
        (4, 4, 2), 3, (2, 2), (1, 1), "same",
    ),
    "model537_convTrans_16x8_inp_1x5x5x3_ker4x4_stride1_padvalid": conv2d_transpose(
        (5, 5, 3), 23, (4, 4),
    ),
    "model538_convTrans_16x8_inp_1x7x7x2_ker4x3_stride2_padsame": conv2d_transpose(
        (7, 7, 2), 3, (4, 3), (2, 2), "same",
    ),
    "model539_convTrans_16x8_inp_1x3x3x1_ker3x3_stride3_padsame": conv2d_transpose(
        (3, 3, 1), 1, (3, 3), (3, 3), "same",
    ),
    "model540_convTrans_16x8_inp_1x3x3x1_ker3x3_stride3_padvalid": conv2d_transpose(
        (3, 3, 1), 1, (3, 3), (3, 3),
    ),
    "model541_convTrans_16x8_inp_1x11x10x7_ker2x4_stride2x4_valid": conv2d_transpose(
        (11, 10, 7), 11, (2, 4), (2, 4),
    ),
    "transpose_conv_model_quantized_i8": sequential(
        (8, 8, 4),
        layers.Conv2DTranspose(filters=4, kernel_size=4, strides=2, padding="same"),
        seed=21321,
    ),
}
CASES["transpose_conv_model_quantized_i16"] = CASES["transpose_conv_model_quantized_i8"]

# A string quantize gives a single case with no _int8/_int16 suffix.
CONFIG = {
    "transpose_conv_model_quantized_i8": {"quantize": "int8"},
    "transpose_conv_model_quantized_i16": {"quantize": "int16"},
}
