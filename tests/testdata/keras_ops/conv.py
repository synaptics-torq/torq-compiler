"""Keras test models: Conv2D, alone or followed by ReLU or another conv."""

from functools import partial

from tests.models.keras_models import build_conv_model, conv_model_params
from torq.testing.keras_builder import layers, sequential, conv2d

CASES = {
    "model001_conv_3X3_small": sequential(
        (24, 32, 1),
        layers.Conv2D(filters=4, kernel_size=(3, 3), strides=(2, 2), padding="same"),
        layers.ReLU(6.0),
    ),
    "model002_conv_2x3_small_valid": conv2d((3, 4, 2), 3, (2, 3)),
    "model004_conv_3x3_valid_bias": sequential(
        (3, 4, 2),
        layers.Conv2D(filters=3, kernel_size=(3, 3)),
        layers.ReLU(),
    ),
    "model005_conv2d_4x3x3x3_padding": sequential(
        (3, 4, 3),
        layers.Conv2D(filters=4, kernel_size=(3, 3), padding="same"),
        layers.ReLU(),
    ),
    "model006_conv2d_4x6x6x1_pad_stride2": sequential(
        (13, 35, 1),
        layers.Conv2D(filters=4, kernel_size=(6, 6), strides=(2, 2), padding="same"),
        layers.ReLU(),
    ),
    "model007_conv2d_8x3x3x4": sequential(
        (12, 14, 4),
        layers.Conv2D(filters=8, kernel_size=(3, 3), padding="same"),
        layers.ReLU(),
    ),
    "model008_conv2d_4x4x4x4": sequential(
        (12, 14, 4),
        layers.Conv2D(filters=4, kernel_size=(4, 4), padding="same"),
        layers.ReLU(),
    ),
    "model009_conv_3x3_same_bias": sequential(
        (4, 4, 1),
        layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same"),
        layers.ReLU(),
    ),
    "model200_conv_4_4_5_19_valid": conv2d((7, 30, 19), 4, (4, 5)),
    "model201_conv_20x4x1x12_valid": conv2d((38, 50, 12), 20, (4, 1)),
    "model202_conv_20x1x2x14_same": conv2d((39, 26, 14), 20, (1, 2), (1, 1), "same"),
    "model203_conv_19x5x2x24_same": conv2d((58, 9, 24), 19, (5, 2), (1, 1), "same"),
    "model204_conv_32x5x4x5_valid_stride4": conv2d((41, 19, 5), 32, (5, 4), (4, 4)),
    "model205_conv_9x2x5x21_same": conv2d((22, 55, 21), 9, (2, 5), (1, 1), "same"),
    "model206_conv_17x1x1x19_valid": conv2d((18, 26, 19), 17, (1, 1)),
    "model207_conv_25x2x1x26_same": conv2d((4, 3, 26), 25, (2, 1), (1, 1), "same"),
    "model208_conv_13x5x3x7_same_stride3": conv2d((16, 55, 7), 13, (5, 3), (3, 3), "same"),
    "model209_conv_25x1x5x9_same": conv2d((20, 42, 9), 25, (1, 5), (1, 1), "same"),
    "model210_conv_12x3x3x24_same": conv2d((5, 40, 24), 12, (4, 3), (1, 1), "same"),
    "model211_conv_26x5x4x2_valid_stride3": conv2d((34, 6, 2), 26, (5, 4), (3, 3)),
    "model212_conv_15x4x2x29_valid": conv2d((13, 16, 29), 15, (4, 2)),
    "model213_conv_22x1x3x9_valid": conv2d((60, 12, 9), 22, (1, 3)),
    "model214_conv_8x4x4x27_same_stride3": conv2d((45, 22, 27), 8, (4, 4), (3, 3), "same"),
    "model215_conv_16x5x2x7_valid": conv2d((54, 63, 7), 16, (5, 2)),
    "model216_conv_9x4x3x11_same": conv2d((6, 46, 11), 9, (4, 3), (1, 1), "same"),
    "model217_conv_17x3x5x10_valid_stride3": conv2d((15, 13, 10), 17, (3, 5), (3, 3)),
    "model218_conv_23x5x4x10_valid_stride2": conv2d((12, 52, 10), 23, (5, 4), (2, 2)),
    "model219_conv_26x3x3x20_same_stride3": conv2d((43, 61, 20), 26, (3, 3), (3, 3), "same"),
    "model220_conv_inp1x4x4x2_ker3x3_same_ker2x2_same": sequential(
        (4, 4, 2),
        layers.Conv2D(filters=2, kernel_size=(3, 3), padding="same"),
        layers.ReLU(),
        layers.Conv2D(filters=2, kernel_size=(2, 2), padding="same"),
        layers.ReLU(),
    ),
    "model221_conv_inp1x4x4x2_ker3x3_same_ker3x3_same": sequential(
        (4, 4, 2),
        layers.Conv2D(filters=2, kernel_size=(3, 3), padding="same"),
        layers.ReLU(),
        layers.Conv2D(filters=2, kernel_size=(3, 3), padding="same"),
        layers.ReLU(),
    ),
    "model222_conv_inp1x4x4x2_ker2x2_same_ker3x3_same": sequential(
        (4, 4, 2),
        layers.Conv2D(filters=2, kernel_size=(2, 2), padding="same"),
        layers.ReLU(),
        layers.Conv2D(filters=2, kernel_size=(3, 3), padding="same"),
        layers.ReLU(),
    ),
    "model237_conv_inp1x18x76x24_1x5x2_same_stride1x1": conv2d(
        (18, 76, 24), 1, (5, 2), (1, 1), "same",
    ),
    "model238_conv_inp1x58x26x24_1x5x2_same_stride1x1": conv2d(
        (58, 26, 24), 1, (5, 2), (1, 1), "same",
    ),
    # The conv_model of test_conv_sweep, with the two default parameter sets.
    **{f"conv_model_{p.idfn()}": partial(build_conv_model, p) for p in conv_model_params},
}

# A string quantize gives a single case with no _int8/_int16 suffix.
CONFIG = {f"conv_model_{p.idfn()}": {"quantize": "int8"} for p in conv_model_params}
