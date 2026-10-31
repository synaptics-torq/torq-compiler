"""Keras test models: DepthwiseConv2D layers of the EMZA 70/75, MobileNetV2 and NNR 301 models."""

from torq.testing.keras_builder import depthwise

# These layers are only tested with int8 quantization.
QUANTIZE = ("int8",)

CASES = {
    "dw001_EMZA_70_dw_in38x32x10_k3x3_dm1_s1x1_valid": depthwise((38, 32, 10), (3, 3)),
    "dw002_EMZA_75_dw_in122x70x16_k3x3_dm1_s1x1_valid": depthwise((122, 70, 16), (3, 3)),
    "dw003_EMZA_75_dw_in122x70x32_k3x3_dm1_s2x2_valid": depthwise((122, 70, 32), (3, 3), (2, 2)),
    "dw004_EMZA_75_dw_in62x36x32_k3x3_dm1_s1x1_valid": depthwise((62, 36, 32), (3, 3)),
    "dw005_EMZA_75_dw_in62x36x32_k3x3_dm1_s2x2_valid": depthwise((62, 36, 32), (3, 3), (2, 2)),
    "dw006_EMZA_75_dw_in32x19x64_k3x3_dm1_s1x1_valid": depthwise((32, 19, 64), (3, 3)),
    "dw007_EMZA_75_dw_in32x19x64_k3x3_dm1_s2x2_valid": depthwise((32, 19, 64), (3, 3), (2, 2)),
    "dw008_EMZA_75_dw_in17x11x128_k3x3_dm1_s1x1_valid": depthwise((17, 11, 128), (3, 3)),
    "dw009_EMZA_75_dw_in17x11x128_k3x3_dm1_s2x2_valid": depthwise((17, 11, 128), (3, 3), (2, 2)),
    "dw010_EMZA_75_dw_in10x7x256_k3x3_dm1_s1x1_valid": depthwise((10, 7, 256), (3, 3)),
    "dw011_EMZA_75_dw_in10x7x64_k3x3_dm1_s2x2_valid": depthwise((10, 7, 64), (3, 3), (2, 2)),
    "dw012_MobileNetV2_dw_in112x112x32_k3x3_dm1_s1x1_same": depthwise(
        (112, 112, 32), (3, 3), "same",
    ),
    "dw013_MobileNetV2_dw_in112x112x96_k3x3_dm1_s2x2_same": depthwise(
        (112, 112, 96), (3, 3), (2, 2), "same",
    ),
    "dw014_MobileNetV2_dw_in56x56x144_k3x3_dm1_s1x1_same": depthwise((56, 56, 144), (3, 3), "same"),
    "dw015_MobileNetV2_dw_in56x56x144_k3x3_dm1_s2x2_same": depthwise(
        (56, 56, 144), (3, 3), (2, 2), "same",
    ),
    "dw016_MobileNetV2_dw_in28x28x192_k3x3_dm1_s1x1_same": depthwise((28, 28, 192), (3, 3), "same"),
    "dw017_MobileNetV2_dw_in28x28x192_k3x3_dm1_s2x2_same": depthwise(
        (28, 28, 192), (3, 3), (2, 2), "same",
    ),
    "dw018_MobileNetV2_dw_in14x14x384_k3x3_dm1_s1x1_same": depthwise((14, 14, 384), (3, 3), "same"),
    "dw019_MobileNetV2_dw_in14x14x576_k3x3_dm1_s1x1_same": depthwise((14, 14, 576), (3, 3), "same"),
    "dw020_MobileNetV2_dw_in14x14x576_k3x3_dm1_s2x2_same": depthwise(
        (14, 14, 576), (3, 3), (2, 2), "same",
    ),
    "dw021_MobileNetV2_dw_in7x7x960_k3x3_dm1_s1x1_same": depthwise((7, 7, 960), (3, 3), "same"),
    "dw022_NNR_301_dw_in128x1x32_k1x3_dm1_s1x1_same": depthwise((128, 1, 32), (1, 3), "same"),
    "dw023_NNR_301_dw_in128x1x32_k1x3_dm1_s2x2_same": depthwise(
        (128, 1, 32), (1, 3), (1, 2), "same",
    ),
    "dw024_NNR_301_dw_in64x1x64_k1x3_dm1_s1x1_same": depthwise((64, 1, 64), (1, 3), "same"),
    "dw025_NNR_301_dw_in64x1x64_k1x3_dm1_s2x2_same": depthwise((64, 1, 64), (1, 3), (1, 2), "same"),
    "dw026_NNR_301_dw_in32x1x64_k1x3_dm1_s2x2_same": depthwise((32, 1, 64), (1, 3), (1, 2), "same"),
    "dw027_mnist_dw_in26x26x8_k3x3_dm1_s2x2_same": depthwise((26, 26, 8), (3, 3), (2, 2), "same"),
}
