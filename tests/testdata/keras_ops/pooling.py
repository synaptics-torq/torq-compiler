"""Keras test models: AveragePooling2D and MaxPooling2D."""

from torq.testing.keras_builder import layers, sequential

CASES = {
    "model030_avgpool_inp1x10x10x4_pool3x3_stride3x3_valid": sequential(
        (10, 10, 4),
        layers.AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding="valid"),
    ),
    "model031_avgpool_inp1x10x10x4_pool4x4_stride4x4_valid": sequential(
        (10, 10, 4),
        layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding="valid"),
    ),
    "model032_avgpool_inp1x10x10x4_pool5x5_stride5x5_valid": sequential(
        (10, 10, 4),
        layers.AveragePooling2D(pool_size=(5, 5), strides=(5, 5), padding="valid"),
    ),
    "model033_avgpool_inp1x10x10x4_pool5x5_stride1x1_valid": sequential(
        (10, 10, 4),
        layers.AveragePooling2D(pool_size=(5, 5), strides=(5, 5), padding="valid"),
    ),
    "model034_avgpool_inp1x10x10x4_pool1x3_stride1x3_valid": sequential(
        (10, 10, 4),
        layers.AveragePooling2D(pool_size=(1, 3), strides=(1, 3), padding="valid"),
    ),
    "model035_avgpool_inp1x10x10x4_pool3x3_stride3x3_same": sequential(
        (10, 10, 4),
        layers.AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding="same"),
    ),
    "model036_avgpool_inp1x10x10x4_pool4x4_stride4x4_same": sequential(
        (10, 10, 4),
        layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding="same"),
    ),
    "model037_avgpool_inp1x10x10x4_pool5x5_stride5x5_same": sequential(
        (10, 10, 4),
        layers.AveragePooling2D(pool_size=(5, 5), strides=(5, 5), padding="same"),
    ),
    "model038_avgpool_inp1x10x10x4_pool5x5_stride1x1_same": sequential(
        (10, 10, 4),
        layers.AveragePooling2D(pool_size=(5, 5), strides=(5, 5), padding="same"),
    ),
    "model039_avgpool_inp1x10x10x4_pool1x3_stride1x1_same": sequential(
        (10, 10, 4),
        layers.AveragePooling2D(pool_size=(1, 3), strides=(1, 3), padding="same"),
    ),
    "model040_maxpool_inp1x10x10x4_pool3x3_stride3x3_valid": sequential(
        (10, 10, 4),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding="valid"),
    ),
    "model041_maxpool_inp1x10x10x4_pool4x4_stride4x4_valid": sequential(
        (10, 10, 4),
        layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding="valid"),
    ),
    "model042_maxpool_inp1x10x10x4_pool5x5_stride5x5_valid": sequential(
        (10, 10, 4),
        layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding="valid"),
    ),
    "model043_maxpool_inp1x10x10x4_pool5x5_stride1x1_valid": sequential(
        (10, 10, 4),
        layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding="valid"),
    ),
    "model044_maxpool_inp1x10x10x4_pool1x3_stride1x3_valid": sequential(
        (10, 10, 4),
        layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding="valid"),
    ),
    "model045_maxpool_inp1x10x10x4_pool3x3_stride3x3_same": sequential(
        (10, 10, 4),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding="same"),
    ),
    "model046_maxpool_inp1x10x10x4_pool4x4_stride4x4_same": sequential(
        (10, 10, 4),
        layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding="same"),
    ),
    "model047_maxpool_inp1x10x10x4_pool5x5_stride5x5_same": sequential(
        (10, 10, 4),
        layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding="same"),
    ),
    "model048_maxpool_inp1x10x10x4_pool5x5_stride1x1_same": sequential(
        (10, 10, 4),
        layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding="same"),
    ),
    "model049_maxpool_inp1x10x10x4_pool1x3_stride1x1_same": sequential(
        (10, 10, 4),
        layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding="same"),
    ),
    "model055_avgpool_inp1x10x10x4_pool2x2_stride2x2_valid": sequential(
        (10, 10, 4),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
    ),
    "model056_avgpool_inp1x100x100x1_pool2x2_stride2x2_valid": sequential(
        (100, 100, 1),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
    ),
    "model057_avgpool_inp1x100x100x4_pool2x2_stride2x2_valid": sequential(
        (100, 100, 4),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
    ),
    "model058_avgpool_inp1x134x158x5_pool2x2_stride2x2_valid": sequential(
        (134, 158, 5),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
    ),
    "model065_SmartAve_1x25x5x4": sequential(
        (25, 5, 4),
        layers.AveragePooling2D(pool_size=(25, 5), strides=(25, 5), padding="valid"),
    ),
    "model066_SmartAve_1x5x25x4": sequential(
        (5, 25, 4),
        layers.AveragePooling2D(pool_size=(5, 25), strides=(5, 25), padding="valid"),
    ),
    "model117_avgpool_valid_model70_L1": sequential(
        (134, 158, 5),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
    ),
    "model118_avgpool_valid_model70_L3": sequential(
        (65, 77, 10),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
    ),
    "model119_avgpool_valid_model70_L6": sequential(
        (30, 36, 15),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
    ),
}
