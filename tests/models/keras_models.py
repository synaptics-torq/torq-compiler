import tensorflow as tf

from dataclasses import dataclass
from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, versioned_unhashable_object_fixture


@dataclass
class ConvModelParams:
    width: int
    height: int
    filters: int
    kernel_size: int
    input_channels: int

    def idfn(val):
        return f"w{val.width}_h{val.height}_f{val.filters}_k{val.kernel_size}_c{val.input_channels}"


conv_model_params = [ConvModelParams(12, 12, 5, 1, 64), ConvModelParams(100, 100, 5, 6, 4)]


@versioned_hashable_object_fixture
def keras_model_params(case_config):
    return case_config['keras_model_params']


@versioned_unhashable_object_fixture
def conv_model(keras_model_params):

    # make sure tests use reproducible weights
    tf.keras.utils.set_random_seed(21321)

    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(keras_model_params.height, keras_model_params.width, keras_model_params.input_channels)),
        tf.keras.layers.Conv2D(filters=keras_model_params.filters, kernel_size=keras_model_params.kernel_size, padding='same')
    ])


@versioned_unhashable_object_fixture
def conv_act_model():

    # make sure tests use reproducible weights
    tf.keras.utils.set_random_seed(21321)

    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same'),
        tf.keras.layers.ReLU(6.0)
    ])


@versioned_unhashable_object_fixture
def transpose_conv_model():

    # make sure tests use reproducible weights
    tf.keras.utils.set_random_seed(21321)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 8, 4)),
        tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=4, strides=2, padding='same')
    ])

@versioned_unhashable_object_fixture
def model001_conv_3X3_small():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on JSON summary:
    # - Input: (1, 24, 32, 1)
    # - Single CONV_2D op with:
    #     filters = 4  (from weight tensor shape [4, 3, 3, 1])
    #     kernel_size = (3, 3)
    #     strides = (2, 2)
    #     padding = 'same'
    #     activation = RELU6 (implemented as separate ReLU(6.0) layer)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(24, 32, 1)),
        tf.keras.layers.Conv2D(
            filters=4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same'
        ),
        tf.keras.layers.ReLU(6.0),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model002_conv_2x3_small_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # TFLite analysis:
    # Input:  (1, 3, 4, 2)
    # Op: CONV_2D, padding=VALID, stride=(1,1), activation=NONE
    # Output: (1, 2, 2, 3)
    #
    # For VALID conv: H_out = H_in - kH + 1, W_out = W_in - kW + 1
    # 2 = 3 - kH + 1 -> kH = 2
    # 2 = 4 - kW + 1 -> kW = 3
    # Output channels = 3 -> filters=3
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 4, 2)),
        tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=(2, 3),
            strides=(1, 1),
            padding='valid',
            use_bias=True
        )
        # activation is NONE, so no extra activation layer
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model003_hello_world():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input: shape (1,)
    # 1st Dense: units=16, activation=RELU
    # 2nd Dense: units=16, activation=RELU
    # 3rd Dense: units=1, activation=None (linear)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1,)),
        tf.keras.layers.Dense(16, use_bias=True, name="dense_2"),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(16, use_bias=True, name="dense_3"),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(1, use_bias=True, name="dense_4"),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model004_conv_3x3_valid_bias():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 3, 4, 2)
    # Op: CONV_2D with:
    #   - filters: 3 (from bias shape [3] / output channels)
    #   - kernel_size: (3, 3)
    #   - strides: (1, 1)
    #   - padding: 'valid'
    #   - activation: RELU (as separate ReLU layer)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 4, 2)),
        tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            use_bias=True
        ),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model005_conv2d_4x3x3x3_padding():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture from JSON summary:
    # - Input:  (1, 3, 4, 3)
    # - CONV_2D with:
    #     filters = 4 (from weight tensor shape [4, 3, 3, 3])
    #     kernel_size = (3, 3)
    #     strides = (1, 1)
    #     padding = "SAME"
    #     activation = RELU (as a separate layer)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 4, 3)),
        tf.keras.layers.Conv2D(
            filters=4,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=True
        ),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model006_conv2d_4x6x6x1_pad_stride2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on the JSON summary:
    # - Input: (1, 13, 35, 1)
    # - Single Conv2D:
    #     filters = 4  (from weight tensor shape [4, 6, 6, 1])
    #     kernel_size = (6, 6)
    #     strides = (2, 2)
    #     padding = "SAME"
    #     activation = RELU (separate layer)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(13, 35, 1)),
        tf.keras.layers.Conv2D(
            filters=4,
            kernel_size=(6, 6),
            strides=(2, 2),
            padding='same'
        ),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model007_conv2d_8x3x3x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on JSON summary
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(12, 14, 4)),
        tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1)
        ),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model008_conv2d_4x4x4x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on JSON summary
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(12, 14, 4)),
        tf.keras.layers.Conv2D(
            filters=4,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding='same'
        ),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model009_conv_3x3_same_bias():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input: (1, 4, 4, 1) int8 -> modeled as float32 in Keras
    # Op: CONV_2D with SAME padding, stride 1x1, kernel 3x3, filters=1, activation=RELU
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1)),
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=True
        ),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model010_depthwise_1x3x3x16_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture from ops_structure:
    # Input: [1, 6, 42, 16]
    # Op: DEPTHWISE_CONV_2D with:
    #   kernel: 3x3, stride 1x1, padding VALID, depth_multiplier=1, activation=RELU
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(6, 42, 16)),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            depth_multiplier=1
        ),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model011_depthwise_1x3x3x16_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input: (1, 6, 4, 16) int8 in TFLite -> float32 here
    # Op: DEPTHWISE_CONV_2D with:
    #   kernel: 3x3
    #   strides: (1, 1)
    #   padding: VALID
    #   depth_multiplier: 1
    #   activation: RELU (separate layer)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(6, 4, 16)),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            depth_multiplier=1
        ),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model012_depthwise_1x3x3x2_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(6, 6, 2)),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            padding="valid",
            strides=(1, 1),
            depth_multiplier=1,
            use_bias=True
        ),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model013_depthwise_1x3x3x2_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on JSON summary
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(6, 6, 2)),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1),
            depth_multiplier=1
        ),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model014_depthwise_1x3x3x8_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 13, 15, 8)
    # Op:     DEPTHWISE_CONV_2D with:
    #         - padding: VALID
    #         - strides: (1, 1)
    #         - depth_multiplier: 1
    #         - kernel_size: (3, 3)
    #         - activation: RELU (added as a separate layer)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(13, 15, 8)),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            depth_multiplier=1
        ),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model016_pointwise_8x8x8x16_stride2x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on the JSON summary:
    # - Input:  (1, 8, 8, 16)
    # - Op:     CONV_2D
    #   * filters: 8        (from weight tensor shape [8, 1, 1, 16])
    #   * kernel: 1x1
    #   * strides: 2x2
    #   * padding: SAME
    #   * activation: NONE (no separate activation layer)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 8, 16)),
        tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(1, 1),
            strides=(2, 2),
            padding='same',
            use_bias=True
        ),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model017_pointwise_8x8x9x16_stride2x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on JSON summary:
    # Input:  (1, 8, 9, 16), int8
    # Op:     CONV_2D, padding=SAME, stride=(2,2), kernel_size=(1,1), filters=8, no activation
    # Output: (1, 4, 5, 8), int8
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 9, 16)),
        tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(1, 1),
            strides=(2, 2),
            padding="same"
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model018_pointwise_8x9x8x16_stride2x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(9, 8, 16)),
        tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(1, 1),
            strides=(2, 2),
            padding='same',
            use_bias=True
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model019_pointwise_8x9x9x16_stride2x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(9, 9, 16)),
        tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(1, 1),
            strides=(2, 2),
            padding='same',
            use_bias=True
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model020_fc_1x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # [LLM: RECONSTRUCT THE ARCHITECTURE HERE BASED ON THE JSON SUMMARY]
    # Use tf.keras.Sequential with explicit layers
    # Example structure:
    # return tf.keras.Sequential([
    #     tf.keras.layers.Input(batch_size=1, shape=(28, 28, 1)),
    #     tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same'),
    #     tf.keras.layers.ReLU(6.0),  # Use ReLU(6.0) for RELU6 activation
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(64),
    #     tf.keras.layers.ReLU(),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1,)),
        tf.keras.layers.Dense(1),
        tf.keras.layers.ReLU(6.0)
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model022_fc_1024x1024():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  [1, 1024] int8
    # Op:     FULLY_CONNECTED (Dense) 1024 -> 1024, no activation
    # Output: [1, 1024] int8 (after quantization)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1024,)),
        tf.keras.layers.Dense(1024, use_bias=True),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model023_fc_512x1000():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 1000)
    # Op:     FULLY_CONNECTED with 512 units and RELU6 activation
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1000,)),
        tf.keras.layers.Dense(512, activation=None),
        tf.keras.layers.ReLU(6.0),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model024_fc_97x2000():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(2000,)),
        tf.keras.layers.Dense(97, use_bias=True),
        tf.keras.layers.ReLU(6.0)
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model026_fc_1991x61():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 61)
    # Op:     FULLY_CONNECTED with 1991 units, RELU activation
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(61,)),
        tf.keras.layers.Dense(1991, use_bias=True),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model027_fc_500x700():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on the JSON summary
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(700,)),
        tf.keras.layers.Dense(500, use_bias=True),
        tf.keras.layers.Activation('sigmoid')
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model029_fc_1000x1000():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture from TFLite JSON summary:
    # Input: [1, 1000] -> Fully Connected (1000 units, no activation) -> Softmax -> Output: [1, 1000]
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1000,)),
        tf.keras.layers.Dense(1000, activation=None),
        tf.keras.layers.Softmax()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model030_avgpool_inp1x10x10x4_pool3x3_stride3x3_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 10, 10, 4)
    # Op: AVERAGE_POOL_2D with pool_size=(3,3), strides=(3,3), padding='VALID'
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(3, 3),
            strides=(3, 3),
            padding='valid'
        ),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model031_avgpool_inp1x10x10x4_pool4x4_stride4x4_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 10, 10, 4)
    # Op:     AVERAGE_POOL_2D
    #   - filter_height = 4, filter_width = 4  -> pool_size=(4, 4)
    #   - stride_h = 4, stride_w = 4          -> strides=(4, 4)
    #   - padding = "VALID"                   -> padding="valid"
    #   - activation = "NONE"                 -> no extra activation layer
    # Output: (1, 2, 2, 4)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(4, 4),
                                         strides=(4, 4),
                                         padding='valid'),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model032_avgpool_inp1x10x10x4_pool5x5_stride5x5_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 10, 10, 4)
    # Op: AVERAGE_POOL_2D with
    #   filter_height = 5, filter_width = 5  -> pool_size=(5, 5)
    #   stride_h = 5, stride_w = 5          -> strides=(5, 5)
    #   padding = "VALID"                   -> padding="valid"
    #   activation = "NONE"                 -> no activation layer
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(5, 5),
            strides=(5, 5),
            padding='valid'
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model033_avgpool_inp1x10x10x4_pool5x5_stride1x1_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 10, 10, 4)
    # Op: AVERAGE_POOL_2D with:
    #   filter_height = 5, filter_width = 5  -> pool_size=(5, 5)
    #   stride_h = 5, stride_w = 5          -> strides=(5, 5)
    #   padding = "VALID"                   -> padding="valid"
    #   activation = "NONE"                 -> no activation layer
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(5, 5), padding="valid"),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model035_avgpool_inp1x10x10x4_pool3x3_stride3x3_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture from JSON summary:
    # Input:  (1, 10, 10, 4)
    # Op: AVERAGE_POOL_2D with:
    #   padding = "SAME"
    #   stride_h = 3, stride_w = 3
    #   filter_height = 3, filter_width = 3
    #   activation = NONE
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(3, 3),
            strides=(3, 3),
            padding='same'
        ),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model036_avgpool_inp1x10x10x4_pool4x4_stride4x4_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input: (1, 10, 10, 4), dtype int8 in TFLite (float32 in Keras prior to quantization)
    # Operation: AVERAGE_POOL_2D with:
    #   - pool_size: (4, 4)
    #   - strides: (4, 4)
    #   - padding: 'SAME'
    #   - activation: NONE
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(4, 4),
            strides=(4, 4),
            padding='same'
        ),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model037_avgpool_inp1x10x10x4_pool5x5_stride5x5_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on JSON summary
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(5, 5),
            strides=(5, 5),
            padding='same'
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model039_avgpool_inp1x10x10x4_pool1x3_stride1x1_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 10, 10, 4)
    # Op: AVERAGE_POOL_2D with:
    #   filter_height=1, filter_width=3  -> pool_size=(1, 3)
    #   stride_h=1, stride_w=3          -> strides=(1, 3)
    #   padding="SAME"                  -> padding="same"
    #   activation="NONE"               -> no activation layer
    # Output: (1, 10, 4, 4)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(1, 3),
            strides=(1, 3),
            padding="same"
        ),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model044_maxpool_inp1x10x10x4_pool1x3_stride1x3_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture from TFLite JSON summary:
    # Input:  [1, 10, 10, 4]
    # Op: MAX_POOL_2D with
    #   filter_height = 1, filter_width = 3  -> pool_size = (1, 3)
    #   stride_h = 1, stride_w = 3          -> strides = (1, 3)
    #   padding = "VALID"
    #   activation = "NONE" (no extra activation layer)
    # Output: [1, 10, 3, 4]
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(
            pool_size=(1, 3),
            strides=(1, 3),
            padding='valid'
        ),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model045_maxpool_inp1x10x10x4_pool3x3_stride3x3_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture: single MAX_POOL_2D op
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(
            pool_size=(3, 3),
            strides=(3, 3),
            padding='same'
        ),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model046_maxpool_inp1x10x10x4_pool4x4_stride4x4_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture: Input -> MaxPool2D
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model047_maxpool_inp1x10x10x4_pool5x5_stride5x5_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 10, 10, 4)
    # Op: MAX_POOL_2D with
    #   filter_height = 5, filter_width = 5  -> pool_size = (5, 5)
    #   stride_h = 5, stride_w = 5          -> strides = (5, 5)
    #   padding = "SAME"                    -> padding = "same"
    #   activation = "NONE"                 -> no activation layer
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(
            pool_size=(5, 5),
            strides=(5, 5),
            padding="same"
        ),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model048_maxpool_inp1x10x10x4_pool5x5_stride1x1_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on the JSON summary
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(
            pool_size=(5, 5),
            strides=(5, 5),
            padding='same'
        ),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model049_maxpool_inp1x10x10x4_pool1x3_stride1x1_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on the JSON summary
    # Input:  (1, 10, 10, 4)
    # Op: MAX_POOL_2D with:
    #   filter_height = 1, filter_width = 3  -> pool_size=(1, 3)
    #   stride_h = 1, stride_w = 3          -> strides=(1, 3)
    #   padding = "SAME"                    -> padding="same"
    # Output: (1, 10, 4, 4)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same'),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model051_mult_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Architecture based on ops_structure:
    # 1) Input: (1, 4, 4, 1)
    # 2) CONV_2D with:
    #    - filters = 1  (from weight tensor shape [1, 3, 3, 1])
    #    - kernel_size = (3, 3)
    #    - padding = "SAME"
    #    - strides = (1, 1)
    #    - activation = None
    # 3) MUL op: elementwise multiplication.
    #    In TFLite this multiplies input and conv output; here we
    #    approximate with x * x as a single-input Lambda layer.
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1)),
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            padding="same",
            strides=(1, 1),
            use_bias=True
        ),
        tf.keras.layers.Lambda(lambda x: x * x, name="mul_approx")
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model053_mult_inp1x4x4x1_zp128_B():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on the JSON summary:
    # Input: (1, 4, 4, 1), int8
    # Op 1: CONV_2D with SAME padding, stride 1x1, dilation 1x1, RELU activation
    #       Kernel size inferred as 3x3, with 1 output channel (filters=1)
    # Op 2: MUL between the original input and Conv/ReLU output
    # In Sequential, we approximate MUL as elementwise x * x on the Conv/ReLU output.
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1)),
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1),
            dilation_rate=(1, 1),
            use_bias=True
        ),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Lambda(lambda x: x * x, name="mul_approx")
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model054_mult_inp1x4x4x1_zp128_A():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on JSON summary:
    # - Input: (1, 4, 4, 2)
    # - CONV_2D: filters=2, kernel_size=(2,2), padding='same', strides=(1,1), no activation
    # - MUL: element-wise multiplication (approximated here as output * output)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 2)),
        tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=(2, 2),
            padding='same',
            strides=(1, 1),
            dilation_rate=(1, 1)
        ),
        tf.keras.layers.Lambda(lambda x: x * x, name="mul")
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model055_avgpool_inp1x10x10x4_pool2x2_stride2x2_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Model architecture reconstructed from TFLite ops_structure:
    # Input:  (1, 10, 10, 4)
    # Op: AVERAGE_POOL_2D with:
    #   filter_height = 2, filter_width = 2  -> pool_size=(2, 2)
    #   stride_h = 2, stride_w = 2          -> strides=(2, 2)
    #   padding = "VALID"                   -> padding="valid"
    #   activation = "NONE"                 -> no activation layer
    # Output: (1, 5, 5, 4)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        ),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model056_avgpool_inp1x100x100x1_pool2x2_stride2x2_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input: (1, 100, 100, 1)
    # Op: AVERAGE_POOL_2D with:
    #   padding: VALID
    #   stride_h: 2, stride_w: 2
    #   filter_height: 2, filter_width: 2
    # Output: (1, 50, 50, 1)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(100, 100, 1)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model057_avgpool_inp1x100x100x4_pool2x2_stride2x2_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 100, 100, 4)
    # Op 0:   AVERAGE_POOL_2D with
    #         filter_height = 2, filter_width = 2
    #         stride_h = 2, stride_w = 2
    #         padding = VALID
    #         activation = NONE
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(100, 100, 4)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model058_avgpool_inp1x134x158x5_pool2x2_stride2x2_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on the JSON summary
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(134, 158, 5)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model061_add_inp1x4x4x1_zp128_x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Model architecture reconstructed from TFLite ops_structure:
    # Input: 1x4x4x1
    # CONV_2D (padding=SAME, stride=1, activation=RELU, 1 filter, 3x3 kernel assumed)
    # ADD (modeled as elementwise addition with itself for Sequential compatibility)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1)),
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1)
        ),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Lambda(lambda x: x + x)
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model062_add_inp1x4x4x1_zp128_x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Core functional model matching the TFLite ops_structure:
    # Input -> Conv2D (filters=1, kernel 3x3, SAME, stride 1x1) -> ReLU -> Add(input, conv_out)
    core_input = tf.keras.Input(batch_size=1, shape=(4, 4, 1), name="input_1")
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        dilation_rate=(1, 1),
        use_bias=True,
        name="conv2d"
    )(core_input)
    x = tf.keras.layers.ReLU(name="relu")(x)
    core_output = tf.keras.layers.Add(name="add")([core_input, x])
    core_model = tf.keras.Model(inputs=core_input, outputs=core_output, name="core_model")
    
    # Wrap the core model in a Sequential with an explicit Input layer
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1)),
        core_model
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model065_SmartAve_1x25x5x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(25, 5, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(25, 5), strides=(25, 5), padding='valid'),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model066_SmartAve_1x5x25x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on ops_structure:
    # Input: (1, 5, 25, 4)
    # Operation: AVERAGE_POOL_2D
    #   padding: VALID
    #   stride_h: 5, stride_w: 25
    #   filter_height: 5, filter_width: 25
    #   activation: NONE
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(5, 25, 4)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(5, 25),
            strides=(5, 25),
            padding='valid'
        ),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model077_tanh_zpm26():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on the JSON summary:
    # Input: [1, 256]
    # Operation: TANH
    # Output: [1, 256]
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(256,)),
        tf.keras.layers.Activation('tanh'),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model078_tanh_zpm115():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 256)
    # Op:     TANH
    # Output: (1, 256)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(256,)),
        tf.keras.layers.Activation('tanh'),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model079_tanh_zpm128():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Single TANH operation on a (1, 256) input vector
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(256,)),
        tf.keras.layers.Activation('tanh'),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model080_tanh_zp76():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input: [1, 1000]
    # Op: TANH
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1000,)),
        tf.keras.layers.Activation('tanh'),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model081_tanh_zp127():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input: (1, 40) int8 in TFLite -> Keras input shape (40,)
    # Single TANH operation -> Activation('tanh')
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(40,)),
        tf.keras.layers.Activation('tanh')
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model082_tanh_1x4x6x8():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture: single TANH operation
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 6, 8)),
        tf.keras.layers.Activation('tanh'),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model083_sigmoid_1x4x6x8():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Single LOGISTIC (sigmoid) operation on input tensor of shape (1, 4, 6, 8)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 6, 8)),
        tf.keras.layers.Activation('sigmoid')
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model084_sigmoid_1x1x16x128():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Model: single LOGISTIC (sigmoid) op with input/output shape [1, 1, 16, 128]
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 16, 128)),
        tf.keras.layers.Activation('sigmoid')
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model085_softmax_inp1x1916x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 1916, 2)
    # Op:     SOFTMAX
    # Output: (1, 1916, 2)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1916, 2)),
        tf.keras.layers.Softmax(axis=-1),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model086_sigmoid_zp26():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Model reconstructed from TFLite ops_structure:
    # - Input: shape (1, 1, 256), batch_size=1
    # - Single LOGISTIC op => sigmoid activation applied element-wise
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 1, 256)),
        tf.keras.layers.Activation('sigmoid')
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model087_sigmoid_zp128():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture: Input -> Logistic (Sigmoid) Activation
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(512,)),
        tf.keras.layers.Activation('sigmoid'),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model088_sigmoid_zpm76():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Architecture from TFLite ops_structure:
    # Input: [1, 1, 1, 100] -> Keras shape=(1, 1, 100), batch_size=1
    # Single op: LOGISTIC (i.e., sigmoid activation)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 1, 100), name="input_1"),
        tf.keras.layers.Activation("sigmoid", name="Identity"),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model089_sigmoid_zp128():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture: single LOGISTIC (sigmoid) op on input tensor
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 1, 256)),
        tf.keras.layers.Activation('sigmoid')
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model090_sigmoid_zp16():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture: single LOGISTIC (sigmoid) op
    # Input shape from analysis: [1, 1, 1, 256] (batch, h, w, c)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 1, 256), name="input_1"),
        tf.keras.layers.Activation('sigmoid', name="logistic")
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model091_relu_8x8_inp_1x10x1x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  [1, 10, 1, 1]
    # Op: RELU (elementwise), no other ops
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 1, 1)),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model092_relu_zp26_8x8_inp_1x10x1x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on ops_structure:
    # Input:  [1, 10, 1, 1]
    # Op 0:   RELU
    # Output: [1, 10, 1, 1]
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 1, 1)),
        tf.keras.layers.ReLU(),  # RELU activation
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model097_conv_transpose_1x3_1x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 1, 64, 32) in TFLite -> (1, 64, 32) in Keras (channels_last)
    # Op: TRANSPOSE_CONV with filter shape [32, 1, 3, 32]
    #  -> kernel_size=(1, 3), filters=32, strides=(1, 1), padding='same'
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 64, 32)),
        tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=(1, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model105_pointwise_1x1x1024():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Single 1x1 pointwise Conv2D:
    # Input:  (1, 8, 8, 1024)
    # Weights: (8, 1, 1, 1024) -> 8 output channels, 1x1 kernel, 1024 input channels
    # Output: (1, 8, 8, 8)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 8, 1024)),
        tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(1, 1),
            padding='same',
            strides=(1, 1)
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model109_avgpool_valid_model70_L1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(134, 158, 5)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        ),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model110_avgpool_valid_model70_L3():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on the JSON summary
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(65, 77, 10)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model111_avgpool_valid_model70_L6():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture: AVERAGE_POOL_2D with VALID padding
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(30, 36, 15)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model112_mult_inp1x8x8x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input: 1x8x8x1
    # Op: MUL of input with itself (elementwise square)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 8, 1)),
        tf.keras.layers.Lambda(lambda x: x * x, name="mul_self")
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model113_add_inp1x8x8x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 8, 8, 1)
    # Op: ADD(input, input) -> output (1, 8, 8, 1)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 8, 1)),
        tf.keras.layers.Lambda(lambda x: x + x, name="add_input_to_itself"),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model114_conv_mult_inp1x8x8x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Architecture reconstructed from TFLite ops_structure:
    # Single MUL op: output = input * input (elementwise square)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 8, 1)),
        tf.keras.layers.Lambda(lambda x: x * x, name="mul_input_with_itself")
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model117_mult_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Model reproducing: output = input * input (elementwise MUL)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Lambda(lambda x: x * x, name="mul")
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model118_add_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 3, 3, 1) int8 in TFLite (float32 here, later quantized)
    # Op: ADD with both inputs being the same tensor -> y = x + x
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Lambda(lambda x: x + x, name="add_x_x")
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model121_conv3x3_conv2x2_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on JSON summary:
    # - Input: (1, 3, 3, 1)
    # - Conv2D #1: kernel 3x3, stride 1x1, SAME padding, ReLU activation, 1 filter
    # - Conv2D #2: kernel 2x2, stride 1x1, SAME padding, ReLU activation, 1 filter
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1),
            use_bias=True,
        ),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(2, 2),
            padding='same',
            strides=(1, 1),
            use_bias=True,
        ),
        tf.keras.layers.ReLU(),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model122_conv2x2_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 3, 3, 1)
    # Op: CONV_2D with:
    #   - filters: 1
    #   - kernel_size: 2x2 (from function/model naming)
    #   - padding: SAME
    #   - strides: 1x1
    #   - dilation: 1x1
    #   - activation: RELU (separate ReLU layer)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(2, 2),
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            use_bias=True
        ),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model123_add_mult_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Model architecture based on TFLite ops:
    # Input: (1, 3, 3, 1)
    # Op1: ADD(input, input) -> elementwise add
    # Op2: MUL(add_output, add_output) -> elementwise multiply
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Lambda(lambda x: x + x, name="add"),
        tf.keras.layers.Lambda(lambda x: x * x, name="mul")
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model124_mult_mult_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Model: input -> MUL(input, input) -> MUL(prev, prev)
    # First MUL: square the input
    # Second MUL: square again -> x^4
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Lambda(lambda x: x * x, name="mul_1"),
        tf.keras.layers.Lambda(lambda x: x * x, name="mul_2"),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model125_conv3x3_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 4, 4, 1)
    # Op 0:   CONV_2D with:
    #         - filters = 1 (from weight tensor shape [1, 3, 3, 1])
    #         - kernel_size = (3, 3)
    #         - padding = 'same'
    #         - strides = (1, 1)
    #         - dilation_rate = (1, 1)
    #         - activation = RELU (implemented as separate ReLU layer)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1)),
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1),
            use_bias=True
        ),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model126_mult_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Model: single MUL operation where the input is multiplied by itself
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1)),
        tf.keras.layers.Lambda(lambda x: x * x, name="mul_self")
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model127_add3x3_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Model: y = x + x (ADD of input with itself)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1)),
        tf.keras.layers.Lambda(lambda x: x + x, name="add_input_with_itself")
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model128_conv3x3_mult_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 4, 4, 1), int8 in TFLite (float32 in Keras model before quantization)
    # Ops:
    #   1) CONV_2D: padding=SAME, stride=1x1, kernel=3x3, filters=1, activation=RELU
    #   2) MUL: elementwise multiply (approximated here as x * x on Conv output)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1)),
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            use_bias=True
        ),
        tf.keras.layers.ReLU(),  # RELU activation from ops_structure
        tf.keras.layers.Lambda(lambda x: x * x)  # MUL op approximation
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model130_conv3x3_conv2x2_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on JSON summary:
    # Input: (1, 4, 4, 1)
    # Op 1: CONV_2D, filters=1, kernel_size=(3,3), strides=(1,1), padding='same', activation=RELU
    # Op 2: CONV_2D, filters=1, kernel_size=(2,2), strides=(1,1), padding='same', activation=RELU
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1)),
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same'
        ),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(2, 2),
            strides=(1, 1),
            padding='same'
        ),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model131_conv2x2_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture from JSON summary:
    # - Input:  (1, 4, 4, 1)
    # - Single CONV_2D with:
    #     * padding: SAME
    #     * stride_h = 1, stride_w = 1
    #     * activation: RELU (as separate layer)
    #     * kernel size inferred as 2x2 from model name and shapes
    #     * number of filters inferred as 1 from bias shape [1]
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1)),
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(2, 2),
            strides=(1, 1),
            padding="same",
            dilation_rate=(1, 1),
            use_bias=True,
        ),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model132_conv_inp1x32x32x16_16x3x3_same_stride1x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 32, 32, 16)
    # Op:     CONV_2D with:
    #         filters = 16 (from weight tensor shape [16, 3, 3, 16])
    #         kernel_size = (3, 3)
    #         padding = 'SAME'
    #         strides = (1, 1)
    #         dilation_rate = (1, 1)
    #         activation = NONE (no separate activation)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(32, 32, 16)),
        tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            use_bias=True
        ),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model143_transpose_ChLastToFirst_1x8x10x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on the JSON summary:
    # Single TRANSPOSE op: input (1, 8, 10, 4) -> output (1, 4, 8, 10)
    # This corresponds to permuting (N, H, W, C) -> (N, C, H, W)
    # Keras Permute operates on feature dims only, so (H, W, C) -> (C, H, W) = (3, 1, 2)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 10, 4)),
        tf.keras.layers.Permute((3, 1, 2)),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model144_transpose_ChFirstToLast_1x8x10x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Model architecture reconstructed from TFLite TRANSPOSE op
    # Input:  (1, 8, 10, 4)
    # Output: (1, 10, 4, 8)
    # Permutation (including batch): [0, 2, 3, 1]
    # Corresponding Keras Permute over (H, W, C): (2, 3, 1)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 10, 4)),
        tf.keras.layers.Permute((2, 3, 1)),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model145_transpose_2d_matrix_1x4x6x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Single TRANSPOSE op with perm [0, 2, 1, 3] to map (1, 4, 6, 2) -> (1, 6, 4, 2)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 6, 2)),
        tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))
    ])

@versioned_unhashable_object_fixture
def model034_avgpool_inp1x10x10x4_pool1x3_stride1x3_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on the provided TFLite JSON summary:
    # Input:  (1, 10, 10, 4)
    # Op: AVERAGE_POOL_2D with
    #   - filter_height = 1, filter_width = 3  -> pool_size = (1, 3)
    #   - stride_h = 1, stride_w = 3          -> strides = (1, 3)
    #   - padding = "VALID"
    #   - activation = "NONE" (no activation layer)
    # Output: (1, 10, 3, 4)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(1, 3),
            strides=(1, 3),
            padding='valid'
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model040_maxpool_inp1x10x10x4_pool3x3_stride3x3_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(
            pool_size=(3, 3),
            strides=(3, 3),
            padding='valid'
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model041_maxpool_inp1x10x10x4_pool4x4_stride4x4_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(
            pool_size=(4, 4),
            strides=(4, 4),
            padding='valid'
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model042_maxpool_inp1x10x10x4_pool5x5_stride5x5_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture based on JSON summary:
    # Input: (1, 10, 10, 4)
    # Op: MAX_POOL_2D with filter_height=5, filter_width=5, stride_h=5, stride_w=5, padding=VALID
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(
            pool_size=(5, 5),
            strides=(5, 5),
            padding='valid'
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model043_maxpool_inp1x10x10x4_pool5x5_stride1x1_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture: single MAX_POOL_2D layer
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(
            pool_size=(5, 5),
            strides=(5, 5),
            padding='valid'
        )
    ])

@versioned_unhashable_object_fixture
def model021_fc_1991x61():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 61) int8  -> modeled as float32 in Keras with fixed batch_size=1
    # Op:     FULLY_CONNECTED with RELU activation
    # We map this to: Dense(1991) followed by ReLU
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(61,)),
        tf.keras.layers.Dense(1991),
        tf.keras.layers.ReLU(),  # RELU activation from TFLite op
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model038_avgpool_inp1x10x10x4_pool5x5_stride1x1_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Single AVERAGE_POOL_2D operation:
    # - Input:  (1, 10, 10, 4)
    # - Output: (1, 2, 2, 4)
    # - filter_height = 5, filter_width = 5
    # - stride_h = 5, stride_w = 5
    # - padding = SAME
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(5, 5), padding="same"),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model050_mult_inp1x10x10x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API model with a Conv2D followed by element-wise multiplication
    # between the original input and the Conv2D output.
    inputs = tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4))
    
    x = tf.keras.layers.Conv2D(
        filters=4,
        kernel_size=(4, 4),
        strides=(1, 1),
        padding="same",
        use_bias=True,
        activation=None,
    )(inputs)
    
    outputs = tf.keras.layers.Multiply()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model052_mult_inp1x4x4x1_zp128_AB():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API: input -> Conv2D -> ReLU -> Multiply(input, conv_output)
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1), name="input_1")
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=True,
    )(inputs)
    x = tf.keras.layers.ReLU()(x)
    outputs = tf.keras.layers.Multiply()([inputs, x])
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="model052_mult_inp1x4x4x1_zp128_AB")

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model060_add_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API model with Conv2D followed by residual ADD with input
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1))
    
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        use_bias=True
    )(inputs)
    
    outputs = tf.keras.layers.Add()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model073_relu_second_verifier_light_int():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API model reconstruction based on the provided TFLite structure
    inputs = tf.keras.layers.Input(batch_size=1, shape=(65, 52, 1), name="input_2")
    
    # CONV_2D (functional_1/conv2d_2) with RELU, VALID padding, strides (1,1), kernel (3,3), filters=12
    x = tf.keras.layers.Conv2D(
        filters=12,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="conv2d_2"
    )(inputs)
    x = tf.keras.layers.ReLU(name="conv2d_2_relu")(x)
    
    # AVERAGE_POOL_2D (functional_1/average_pooling2d_3) pool (2,2), strides (2,2), VALID
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        name="average_pooling2d_3"
    )(x)
    
    # DEPTHWISE_CONV_2D (functional_1/separable_conv2d_1 depthwise), kernel (3,3), depth_multiplier=1, VALID
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        depth_multiplier=1,
        use_bias=True,
        name="separable_conv2d_1_depthwise"
    )(x)
    
    # Pointwise CONV_2D (functional_1/separable_conv2d_1 pointwise) with RELU, kernel (1,1), filters=40
    x = tf.keras.layers.Conv2D(
        filters=40,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="separable_conv2d_1_pointwise"
    )(x)
    x = tf.keras.layers.ReLU(name="separable_conv2d_1_relu")(x)
    
    # AVERAGE_POOL_2D (functional_1/average_pooling2d_4) pool (2,2), strides (2,2), VALID
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        name="average_pooling2d_4"
    )(x)
    
    # DEPTHWISE_CONV_2D (functional_1/separable_conv2d_2 depthwise), kernel (3,3), depth_multiplier=1, VALID
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        depth_multiplier=1,
        use_bias=True,
        name="separable_conv2d_2_depthwise"
    )(x)
    
    # Pointwise CONV_2D (functional_1/separable_conv2d_2 pointwise) with RELU, kernel (1,1), filters=80
    x = tf.keras.layers.Conv2D(
        filters=80,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="separable_conv2d_2_pointwise"
    )(x)
    x = tf.keras.layers.ReLU(name="separable_conv2d_2_relu")(x)
    
    # AVERAGE_POOL_2D (functional_1/average_pooling2d_5) pool (2,2), strides (2,2), VALID
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        name="average_pooling2d_5"
    )(x)
    
    # CONV_2D (functional_1/conv2d_3) with RELU, kernel (6,4), filters=64, VALID
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(6, 4),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="conv2d_3"
    )(x)
    x = tf.keras.layers.ReLU(name="conv2d_3_relu")(x)
    
    # CONV_2D (functional_1/conv2d_4) with RELU, kernel (1,1), filters=32, VALID
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="conv2d_4"
    )(x)
    x = tf.keras.layers.ReLU(name="conv2d_4_relu")(x)
    
    # Head 1: CONV_2D (functional_1/conv2d_5) to 4 channels, kernel (1,1), VALID
    out_main = tf.keras.layers.Conv2D(
        filters=4,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="conv2d_5"
    )(x)
    
    # Head 2: CONV_2D (functional_1/conv2d_6) to 1 channel, kernel (1,1), VALID
    out_aux1 = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="conv2d_6"
    )(x)
    
    # Head 3: CONV_2D (functional_1/conv2d_7) to 1 channel, kernel (1,1), VALID
    out_aux2 = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="conv2d_7"
    )(x)
    
    return tf.keras.Model(
        inputs=inputs,
        outputs=[out_main, out_aux1, out_aux2],
        name="model073_relu_second_verifier_light_int"
    )

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model076_fc_7x3():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Model structure from TFLite:
    # - Input:  (1, 3), int8
    # - Op: FULLY_CONNECTED with activation RELU
    #   * Weight tensor shape: (7, 3)
    #   * Bias tensor shape:   (7,)
    # - Output: (1, 7), int8
    #
    # This corresponds to:
    # Input -> Dense(7 units, no activation) -> ReLU
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3,)),
        tf.keras.layers.Dense(7),
        tf.keras.layers.ReLU()
    ])

@versioned_unhashable_object_fixture
def model021_fc_1991x61():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 61) int8
    # Op:    FULLY_CONNECTED with 1991 units, RELU activation
    # Output: (1, 1991) int8
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(61,)),
        tf.keras.layers.Dense(1991),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model038_avgpool_inp1x10x10x4_pool5x5_stride1x1_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Single-input, single-output linear model:
    # AVERAGE_POOL_2D with:
    # - input shape: (1, 10, 10, 4)
    # - filter_height = 5, filter_width = 5  -> pool_size = (5, 5)
    # - stride_h = 5, stride_w = 5          -> strides = (5, 5)
    # - padding = "SAME"                    -> padding = "same"
    # - activation = "NONE"                 -> no activation layer
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(
            pool_size=(5, 5),
            strides=(5, 5),
            padding='same'
        )
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model050_mult_inp1x10x10x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4))
    x = tf.keras.layers.Conv2D(
        filters=4,
        kernel_size=(4, 4),
        strides=(1, 1),
        padding='same',
        dilation_rate=(1, 1),
        use_bias=True
    )(inputs)
    outputs = tf.keras.layers.Multiply()([inputs, x])
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model052_mult_inp1x4x4x1_zp128_AB():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API reconstruction based on TFLite ops_structure:
    # Input:  (1, 4, 4, 1), int8
    # Op 1: CONV_2D with RELU activation, SAME padding, stride 1x1
    # Op 2: MUL between original input and Conv+ReLU output
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1))
    
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=True
    )(inputs)
    x = tf.keras.layers.ReLU()(x)
    
    outputs = tf.keras.layers.Multiply()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model060_add_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API model with Conv2D followed by Add with the input
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1))
    
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        use_bias=True
    )(inputs)
    
    outputs = tf.keras.layers.Add()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model064_model():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API: single input, add a trainable constant vector (bias)
    inputs = tf.keras.layers.Input(batch_size=1, shape=(32,), name="Input_1")
    outputs = AddConstLayer(name="add_const")(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="model064")


# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model069_model():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API reconstruction:
    # TFLite structure:
    # - Single input: shape (1, 32), int8
    # - Single op: ADD(input, Const[1, 32]) -> output (1, 32)
    #
    # We model this as a single-input Keras model where the output is
    # the input plus a constant vector, implemented via a Lambda layer.
    inputs = tf.keras.layers.Input(batch_size=1, shape=(32,))
    
    const_vector = tf.constant(0.0, shape=(32,), dtype=tf.float32)
    
    def add_const(x):
        return x + const_vector
    
    outputs = tf.keras.layers.Lambda(add_const, name="add_const")(inputs)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="model069")

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model071_HM11B1_relu_proposer_2_int():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API model reconstruction based on provided TFLite ops_structure
    inputs = tf.keras.layers.Input(batch_size=1, shape=(92, 80, 1))
    
    # CONV_2D (filters=5, kernel=3x3, stride=1, padding=VALID) + RELU
    x = tf.keras.layers.Conv2D(
        filters=5,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        use_bias=True
    )(inputs)
    x = tf.keras.layers.ReLU()(x)
    
    # AVERAGE_POOL_2D (pool=2x2, stride=2, padding=VALID)
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid'
    )(x)
    
    # CONV_2D (filters=10, kernel=3x3, stride=1, padding=VALID) + RELU
    x = tf.keras.layers.Conv2D(
        filters=10,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        use_bias=True
    )(x)
    x = tf.keras.layers.ReLU()(x)
    
    # AVERAGE_POOL_2D (pool=2x2, stride=2, padding=VALID)
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid'
    )(x)
    
    # DEPTHWISE_CONV_2D (kernel=3x3, depth_multiplier=1, stride=1, padding=VALID), no activation
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        depth_multiplier=1,
        use_bias=True
    )(x)
    
    # Pointwise CONV_2D in separable block (filters=15, kernel=1x1, stride=1, padding=VALID) + RELU
    x = tf.keras.layers.Conv2D(
        filters=15,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=True
    )(x)
    x = tf.keras.layers.ReLU()(x)
    
    # AVERAGE_POOL_2D (pool=2x2, stride=2, padding=VALID)
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid'
    )(x)
    
    # CONV_2D (filters=64, kernel=4x3, stride=1, padding=VALID) + RELU
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(4, 3),
        strides=(1, 1),
        padding='valid',
        use_bias=True
    )(x)
    x = tf.keras.layers.ReLU()(x)
    
    # CONV_2D (filters=32, kernel=1x1, stride=1, padding=VALID) + RELU
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=True
    )(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Branch 1: CONV_2D (filters=4, kernel=1x1, stride=1, padding=VALID), no activation
    out1 = tf.keras.layers.Conv2D(
        filters=4,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=True
    )(x)
    
    # Branch 2: CONV_2D (filters=1, kernel=1x1, stride=1, padding=VALID), no activation
    out2 = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        use_bias=True
    )(x)
    
    return tf.keras.Model(inputs=inputs, outputs=[out1, out2])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model073_relu_second_verifier_light_int():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional model with single input and three output heads
    inputs = tf.keras.layers.Input(batch_size=1, shape=(65, 52, 1), name="input_2")
    
    # CONV_2D: filters=12, kernel_size=(3,3), strides=(1,1), padding='VALID', activation=RELU
    x = tf.keras.layers.Conv2D(
        filters=12,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="conv2d_2"
    )(inputs)
    x = tf.keras.layers.ReLU(name="conv2d_2_relu")(x)
    
    # AVERAGE_POOL_2D: pool_size=(2,2), strides=(2,2), padding='VALID'
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        name="average_pooling2d_3"
    )(x)
    
    # DEPTHWISE_CONV_2D: kernel_size=(3,3), depth_multiplier=1, strides=(1,1), padding='VALID', no activation
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        depth_multiplier=1,
        use_bias=True,
        name="separable_conv2d_1_depthwise"
    )(x)
    
    # Pointwise CONV_2D: filters=40, kernel_size=(1,1), strides=(1,1), padding='VALID', activation=RELU
    x = tf.keras.layers.Conv2D(
        filters=40,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="separable_conv2d_1_pointwise"
    )(x)
    x = tf.keras.layers.ReLU(name="separable_conv2d_1_relu")(x)
    
    # AVERAGE_POOL_2D: pool_size=(2,2), strides=(2,2), padding='VALID'
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        name="average_pooling2d_4"
    )(x)
    
    # DEPTHWISE_CONV_2D: kernel_size=(3,3), depth_multiplier=1, strides=(1,1), padding='VALID', no activation
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        depth_multiplier=1,
        use_bias=True,
        name="separable_conv2d_2_depthwise"
    )(x)
    
    # Pointwise CONV_2D: filters=80, kernel_size=(1,1), strides=(1,1), padding='VALID', activation=RELU
    x = tf.keras.layers.Conv2D(
        filters=80,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="separable_conv2d_2_pointwise"
    )(x)
    x = tf.keras.layers.ReLU(name="separable_conv2d_2_relu")(x)
    
    # AVERAGE_POOL_2D: pool_size=(2,2), strides=(2,2), padding='VALID'
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        name="average_pooling2d_5"
    )(x)
    
    # CONV_2D: filters=64, kernel_size=(6,4), strides=(1,1), padding='VALID', activation=RELU
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(6, 4),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="conv2d_3"
    )(x)
    x = tf.keras.layers.ReLU(name="conv2d_3_relu")(x)
    
    # CONV_2D: filters=32, kernel_size=(1,1), strides=(1,1), padding='VALID', activation=RELU
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="conv2d_4"
    )(x)
    x = tf.keras.layers.ReLU(name="conv2d_4_relu")(x)
    
    # Output head 1: CONV_2D: filters=4, kernel_size=(1,1), no activation
    out1 = tf.keras.layers.Conv2D(
        filters=4,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="conv2d_5"
    )(x)
    
    # Output head 2: CONV_2D: filters=1, kernel_size=(1,1), no activation
    out2 = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="conv2d_6"
    )(x)
    
    # Output head 3: CONV_2D: filters=1, kernel_size=(1,1), no activation
    out3 = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="conv2d_7"
    )(x)
    
    return tf.keras.Model(inputs=inputs, outputs=[out1, out2, out3], name="model073_relu_second_verifier_light_int")

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model076_fc_7x3():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Reconstructed architecture:
    # Input:  (1, 3) int8  -> Keras Input(batch_size=1, shape=(3,))
    # Op:     FULLY_CONNECTED with weights shape (7, 3), bias shape (7)
    # Output: (1, 7) int8
    # Activation: RELU (separate ReLU layer)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3,)),
        tf.keras.layers.Dense(7),
        tf.keras.layers.ReLU()
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model103_transpose_first2last():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # The TFLite model performs a TRANSPOSE from [batch, 8, 32, 48] to [batch, 32, 48, 8]
    # This corresponds to permuting the non-batch dimensions (1, 2, 3) -> (2, 3, 1),
    # i.e., tf.transpose(x, perm=[0, 2, 3, 1]).
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=10, shape=(8, 32, 48)),
        tf.keras.layers.Permute((2, 3, 1)),
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model115_conv_add_inp1x8x8x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API with residual ADD:
    # Input: (1, 8, 8, 1)
    # Conv2D: filters=1, kernel_size=(3,3), padding='same', strides=(1,1), activation=RELU
    # Output: Add(input, relu(conv(input)))
    inputs = tf.keras.layers.Input(batch_size=1, shape=(8, 8, 1))
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(inputs)
    x = tf.keras.layers.ReLU()(x)
    outputs = tf.keras.layers.Add()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model116_conv3x3_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Model structure from TFLite analysis:
    # - Input:  (1, 3, 3, 1), int8
    # - Op:     CONV_2D with RELU activation, SAME padding, stride 1x1
    # - Output: (1, 3, 3, 1), int8
    #
    # Single linear chain -> use Sequential API.
    # Conv2D parameters inferred:
    # - filters: 1  (from output channel dimension)
    # - kernel_size: (3, 3)  (from model name / conv3x3, and typical setup)
    # - strides: (1, 1)
    # - padding: 'same'
    # - activation: separate ReLU layer
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=True,
        ),
        tf.keras.layers.ReLU(),  # RELU activation
    ])

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model119_conv3x3_mult_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API reconstruction based on ops_structure:
    # Input: (1, 3, 3, 1), dtype int8
    # 1) CONV_2D with RELU activation, SAME padding, stride 1x1
    #    - Kernel size inferred as 3x3 from context (conv3x3)
    #    - Filters inferred as 1 from output tensor shape (1, 3, 3, 1)
    # 2) MUL: element-wise multiply of original input and Conv+ReLU output
    inputs = tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1))
    
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        use_bias=True
    )(inputs)
    
    x = tf.keras.layers.ReLU()(x)  # RELU activation as separate layer
    
    outputs = tf.keras.layers.Multiply()([inputs, x])  # Element-wise MUL op
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model120_conv3x3_add_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API model with Conv2D followed by ReLU and residual ADD with input
    inputs = tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1))
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=True
    )(inputs)
    x = tf.keras.layers.ReLU()(x)
    outputs = tf.keras.layers.Add()([inputs, x])
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# --- BUILD AND INITIALIZE MODEL ---


@versioned_unhashable_object_fixture
def model129_conv3x3_add_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API model with Conv2D (RELU) and residual ADD with input
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1))
    
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        padding='same',
        strides=(1, 1),
        dilation_rate=(1, 1),
        use_bias=True
    )(inputs)
    x = tf.keras.layers.ReLU()(x)  # RELU activation from ops_structure
    
    outputs = tf.keras.layers.Add()([inputs, x])  # ADD op: input + conv output
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
