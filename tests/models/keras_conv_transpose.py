import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, versioned_unhashable_object_fixture


@versioned_unhashable_object_fixture
def model100_conv_transpose_1x3_1x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 32, 48)),
        tf.keras.layers.Conv2DTranspose(
            filters=48,
            kernel_size=(1, 3),
            strides=(1, 2),
            padding='same',
            use_bias=False
        )
    ])



@versioned_unhashable_object_fixture
def model101_conv_transpose_1x3_1x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 64, 32)),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)
    ])



@versioned_unhashable_object_fixture
def model102_conv_transpose_stride1_ker_1x3_input_1x1x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 1)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=False)
    ])



@versioned_unhashable_object_fixture
def model103_conv_transpose_stride2_ker_1x3_1x1x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 1)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 3), strides=(1, 2), padding='same', use_bias=False)
    ])



@versioned_unhashable_object_fixture
def model104_conv_transpose_1x3_stride2_input_1x1x12x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(12, 4)),
        # The TFLite ops show a Reshape -> Conv2DTranspose -> Reshape pattern,
        # which is the low-level implementation of a Conv1DTranspose.
        # We use the more direct Keras layer here.
        tf.keras.layers.Conv1DTranspose(filters=4, kernel_size=3, strides=2, padding='same', use_bias=False)
    ])



@versioned_unhashable_object_fixture
def model105_conv_transpose_stride1_ker_1x3_input1x1x4x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 2)),
        tf.keras.layers.Conv2DTranspose(
            filters=2,
            kernel_size=(1, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False
        )
    ])



@versioned_unhashable_object_fixture
def model500_conv_transpose_stride1_ker_1x1x3x1_padSame():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 1)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model501_conv_transpose_stride1x2_ker1x1x3x1_padSame():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 1)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 3), strides=(1, 2), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model502_conv_transpose_stride1_ker2x1x3x1_padSame():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 1)),
        tf.keras.layers.Conv2DTranspose(
            filters=2,
            kernel_size=(1, 3),
            strides=(1, 1),
            padding='same',
            use_bias=True
        )
    ])



@versioned_unhashable_object_fixture
def model503_conv_transpose_stride1_ker_1x1x3x1_padValid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 1)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 3), strides=(1, 1), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model504_conv_transpose_stride1x2_ker1x1x3x1_padValid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 1)),
        tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(1, 3),
            strides=(1, 2),
            padding='valid',
            use_bias=True
        )
    ])



@versioned_unhashable_object_fixture
def model505_conv_transpose_stride1_ker2x1x3x1_padValid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 1)),
        tf.keras.layers.Conv2DTranspose(
            filters=2,
            kernel_size=(1, 3),
            strides=(1, 1),
            padding='valid',
            use_bias=True
        )
    ])



@versioned_unhashable_object_fixture
def model506_conv_transpose_stride1_1x3x3x1_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=True
        )
    ])



@versioned_unhashable_object_fixture
def model507_conv_transpose_stride2_1x3x3x1_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model508_conv_transpose_stride1_3x2x2x2_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 2)),
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(2, 2), strides=(1, 1), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model509_conv_transpose_stride1_2x4x4x3_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(5, 5, 3)),
        tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=(4, 4), strides=(1, 1), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model510_conv_transpose_stride2_3x4x3x2_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(7, 7, 2)),
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(4, 3), strides=(2, 2), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model511_conv_transpose_stride1_16x1x3x16_padsame():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 16)),
        tf.keras.layers.Conv2DTranspose(
            filters=16,
            kernel_size=(1, 3),
            strides=(1, 1),
            padding='same',
            use_bias=True
        )
    ])



@versioned_unhashable_object_fixture
def model512_conv_transpose_stride1_16x1x3x16_padvalid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 16)),
        tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(1, 3), strides=(1, 1), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model513_conv_transpose_stride2_8x1x3x8_padSame():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 8)),
        tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(1, 3), strides=(1, 2), padding="same", use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model514_conv_transpose_stride2_8x1x3x8_padValid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 8)),
        tf.keras.layers.Conv2DTranspose(
            filters=8,
            kernel_size=(1, 3),
            strides=(1, 2),
            padding='valid',
            use_bias=True
        )
    ])



@versioned_unhashable_object_fixture
def model515_conv_transpose_stride1_ker1x1x3x2_padSame():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 2)),
        tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(1, 3),
            strides=(1, 1),
            padding='same',
            use_bias=True
        )
    ])



@versioned_unhashable_object_fixture
def model516_conv_transpose_stride1x2_ker1x1x3x2_padSame():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 2)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 3), strides=(1, 2), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model517_conv_transpose_stride1_ker1x1x3x2_padValid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 2)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 3), strides=(1, 1), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model518_conv_transpose_stride1x2_ker1x1x3x2_padValid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 2)),
        tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(1, 3),
            strides=(1, 2),
            padding='valid',
            use_bias=True
        )
    ])



@versioned_unhashable_object_fixture
def model519_conv_transpose_stride1_1x3x3x1_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model520_conv_transpose_stride2_1x3x3x1_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model521_conv_transpose_stride2_3x3x3x2_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 2)),
        tf.keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='valid',
            use_bias=True
        )
    ])



@versioned_unhashable_object_fixture
def model522_conv_transpose_stride3_1x3x3x1_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(3, 3),
            strides=(3, 3),
            padding='same',
            use_bias=True
        )
    ])



@versioned_unhashable_object_fixture
def model523_conv_transpose_stride3_1x3x3x1_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(3, 3), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model531_convTrans_16x8_inp_1x64x32_ker1x3_stride1_padsame():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 64, 32)),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model532_convTrans_16x8_inp_1x4x1_ker1x3_stride1_padvalid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 1)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 3), strides=(1, 1), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model533_convTrans_16x8_inp_1x4x1_ker1x3_stride2_padvalid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 4, 1)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 3), strides=(1, 2), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model534_convTrans_16x8_inp_1x3x3x1_ker3x3_stride1_padsame():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # The analysis shows a single TRANSPOSE_CONV operation, which is a linear structure.
    # The Sequential API is appropriate.
    # Input shape: [1, 3, 3, 1] -> batch_size=1, shape=(3, 3, 1)
    # Op: TRANSPOSE_CONV
    #   - padding: "SAME" -> padding='same'
    #   - stride_h: 1, stride_w: 1 -> strides=(1, 1)
    #   - use_bias: true -> use_bias=True
    #   - filters and kernel_size are inferred from internal weight tensors.
    #     The weight tensor for Conv2DTranspose is [H, W, Out_C, In_C].
    #     Tensor at index 6 has shape [3, 3, 1, 1], so kernel_size=(3,3) and filters=1.
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=True
        )
    ])



@versioned_unhashable_object_fixture
def model535_convTrans_16x8_inp_1x3x3x1_ker3x3_stride2_padsame():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model536_convTrans_16x8_inp_1x4x4x2_ker2x2_stride1_padsame():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 2)),
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(2, 2), strides=(1, 1), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model537_convTrans_16x8_inp_1x5x5x3_ker4x4_stride1_padvalid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(5, 5, 3)),
        tf.keras.layers.Conv2DTranspose(
            filters=23,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding='valid',
            use_bias=True
        )
    ])



@versioned_unhashable_object_fixture
def model538_convTrans_16x8_inp_1x7x7x2_ker4x3_stride2_padsame():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(7, 7, 2)),
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(4, 3), strides=(2, 2), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model539_convTrans_16x8_inp_1x3x3x1_ker3x3_stride3_padsame():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(3, 3), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model540_convTrans_16x8_inp_1x3x3x1_ker3x3_stride3_padvalid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(3, 3), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model541_convTrans_16x8_inp_1x11x10x7_ker2x4_stride2x4_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(11, 10, 7), dtype=tf.float32),
        tf.keras.layers.Conv2DTranspose(filters=11, kernel_size=(2, 4), strides=(2, 4), padding='valid', use_bias=True)
    ])



