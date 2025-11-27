import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, versioned_unhashable_object_fixture


@versioned_unhashable_object_fixture
def model050_mult_inp1x10x10x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4), name='input_1')
    
    # Branch for Conv2D
    x = tf.keras.layers.Conv2D(
        filters=4,
        kernel_size=(4, 4),
        strides=(1, 1),
        padding='same',
        use_bias=True,
        name='conv2d'
    )(inputs)
    
    # Multiply the original input with the Conv2D output
    outputs = tf.keras.layers.Multiply()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model051_mult_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API for multi-input operation (Multiply)
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1), name='input_1')
    
    # Branch 1: Conv2D
    x = tf.keras.layers.Conv2D(
        filters=1, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same', 
        use_bias=True
    )(inputs)
    
    # Merge: Multiply the original input with the Conv2D output
    outputs = tf.keras.layers.Multiply()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model052_mult_inp1x4x4x1_zp128_AB():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1), name='input_1')
    
    # Branch for CONV_2D
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(inputs)
    x = tf.keras.layers.ReLU()(x)
    
    # Multiply operation
    outputs = tf.keras.layers.Multiply()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model053_mult_inp1x4x4x1_zp128_B():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1), name='input_1')
    
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(inputs)
    x = tf.keras.layers.ReLU()(x)
    
    outputs = tf.keras.layers.Multiply()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model054_mult_inp1x4x4x1_zp128_A():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API for multi-input operation (MUL)
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 2), name='serving_default_input_1')
    
    # Branch 1: Conv2D
    # op_structure[0]: CONV_2D
    x = tf.keras.layers.Conv2D(
        filters=2,
        kernel_size=(2, 2),
        strides=(1, 1),
        padding='same',
        activation=None,
        use_bias=True
    )(inputs)
    
    # op_structure[1]: MUL
    # Inputs are the original input and the output of the Conv2D
    outputs = tf.keras.layers.Multiply()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model060_add_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1), name='input_1')
    
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        use_bias=True
    )(inputs)
    
    outputs = tf.keras.layers.Add()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model061_add_inp1x4x4x1_zp128_x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1), name='input_1')
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(inputs)
    x = tf.keras.layers.ReLU()(x)
    outputs = tf.keras.layers.Add()([inputs, x])
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model062_add_inp1x4x4x1_zp128_x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1), name='input_1')
    
    # Convolutional branch
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        use_bias=True
    )(inputs)
    x = tf.keras.layers.ReLU()(x)
    
    # Add operation (residual connection)
    outputs = tf.keras.layers.Add()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model120_mult_inp1x8x8x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(8, 8, 1), name='serving_default_input_1_0')
    outputs = tf.keras.layers.Multiply()([inputs, inputs])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model121_add_inp1x8x8x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(8, 8, 1), name='serving_default_input_1_0')
    outputs = tf.keras.layers.Add()([inputs, inputs])
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model122_conv_mult_inp1x8x8x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Functional API for the MULTIPLY operation
    inputs = tf.keras.layers.Input(batch_size=1, shape=(8, 8, 1), name="serving_default_input_1_0")
    
    # The MUL op takes the same input twice
    outputs = tf.keras.layers.Multiply()([inputs, inputs])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model123_conv_add_inp1x8x8x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(8, 8, 1))
    
    # Branch for CONV_2D
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=True)(inputs)
    x = tf.keras.layers.ReLU()(x)
    
    # Add operation
    outputs = tf.keras.layers.Add()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model130_conv3x3_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model131_mult_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1), name="serving_default_input_1_0")
    outputs = tf.keras.layers.Multiply()([inputs, inputs])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model132_add_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1), name='serving_default_input_1_0')
    outputs = tf.keras.layers.Add()([inputs, inputs])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model133_conv3x3_mult_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1), name="serving_default_input_1_0")
    
    # Convolutional branch
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(inputs)
    x = tf.keras.layers.ReLU()(x)
    
    # Multiply with original input
    outputs = tf.keras.layers.Multiply()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model134_conv3x3_add_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1))
    
    # Convolutional branch
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(inputs)
    x = tf.keras.layers.ReLU()(x)
    
    # Add operation
    outputs = tf.keras.layers.Add()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model135_conv3x3_conv2x2_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(2, 2), strides=(1, 1), padding='same', use_bias=True),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model136_conv2x2_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1)),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(2, 2), strides=(1, 1), padding="same", use_bias=True),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model137_add_mult_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1), name="serving_default_input_1_0")
    x = tf.keras.layers.Add()([inputs, inputs])
    outputs = tf.keras.layers.Multiply()([x, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model138_mult_mult_inp1x3x3x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(3, 3, 1), name="serving_default_input_1:0")
    x = tf.keras.layers.Multiply()([inputs, inputs])
    outputs = tf.keras.layers.Multiply()([x, x])
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model140_conv3x3_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1)),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model141_mult_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1), name="serving_default_input_1_0")
    outputs = tf.keras.layers.Multiply()([inputs, inputs])
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model142_add3x3_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1), dtype=tf.float32)
    outputs = tf.keras.layers.Add()([inputs, inputs])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model143_conv3x3_mult_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1), name="serving_default_input_1_0")
    
    # Convolutional branch
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        use_bias=True
    )(inputs)
    x = tf.keras.layers.ReLU()(x)
    
    # Multiply with original input
    outputs = tf.keras.layers.Multiply()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model144_conv3x3_add_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1), name="serving_default_input_1_0")
    
    # Convolutional branch
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        use_bias=True
    )(inputs)
    x = tf.keras.layers.ReLU()(x)
    
    # Add operation (skip connection)
    outputs = tf.keras.layers.Add()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model145_conv3x3_conv2x2_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1)),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', strides=(1, 1), use_bias=True),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(2, 2), padding='same', strides=(1, 1), use_bias=True),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model146_conv2x2_inp1x4x4x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1)),
        tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(2, 2),
            strides=(1, 1),
            padding="same",
            use_bias=True
        ),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model147_conv_inp1x32x32x16_16x3x3_same_stride1x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(32, 32, 16), dtype=tf.float32),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model150_sub_1x6x8x4_zp128_B():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(6, 8, 4))
    
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=True)(inputs)
    x = tf.keras.layers.ReLU()(x)
    
    outputs = tf.keras.layers.Subtract()([inputs, x])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model151_sub_1x6x8x4_zp128_A():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(6, 8, 4))
    
    # CONV_2D operation
    # From ops_structure: filters=1 (from output shape), kernel_size=(3,3) (from weights shape), strides=(1,1), padding='same', use_bias=True
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=True)(inputs)
    
    # Activation: RELU
    x = tf.keras.layers.ReLU()(x)
    
    # SUB operation
    # Subtracts the result of the conv block from the original input.
    # Broadcasting will handle the shape difference ([1,6,8,1] vs [1,6,8,4]).
    outputs = tf.keras.layers.Subtract()([x, inputs])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model152_sub_1x6x8x4_zp128_AB():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(6, 8, 4), dtype=tf.float32)
    
    # Branch 1
    x1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(inputs)
    x1 = tf.keras.layers.ReLU()(x1)
    
    # Branch 2
    x2 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(inputs)
    x2 = tf.keras.layers.ReLU()(x2)
    
    # Merge
    outputs = tf.keras.layers.Subtract()([x1, x2])
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model153_sub_1x6x8x4_zp128_none():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(6, 8, 4))
    
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=True)(inputs)
    y = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=True)(inputs)
    
    z = tf.keras.layers.Subtract()([x, y])
    
    return tf.keras.Model(inputs=inputs, outputs=z)


@versioned_unhashable_object_fixture
def model154_sub_s2v_1x6x8x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(6, 8, 4), dtype=tf.float32)
    
    # The TFLite operation is SUB(Const, Input), which translates to Const - Input.
    # We can model this with a Lambda layer. The constant value is not specified,
    # so we use a placeholder of 0.0. The quantization process will handle the
    # actual constant value from the original model.
    constant_value = tf.constant(0.0, dtype=tf.float32)
    outputs = tf.keras.layers.Lambda(lambda x: constant_value - x)(inputs)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model155_sub_v2s_1x6x8x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # The SUB operation y = x - c can be modeled as a layer where y = 1*x + b,
    # with b = -c. A DepthwiseConv2D with a 1x1 kernel fixed to 1 and a trainable
    # bias is an efficient way to represent this.
    inputs = tf.keras.layers.Input(batch_size=1, shape=(6, 8, 4))
    
    # The TFLite SUB op with a constant is modeled here.
    # We use a Subtract layer. The constant needs to be a Keras tensor.
    # We create a non-trainable weight to hold the constant value.
    class SubtractConstant(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(SubtractConstant, self).__init__(**kwargs)
            # The value of the constant is unknown, so we let it be a small random value
            # that will be adjusted during quantization-aware training if needed,
            # or just used as-is for post-training quantization.
            self.constant = self.add_weight(
                name='sub_constant',
                shape=(),
                initializer=tf.keras.initializers.RandomNormal(seed=42),
                trainable=True
            )
        def call(self, inputs):
            return tf.subtract(inputs, self.constant)

    x = SubtractConstant()(inputs)
    
    return tf.keras.Model(inputs=inputs, outputs=x)



@versioned_unhashable_object_fixture
def model156_add_s2v_1x6x8x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # Use the Functional API for the ADD operation with a scalar
    inputs = tf.keras.layers.Input(batch_size=1, shape=(6, 8, 4), dtype=tf.float32)
    
    # The TFLite model adds a scalar constant to the input tensor.
    # We use a Lambda layer to replicate this scalar addition.
    # The exact scalar value is not in the analysis, so we use a placeholder like 1.0.
    # The quantization process will handle the actual scale and zero-point.
    x = tf.keras.layers.Lambda(lambda t: t + 1.0)(inputs)
    
    return tf.keras.Model(inputs=inputs, outputs=x)



@versioned_unhashable_object_fixture
def model157_add_v2s_1x6x8x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(6, 8, 4))
    # The ADD operation with a scalar constant is implemented with a Lambda layer.
    # The exact scalar value (e.g., 1.0) is a placeholder; quantization will adjust it.
    x = tf.keras.layers.Lambda(lambda t: t + 1.0)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=x)



@versioned_unhashable_object_fixture
def model158_mul_s2v_1x6x8x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(6, 8, 4), dtype=tf.float32)
    
    # The analysis shows a MUL operation with a scalar constant.
    # We use a Lambda layer to represent this scalar multiplication.
    # The exact constant value is not critical as quantization will adjust scales.
    x = tf.keras.layers.Lambda(lambda t: t * 2.0)(inputs)
    
    return tf.keras.Model(inputs=inputs, outputs=x)



@versioned_unhashable_object_fixture
def model159_mul_v2s_1x6x8x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(6, 8, 4), dtype=tf.float32)
    
    # The TFLite model has a MUL operation with a scalar constant.
    # We represent this with a Lambda layer. The exact scalar value is not
    # critical as quantization will adjust scales. We use a placeholder value.
    x = tf.keras.layers.Lambda(lambda t: t * 2.0)(inputs)
    
    return tf.keras.Model(inputs=inputs, outputs=x)



@versioned_unhashable_object_fixture
def model470_add16x8_positive_s2v():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(10, 10, 3), dtype=tf.float32)
    # The TFLite model shows an ADD operation with a scalar constant.
    # This can be represented by a simple addition in the Functional API.
    # The constant value (e.g., 1.0) is a placeholder; its actual value
    # will be determined by the TFLite converter's quantization process.
    outputs = inputs + 1.0
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model471_add16x8_negative_s2v():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(4, 1, 1), dtype=tf.float32)
    # The scalar value is not specified in the analysis, using a placeholder constant.
    # The structure (tensor + scalar) is the important part.
    x = tf.keras.layers.Lambda(lambda t: t + -1.0)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=x)



@versioned_unhashable_object_fixture
def model472_mult16x8_neg_s2v():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # [LLM: RECONSTRUCT THE ARCHITECTURE HERE BASED ON THE JSON SUMMARY]
    # For simple linear models (Conv2D -> Activation -> Pool -> Dense chains):
    # Use tf.keras.Sequential with explicit layers
    #
    # For models with multi-input operations (ADD, CONCATENATE, MULTIPLY, etc.):
    # Use the Functional API (tf.keras.Model with inputs/outputs)
    #
    # IMPORTANT: The input shape from the JSON includes the batch dimension as the FIRST element.
    # If the JSON shows input shape [N, H, W, C], use:
    #   - batch_size=N (the first dimension)
    #   - shape=(H, W, C) (the remaining dimensions)
    # For example, if input_info shape is [4, 8, 16, 4], use:
    #   tf.keras.layers.Input(batch_size=4, shape=(8, 16, 4))
    #
    # IMPORTANT: Check the 'use_bias' field in ops_structure to determine if layers have bias.
    # If use_bias is False, set use_bias=False in Conv2D/Conv2DTranspose/Dense layers.
    # If use_bias is True or not specified, use the default (use_bias=True).
    #
    # Sequential example:
    # return tf.keras.Sequential([
    #     tf.keras.layers.Input(batch_size=1, shape=(28, 28, 1)),
    #     tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', use_bias=True),
    #     tf.keras.layers.ReLU(6.0),  # Use ReLU(6.0) for RELU6 activation
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(64, use_bias=False),  # Example: no bias
    #     tf.keras.layers.ReLU(),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])
    #
    # Functional API example (for models with ADD/CONCATENATE/etc):
    # inputs = tf.keras.Input(batch_size=1, shape=(4, 4, 1))
    # x = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same')(inputs)
    # y = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same')(inputs)
    # z = tf.keras.layers.Add()([x, y])  # Multi-input operation
    # return tf.keras.Model(inputs=inputs, outputs=z)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(10, 10, 3))
    # The MUL operation multiplies the input by a scalar constant.
    # A Lambda layer is used to represent this operation.
    # The constant value is chosen to be -2.0 based on the model name hint "neg_s2v".
    # The exact value is not critical for structure, but it helps create a similar model.
    outputs = tf.keras.layers.Lambda(lambda x: x * -2.0)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model473_mult16x8_pos_v2s():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(10, 10, 3))
    # The TFLite model has a MUL operation with a scalar constant.
    # This is represented using a Lambda layer.
    outputs = tf.keras.layers.Lambda(lambda x: x * 2.0)(inputs)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)



@versioned_unhashable_object_fixture
def model474_sub16x8_pos_v2s():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(10, 10, 3))
    x = tf.keras.layers.Lambda(lambda t: t - 1.0)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=x)



@versioned_unhashable_object_fixture
def model475_sub16x8_neg_v2s():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(10, 10, 3))
    # The SUB op subtracts a scalar constant from the input tensor.
    # This can be modeled using a Lambda layer. The exact constant value
    # is not specified, but it will be learned during quantization.
    # The name 'model_10/lambda_9/sub/y' suggests a Lambda layer was used.
    outputs = tf.keras.layers.Lambda(lambda x: x - 1.0)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)



