"""Write small Keras test models as data: the layers are created when the model is built.

Example::

    build = sequential((10, 10, 4), layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model = build()  # seeds the RNG, then creates the layers and the model

``layers.X(*args, **kwargs)`` records a ``tf.keras.layers.X`` call. The layers are created
after ``tf.keras.utils.set_random_seed(seed)``, so their weights are reproducible.
"""

from functools import partial


class LayerSpec:
    def __init__(self, name, args, kwargs):
        self.name, self.args, self.kwargs = name, args, kwargs

    def build(self):
        import tensorflow as tf

        return getattr(tf.keras.layers, self.name)(*self.args, **self.kwargs)


class _Layers:
    def __getattr__(self, name):
        return lambda *args, **kwargs: LayerSpec(name, args, kwargs)


layers = _Layers()


def _build_sequential(shape, specs, batch_size, seed):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    return tf.keras.Sequential(
        [tf.keras.layers.Input(batch_size=batch_size, shape=shape)] + [s.build() for s in specs]
    )


def sequential(shape, *specs, batch_size=1, seed=42):
    """A builder of a Sequential model on a float32 (batch_size, *shape) input."""
    return partial(_build_sequential, tuple(shape), specs, batch_size, seed)


def _build_functional(shape, build, batch_size, seed):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.layers.Input(batch_size=batch_size, shape=shape)
    return tf.keras.Model(inputs=inputs, outputs=build(inputs))


def functional(shape, build, batch_size=1, seed=42):
    """A builder of a functional model: build(inputs) creates the layers and returns the outputs."""
    return partial(_build_functional, tuple(shape), build, batch_size, seed)


def _strides_or_padding(strides, padding):
    # The old *_parametrize helpers accepted the padding in place of the strides.
    return ((1, 1), strides) if isinstance(strides, str) else (strides, padding)


def conv2d(shape, filters, kernel_size, strides=(1, 1), padding="valid", **kwargs):
    """A builder of a single Conv2D (with bias) model."""
    strides, padding = _strides_or_padding(strides, padding)
    return sequential(shape, layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                           strides=strides, padding=padding, use_bias=True,
                                           **kwargs))


def depthwise(shape, kernel_size, strides=(1, 1), padding="valid", **kwargs):
    """A builder of a single DepthwiseConv2D (depth multiplier 1, with bias) model."""
    strides, padding = _strides_or_padding(strides, padding)
    return sequential(shape, layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                                                    padding=padding, depth_multiplier=1,
                                                    use_bias=True, **kwargs))


def conv2d_transpose(shape, filters, kernel_size, strides=(1, 1), padding="valid", **kwargs):
    """A builder of a single Conv2DTranspose (with bias) model."""
    strides, padding = _strides_or_padding(strides, padding)
    return sequential(shape, layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                                    strides=strides, padding=padding,
                                                    use_bias=True, **kwargs))
