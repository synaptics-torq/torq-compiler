import tensorflow as tf

from dataclasses import dataclass
from torq.testing.versioned_fixtures import versioned_hashable_object_fixture
from torq.testing.tensorflow import keras_model_fixture


@dataclass
class ConvModelParams:
    width: int
    height: int
    filters: int
    kernel_size: int          # square kernel; per-axis override via kernel_h/kernel_w
    input_channels: int
    # Optional axes for the parametric conv sweep. Defaults keep the existing
    # square-kernel / stride-1 / same-padding / Conv2D instances valid.
    stride: int = 1           # square stride; per-axis override via stride_h/stride_w
    padding: str = 'same'     # 'same' | 'valid'
    rank: int = 2             # 2 = Conv2D, 1 = Conv1D (width/kernel_w/stride_w unused)
    kernel_h: int = None      # None -> use kernel_size
    kernel_w: int = None      # None -> use kernel_size
    stride_h: int = None      # None -> use stride
    stride_w: int = None      # None -> use stride

    @property
    def kh(self):
        return self.kernel_h if self.kernel_h is not None else self.kernel_size

    @property
    def kw(self):
        return self.kernel_w if self.kernel_w is not None else self.kernel_size

    @property
    def sh(self):
        return self.stride_h if self.stride_h is not None else self.stride

    @property
    def sw(self):
        return self.stride_w if self.stride_w is not None else self.stride

    def idfn(val):
        # Base id (back-compat with the existing two instances), plus only the
        # non-default axes so sweep ids stay readable and unique.
        s = f"w{val.width}_h{val.height}_f{val.filters}_k{val.kernel_size}_c{val.input_channels}"
        if val.rank != 2:
            s += f"_r{val.rank}"
        if (val.kh, val.kw) != (val.kernel_size, val.kernel_size):
            s += f"_khw{val.kh}x{val.kw}"
        if (val.sh, val.sw) != (1, 1):
            s += f"_s{val.sh}x{val.sw}"
        if val.padding != 'same':
            s += f"_p{val.padding}"
        return s


conv_model_params = [ConvModelParams(12, 12, 5, 1, 64), ConvModelParams(100, 100, 5, 6, 4)]


@versioned_hashable_object_fixture
def keras_model_params(case_config):
    return case_config['keras_model_params']


def build_conv_model(p):
    # make sure tests use reproducible weights
    tf.keras.utils.set_random_seed(21321)

    if p.rank == 1:
        return tf.keras.Sequential([
            tf.keras.layers.Input(batch_size=1, shape=(p.width, p.input_channels)),
            tf.keras.layers.Conv1D(filters=p.filters, kernel_size=p.kw,
                                   strides=p.sw, padding=p.padding),
        ])
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(p.height, p.width, p.input_channels)),
        tf.keras.layers.Conv2D(filters=p.filters, kernel_size=(p.kh, p.kw),
                               strides=(p.sh, p.sw), padding=p.padding),
    ])


@keras_model_fixture
def conv_model(request, keras_model_params):
    return build_conv_model(keras_model_params)


@keras_model_fixture
def conv_act_model(request):

    # make sure tests use reproducible weights
    tf.keras.utils.set_random_seed(21321)

    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same'),
        tf.keras.layers.ReLU(6.0)
    ])
