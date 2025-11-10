import tensorflow as tf

# make sure tests use reproducible weights and inputs
tf.keras.utils.set_random_seed(21321)

from torq.testing.compilation_tests import WithParameters, parametrize
from torq.testing.iree import WithRandomUniformIntegerInputData
from torq.testing.tensorflow import *

from .models.keras_models import ConvModelParams, conv_model, transpose_conv_model

import pytest


from dataclasses import dataclass


@dataclass
class TransposeConvTestParams:
    quantize_to_int16: bool

    @staticmethod
    def idfn(val):
        return val.quantize_to_int16 and "i16" or "i8"


@parametrize([TransposeConvTestParams(True), TransposeConvTestParams(False)])
class TestTransposeConv(WithParameters, WithRandomUniformIntegerInputData, TfliteTestCase):

    pytestmark = pytest.mark.ci

    # overrides default quantization to int8
    def quantize_to_int16_value(self):
        return self.params.quantize_to_int16
    
    def model(self):
        return transpose_conv_model()


@parametrize([ConvModelParams(12, 12, 5, 1, 64), ConvModelParams(100, 100, 5, 6, 4)])
class TestConv(WithRandomUniformIntegerInputData, WithParameters, TfliteTestCase):

    pytestmark = pytest.mark.ci

    def model(self):
        return conv_model(self.params)

