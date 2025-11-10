import pytest
import tensorflow as tf

from torq.testing.tensorflow import TfliteTestCase
from torq.testing.iree import MODELS_DIR, MlirTestCase, WithLLVMCPUReference, WithTweakedRandomDataInput
from torq.testing.compilation_tests import parametrize, WithParameters

from .models.keras_models import conv_act_model

class WithCssQemu:

    @property
    def compiler_options(self):
        return ["--torq-disable-slices", "--iree-input-type=tosa", "--torq-css-qemu"]


@parametrize(["matmul-notile", "softmax"])
class TestTosaMlir(WithParameters, WithCssQemu, WithTweakedRandomDataInput, WithLLVMCPUReference, MlirTestCase):

    pytestmark = pytest.mark.ci

    @property
    def mlir_model_file(self):
        return MODELS_DIR / "tosa_ops" / (self.params + ".mlir")


@parametrize(["tensor_pad"])
class TestLinalgMlir(WithParameters, WithCssQemu, WithTweakedRandomDataInput, WithLLVMCPUReference, MlirTestCase):

    pytestmark = pytest.mark.ci

    @property
    def mlir_model_file(self):
        return  MODELS_DIR / 'linalg_ops' / (self.params + ".mlir")


class TestConvRelu(WithTweakedRandomDataInput, WithCssQemu, TfliteTestCase):

    pytestmark = pytest.mark.ci

    def model(self):
        return conv_act_model()
