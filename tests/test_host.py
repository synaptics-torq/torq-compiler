import pytest
import numpy as np

from torq.testing.tensorflow import TfliteTestCase
from torq.testing.iree import MODELS_DIR, MlirTestCase, WithLLVMCPUReference, WithTweakedRandomDataInput
from torq.testing.compilation_tests import parametrize, WithParameters

from .models.keras_models import conv_act_model


class WithHost:

    @property
    def compiler_options(self):
        return ["--torq-disable-slices", "--torq-disable-css", "--iree-input-type=tosa", "--torq-css-qemu"]


@parametrize(["matmul-notile", "softmax"])
class TestTosaMlir(WithParameters, WithHost, WithTweakedRandomDataInput, WithLLVMCPUReference, MlirTestCase):

    pytestmark = pytest.mark.ci

    @property
    def mlir_model_file(self):
        return MODELS_DIR / "tosa_ops" / (self.params + ".mlir")


@parametrize(["trunci", "extui"])
class TestArithMlir(WithParameters, WithHost, WithTweakedRandomDataInput, WithLLVMCPUReference, MlirTestCase):

    pytestmark = pytest.mark.ci

    @property
    def mlir_model_file(self):
        return MODELS_DIR / "arith_ops" / (self.params + ".mlir")



class TestTorchEqual(WithHost, WithLLVMCPUReference, MlirTestCase):

    def generate_input_data(self):

        input_tensor = np.zeros((1, 1, 30, 1), dtype=np.int64)
        
        input_tensor[0, 0, 10, 0] = 10
        input_tensor[0, 0, 20, 0] = 20

        return [input_tensor]

    @property
    def mlir_model_file(self):
        return MODELS_DIR / "torch_ops" / "equal.mlir"
    

class TestTorchInstanceNorm(WithHost, WithTweakedRandomDataInput, WithLLVMCPUReference, MlirTestCase):

    @property
    def mlir_model_file(self):
        return MODELS_DIR / "torch_ops" / "instancenorm.mlir"


class TestConvRelu(WithTweakedRandomDataInput, WithHost, TfliteTestCase):

    pytestmark = pytest.mark.ci

    def model(self):
        return conv_act_model()
