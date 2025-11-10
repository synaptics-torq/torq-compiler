from torq.testing.iree import MlirTestCase, WithLLVMCPUReference, list_mlir_files, WithTweakedRandomDataInput
from torq.testing.compilation_tests import parametrize, WithParameters
import pytest


@parametrize(list_mlir_files("tosa_ops"))
class TestLinalgMlir(WithParameters, WithTweakedRandomDataInput, WithLLVMCPUReference, MlirTestCase):

    pytestmark = pytest.mark.ci

    @property
    def compiler_options(self):
        return ["--iree-input-type=tosa-torq", "--torq-css-qemu"]

    @property
    def mlir_model_file(self):
        return self.params
