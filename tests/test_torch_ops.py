from torq.testing.iree import MlirTestCase, WithLLVMCPUReference, list_mlir_files, WithTweakedRandomDataInput
from torq.testing.compilation_tests import parametrize, WithParameters
import pytest


@parametrize(list_mlir_files("torch_ops"))
class TestTorchMlir(WithParameters, WithTweakedRandomDataInput, WithLLVMCPUReference, MlirTestCase):

    pytestmark = pytest.mark.ci

    @property
    def compiler_options(self):
        return ["--iree-input-type=linalg-torq", "--torq-css-qemu"]

    @property
    def mlir_model_file(self):
        if self.params.split("/")[-1] in ["equal.mlir", "instancenorm.mlir"]:
            pytest.xfail("not implemented yet")

        return self.params
