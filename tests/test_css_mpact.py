import pytest

from .test_css_qemu import get_test_cases, case_config

from torq.testing.comparison import compare_test_results

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture
from .models.keras_models import conv_act_model


@versioned_hashable_object_fixture
def enable_mpact_simulation():
    return True

@pytest.mark.ci
def test_mlir_files(request, torq_results, llvmcpu_reference_results, case_config):    
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
