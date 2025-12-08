import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.iree import list_mlir_file_group
from torq.testing.cases import get_test_cases_from_files


@pytest.fixture(params=get_test_cases_from_files(list_mlir_file_group("torch_ops")))
def case_config(request, runtime_hw_type, chip_config):

    cmodel_failed_tc = [
        'equal.mlir',
        'instancenorm.mlir', 
        '0135_ReduceMean__layers.0_post_attention_layernorm_ReduceMean.mlir'
    ]
    if request.param.data.name in cmodel_failed_tc:
        pytest.xfail("not implemented yet")

    # Next chip failures
    next_chip_failed_tc = [
        'conv2d-nchw-clip-bf16.mlir', # Number of differences: 1914 out of 401408 [0.48%]
        'encoder.mlir.230.Conv_0_small.mlir' # Compiler Timeout (exceeded 300 seconds)
    ]
    next_chip = (chip_config.data['target'] != "SL2610")
    if next_chip and any(s in request.param.data.name for s in next_chip_failed_tc):
        pytest.xfail("output mismatch or error on next chip")

    aws_fpga = (runtime_hw_type.data == "aws_fpga")

    # aws-fpga failures
    aws_fpga_failed_tc = [
        'decoder.mlir.431.Mul_24.mlir' # AssertionError: Number of differences: 53378 out of 59616 [89.54%]
    ]
    if aws_fpga and any(s in request.param.name for s in aws_fpga_failed_tc):
        pytest.xfail("output mismatch")

    return {
        "mlir_model_file": "static_mlir_model_file",
        "static_mlir_model_file": request.param.data,
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir"
    }

@pytest.mark.ci
@pytest.mark.fpga_ci
def test_mlir_files(request, torq_results, llvmcpu_reference_results, case_config):    
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
