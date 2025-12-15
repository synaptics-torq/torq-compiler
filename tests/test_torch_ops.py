import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.iree import list_mlir_file_group
from torq.testing.cases import get_test_cases_from_files


@pytest.fixture(params=get_test_cases_from_files(list_mlir_file_group("torch_ops")))
def case_config(request, runtime_hw_type, chip_config):

    aws_fpga = (runtime_hw_type.data == "aws_fpga")

    failed_tc = [
        'equal.mlir',
        'instancenorm.mlir', 
        '0135_ReduceMean__layers.0_post_attention_layernorm_ReduceMean.mlir'
    ]

    if aws_fpga:
        # aws-fpga failures
        failed_tc += [
            'decoder.mlir.431.Mul_24.mlir', # AssertionError: Number of differences: 53378 out of 59616 [89.54%]
        ]

    if chip_config.data['target'] != "SL2610":
        # Next chip failures
        failed_tc += [
            'conv2d-nchw-clip-bf16.mlir', # Number of differences: 1914 out of 401408 [0.48%]
            'encoder.mlir.230.Conv_0_small.mlir' # Compiler Timeout (exceeded 300 seconds)
        ]
        if aws_fpga:
            # aws-fpga failures
            failed_tc += [
                # native_executable.cc:1085: INTERNAL; torq failed to wait;
                'encoder.mlir.237.Conv_1_small.mlir', # AssertionError: Number of differences: 53378 out of 59616 [89.54%]
                'encoder.mlir.243.Conv_2_small.mlir'
            ]

    if any(s in request.param.name for s in failed_tc):
        pytest.xfail("output mismatch or error")

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
