import pytest

from torq.testing.iree import list_mlir_files, list_mlir_file_group
from torq.testing.cases import Case
from torq.testing.comparison import compare_test_results


def get_test_cases():
    test_cases = []

    base_options = ["--torq-enable-tile-and-fuse"]

    for mlir_file in list_mlir_files("linalg_ops"):
        test_cases.append(Case("linalg_" + mlir_file.stem, {
            "static_mlir_model_file": mlir_file,
            "torq_compiler_options": base_options + ["--iree-input-type=linalg-torq"]
        }))

    for mlir_file in list_mlir_files("tosa_ops"):
        test_cases.append(Case("tosa_" + mlir_file.stem, {
            "static_mlir_model_file": mlir_file,
            "torq_compiler_options": base_options + ["--iree-input-type=tosa-torq"]
        }))

    for mlir_file in list_mlir_file_group("torch_ops"):
        
        if mlir_file.name in ["equal.mlir", "instancenorm.mlir"]:
            continue  # not implemented yet

        test_cases.append(Case("torch_" + mlir_file.stem, {
            "static_mlir_model_file": mlir_file,
            "torq_compiler_options": base_options
        }))

    return test_cases


@pytest.fixture(params=get_test_cases())
def case_config(request):    

    # tosaops
    failed_tc = [
        # crash
        'tosa_pw-32x8-7x7x320',
        'tosa_maxpool2d-stride2-k3x3-pad-224x224x64',
        'tosa_maxpool-seg-tile',
        'tosa_maxpool2d-stride2-k3x3-pad-112x112x128',
        'tosa_maxpool2d-stride2-k3x3-pad-112x112x64',
        'tosa_maxpool-seg-tile-1',
        'tosa_pad-dw',
        'tosa_conv-stride1',
        'tosa_dw_f5_s2_128x128x72',

        # result wrong
        'tosa_pw-stride2',

        # hang
        'tosa_dw_i16_s2_8x8x1',
        'tosa_dw_wzp',
        'tosa_softmax',
    
        # wrong result
        'linalg_tanh-bf16',
        'linalg_rsqrt-bf16',

        # crash
        'linalg_quantized_batch_matmul',

        # hang
        'linalg_softmax-bf16',
        'linalg_reducesum-a0-i32',
        'linalg_batch-matmul-in-int8-out-int16',
        'linalg_reduceall-a2',
        'linalg_fill-56x48x24',
        'linalg_Elementwise-less-than-u16',
        'linalg_broadcast-a0',
        'linalg_reducexor-a0',

        # crash
        'torch_encoder.mlir.230.Conv_0_small',
        'torch_equal',
        'torch_encoder.mlir.243.Conv_2_small',
        'torch_encoder.mlir.237.Conv_1_small',

        # wrong result
        'torch_instancenorm',

        # unable to free enough space for results and operands
        'torch_Conv2d_bf16_1x1x64x8192',
        'torch_Conv2d_bf16_1x1x8192x64',
        # failed to run translation of source executable to target executable for backend
        'torch_ConvTranspose_bf16_1x1x512x512',

        # fails even without tile and fuse
        'torch_0135_ReduceMean__layers.0_post_attention_layernorm_ReduceMean'
    ]

    if request.param.name in failed_tc:
        pytest.xfail("known failure")    

    return {
        "mlir_model_file": "static_mlir_model_file",
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir",
        **request.param.data
    }

@pytest.mark.ci
def test_mlir_files(request, torq_results, llvmcpu_reference_results, case_config):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
