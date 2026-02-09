import pytest

from torq.testing.iree import list_mlir_files, list_mlir_file_group
from torq.testing.cases import Case
from torq.testing.comparison import compare_test_results


def get_test_cases():
    test_cases = []

    base_options = ["--torq-enable-tile-and-fuse"]
    
    # FIXME: This should be the default for tile-and-fuse but it doesn't work for some cases:
    #base_options += ["--torq-unroll-loop-after-bufferization"]

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
        test_cases.append(Case("torch_" + mlir_file.stem, {
            "static_mlir_model_file": mlir_file,
            "torq_compiler_options": base_options
        }))

    return test_cases


@pytest.fixture(params=get_test_cases())
def case_config(request, chip_config):

    # tosaops
    failed_tc = [
        # fails even without tile and fuse
        'linalg_rsqrt-bf16', # AssertionError: Nans differ.
        'torch_equal',
        'torch_instancenorm',
        'torch_reducemean',
        'torch_reducemean-reshape',
        'torch_conv2d-nchw-clip-bf16',

        ### tests from extras ###

        # Compiler hang
        'tosa_pw-32x8-7x7x320',

        # failed to find a tile size for op (compiler timeout)
        'tosa_conv-343',

        # crash
        'torch_encoder.mlir.230.Conv_0_small',
        'torch_encoder.mlir.243.Conv_2_small',
        'torch_encoder.mlir.237.Conv_1_small',

        # unable to free enough space for results and operands
        'torch_Conv2d_bf16_1x1x64x8192',
        'torch_Conv2d_bf16_1x1x8192x64',
        'tosa_conv2d-f4',
        # failed to run translation of source executable to target executable for backend
        'torch_ConvTranspose_bf16_1x1x512x512',

        # fails even without tile and fuse
        'torch_0135_ReduceMean__layers.0_post_attention_layernorm_ReduceMean'
    ]

    if chip_config.data['target'] != "SL2610":
        failed_tc += [
            # error: unable to free enough space for results and operands
            'tosa_add-rescaled-constant',
            'tosa_asr-i32',
            'tosa_resize-31x31x33xi8',
            'tosa_matmul-in-bf16-out-fp32_207x207',

            # error: 'linalg.transpose' op dim(result, 1) = 16 doesn't match dim(input, permutation[1]) = 33
            'tosa_conv2d_f5_s2_64x64x16_i16',
            # Max absolute difference: 255.0
            # Number of differences: 102477 out of 401408 [25.53%]
            'tosa_conv-stride2',
            # Pass but compiler too long
            # Fails with unroll-loop-after-bufferization:
            #    Assertion failed: (inputElementSize * weightElementSize <= sizeof(int32_t)), function iWidth, file Kernel.cpp, line 2251.
            'tosa_conv2d-stride4-i16',
            # Can pass with unroll-loop-after-bufferization but runtime is very long (4min)
            'tosa_pw-32x8',
            # Compiler too long, can pass with unroll-loop-after-bufferization
            'tosa_matmul-in-int8-out-int16-64x128x2048',
        ]

    if any(s in request.param.name for s in failed_tc):
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
