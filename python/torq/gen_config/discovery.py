# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ONNX Executor Discovery Test

Discovers the appropriate executor (NSS/CSS/Host) for each layer of an ONNX model.
Tests each layer on NSS first, then CSS, then Host, recording which one works.

Executor Discovery and Assignment Flow:

    # Step 1: Run layer discovery tests to find optimal executor for each layer
    # This generates torq_gen_config_<model>.json with discovery results
    pytest tests/test_onnx_gen_config.py -v -k "encoder_small_layer_" \
        --model-path=./tests/testdata/onnx_models/encoder_small.onnx --recompute-cache

    # Step 2: Run full model test with discovered executor assignments
    # The torq_torq_gen_config_json fixture provides the executor assignments
    # C++ ExecutorAssignmentPass uses line:column matching to assign executors
    pytest tests/test_onnx_gen_config.py \
        --model-path=./tests/testdata/onnx_models/encoder_small.onnx \
        -v -k "full_model" --debug-ir=tmp --recompute-cache



Design Details:

    Step 1: Layer Discovery Tests
    For each layer (e.g., Tanh, Reshape), test NSS -> CSS -> Host in priority order.
    Save results to torq_gen_config_<model>.json with three status types:
    - "success": Test passed
    - "difference": Accuracy failure (numerical difference)

    Users can re-run specific tests with updated tolerance values:
    pytest tests/test_onnx_gen_config.py -v -k "encoder_small_layer_Tanh_0_nss" \
        --model-path=./tests/testdata/onnx_models/encoder_small.onnx -v --recompute-cache

    Step 2: Full Model Test
    1. Generate full model MLIR from ONNX
    2. Extract CORRECT line numbers from full model MLIR:
       - Tanh at line 10 -> "10:10"
       - Reshape at line 11 -> "11:10"
    3. Update JSON with correct line numbers (replaces temporary "Tanh" with "10:10")
    4. torq_torq_gen_config_json fixture provides executor assignments JSON
       (discovery format with 'ops' key, or compiler format with 'op_assignments' key)
    5. C++ ExecutorAssignmentPass:
       - Loads executor assignments from JSON
       - For each operation, extracts line:column from CallSiteLoc
       - Matches line:column with assignment keys
       - Sets torq-executor attribute

    Note: Line numbers from layer MLIRs (4:10, 6:10) are different from full model
    MLIR (10:10, 11:10). The fixture extracts correct line numbers directly from
    full model MLIR to ensure proper matching.

JSON Format (torq_gen_config_<model>.json):
{
    "version": "1.1",
    "model_name": "encoder",
    "default_tolerance": {"fp_avg_tol": 0.01, "fp_max_tol": 0.01},
    "ops": {
        "Tanh_/Tanh_output_0": {
            "executors": {
                "nss": {
                    "status": "success",
                    "timing": {"runtime_ms": 15.2, "total_ms": 150.5}
                },
                "css": {"status": "difference", ...},
                "host": {"status": "success", ...}
            },
            "expected_executor": "nss",
            "mlir_location": "10:10"
        }
    }
}

Compiler JSON Formats (C++ ExecutorAssignmentPass accepts both):
- Discovery format: {"ops": {"Conv_0": {"recommended_executor": "nss", "mlir_location": "10:10"}}}
- Compiler format: {"op_assignments": {"10:10": {"executor": "nss"}}}
"""
import os
from pathlib import Path


# Default Hugging Face repos for batch model discovery.
# Both test_onnx_discover_gen_config_models.py and test_onnx_run_gen_config_models.py
# import this list so there is a single source of truth.
DEFAULT_HF_REPOS = [
    "onnxmodelzoo/alexnet_Opset17",
    "onnxmodelzoo/adv_inception_v3_Opset16",
    "onnxmodelzoo/cs3darknet_x_Opset18",
    "onnxmodelzoo/cs3edgenet_x_Opset18",
    "onnxmodelzoo/cspresnet50_Opset18",
    "onnxmodelzoo/cspresnext50_Opset18",
    "onnxmodelzoo/darknet53_Opset18",
    "onnxmodelzoo/densenet121_Opset18_timm",
    "onnxmodelzoo/dla102_Opset18",
    "onnxmodelzoo/dpn98_Opset18",
    "onnxmodelzoo/ecaresnet50t_Opset17",
    "onnxmodelzoo/ecaresnetlight_Opset17",
    "onnxmodelzoo/efficientnet_b1_Opset17_timm",
    "onnxmodelzoo/efficientnetv2_rw_t_Opset17",
    "onnxmodelzoo/ens_adv_inception_resnet_v2_Opset18",
    "onnxmodelzoo/fbnetc_100_Opset18",
    "onnxmodelzoo/ghostnet_100_Opset17",
    "onnxmodelzoo/gluon_resnet50_v1s_Opset18",
    "onnxmodelzoo/gluon_resnext50_32x4d_Opset18",
    "onnxmodelzoo/ig_resnext101_32x16d_Opset18",
    "onnxmodelzoo/inception_v3_Opset17_timm",
    "onnxmodelzoo/inception_resnet_v2_Opset18",
    "onnxmodelzoo/mixnet_xl_Opset17",
    "onnxmodelzoo/mnasnet_small_Opset17",
    "onnxmodelzoo/regnetx_320_Opset18",
    "onnxmodelzoo/repvgg_b3g4_Opset18",
    "onnxmodelzoo/res2next50_Opset18",
    "onnxmodelzoo/resnet50_Opset18_timm",
]

# Default directory for ONNX model JSON configs (shared by discovery and run scripts)
_DEFAULT_ONNX_JSON_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "torq-model-configs" / "onnx"

# Re-export state
from torq.gen_config._state import ExecutorDiscoveryState, _discovery_state

# Re-export report functions
from torq.gen_config._report import (
    _generate_final_report_text,
    _generate_report_sections,
    _get_all_critical_failures,
    _print_final_report,
    _save_detailed_report,
)

# Re-export case generation, fixtures, and core discovery logic
from torq.gen_config._cases import (
    _assemble_layer_test_cases,
    _build_duplicate_layer_map,
    _build_onnx_to_mlir_mapping,
    _copy_result_from_source_layer,
    _discover_model_files,
    _extract_failure_report,
    _extract_max_diff,
    _extract_model_name_from_case,
    _extract_op_type_from_layer,
    _generate_layer_cases,
    _generate_subgraph_cases,
    _get_layer_id_from_case,
    _get_skipped_executors,
    _get_subgraph_suffix,
    _is_subgraph_case,
    _maybe_apply_bf16_conversion,
    _maybe_skip_executor,
    _precompute_mlir_mappings,
    _resolve_op_name_to_index,
    _run_layer_test,
    _save_discovery_results,
    _save_json,
    _update_json_with_results,
    _verify_import_ordering,
    comparison_config_for_executor_discovery,
    executor_discovery,
    layer_executor_case,
    onnx_layer_model,
    pytest_generate_tests,
    reference_results,
    save_progress,
    torq_compiler_options,
)
