# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TFLite Executor Discovery Test.

Discovers the appropriate executor (NSS/CSS/Host) for each layer of a TFLite
model, mirroring the ONNX flow in :mod:`torq.gen_config.discovery`.

Executor Discovery and Assignment Flow:

    # Step 1: Run layer discovery to find the optimal executor per layer.
    #   Produces torq_gen_config_<model>.json with discovery results and the
    #   full-model TOSA line:column location for each layer.
    pytest tests/test_tflite_gen_config.py -v -k "_layer_" \
        --model-path=./model.tflite --recompute-cache

    # Step 2: Run the full model with the discovered executor assignments.
    #   The torq_torq_gen_config_json fixture hands the JSON to the compiler and
    #   the C++ ExecutorAssignmentPass matches ops by line:column.
    pytest tests/test_tflite_gen_config.py -v -k "_full_model" \
        --model-path=./model.tflite --recompute-cache

TFLite models are lowered to the TOSA dialect by ``tosa-converter-for-tflite``.
Because one TFLite op expands into several TOSA ops, each layer is mapped to its
primary compute TOSA op location on a best-effort, position-based basis.
"""

# Re-export state
from torq.gen_config._state import ExecutorDiscoveryState, _discovery_state

# Re-export frontend-agnostic fixtures and helpers shared with the ONNX harness
from torq.gen_config._cases import (
    _extract_model_name_from_case,
    _get_skipped_executors,
    _get_subgraph_suffix,
    _maybe_skip_executor,
    _run_layer_test,
    _save_discovery_results,
    comparison_config_for_executor_discovery,
    save_progress,
    torq_compiler_options,
)

# Re-export TFLite-specific case generation, fixtures, and core discovery logic
from torq.gen_config._cases_tflite import (
    TFLITE_TO_TOSA_PRIMARY,
    _build_tflite_to_mlir_mapping,
    _discover_model_files,
    _tflite_layer_id,
    executor_discovery_tflite,
    gen_config_full_model_input_data,
    layer_executor_case,
    pytest_generate_tests,
    reference_results,
    tflite_layer_model,
)
