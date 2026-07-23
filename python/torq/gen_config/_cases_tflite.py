# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TFLite executor discovery test cases and fixtures.

Mirrors :mod:`torq.gen_config._cases` (the ONNX harness) for TFLite models.
Contains ``pytest_generate_tests``, the TFLite-specific fixtures, and the core
``executor_discovery_tflite`` implementation used by
``tests/test_tflite_gen_config.py``.

Full-model executor assignment relies on matching MLIR ``line:column`` locations
in the compiler pass. TFLite models are lowered to the TOSA dialect by
``tosa-converter-for-tflite``, where one TFLite op typically expands into several
TOSA ops (e.g. CONV_2D -> conv2d + rescale + clamp). Each TFLite layer is mapped
to its primary compute TOSA op location on a best-effort, position-based basis.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from torq.testing.cases import Case
from torq.testing.iree import list_files
from torq.testing.tflite_layer_extractor import TFLiteLayerExtractor
from torq.testing.tflite_layer_tests import generate_tflite_layer_cases
from torq.testing.versioned_fixtures import (
    VersionedUncachedData,
    versioned_cached_data_fixture,
)

from torq.gen_config._state import ExecutorDiscoveryState, _discovery_state
from torq.gen_config._utils import extract_tosa_line_numbers_from_mlir
from torq.gen_config._utils_mac import compute_tflite_model_mac_details
from torq.gen_config.core import (
    EXECUTOR_ORDER,
    _discovery_log,
    _load_json,
    _opt,
)
from torq.gen_config._cases import (
    _extract_model_name_from_case,
    _get_skipped_executors,
    _run_layer_test,
)

# Map a TFLite builtin operator name to the primary TOSA op it lowers to.
# Used to locate the representative compute op for a layer within the TOSA MLIR.
# Ops not listed fall back to the next unconsumed non-constant TOSA op.
TFLITE_TO_TOSA_PRIMARY = {
    "CONV_2D": "conv2d",
    "DEPTHWISE_CONV_2D": "depthwise_conv2d",
    "TRANSPOSE_CONV": "transpose_conv2d",
    "FULLY_CONNECTED": "matmul",
    "BATCH_MATMUL": "matmul",
    "ADD": "add",
    "SUB": "sub",
    "MUL": "mul",
    "DIV": "reciprocal",
    "AVERAGE_POOL_2D": "avg_pool2d",
    "MAX_POOL_2D": "max_pool2d",
    "MEAN": "reduce_sum",
    "SUM": "reduce_sum",
    "REDUCE_MAX": "reduce_max",
    "LOGISTIC": "sigmoid",
    "TANH": "tanh",
    "SOFTMAX": "exp",
    "PAD": "pad",
    "PADV2": "pad",
    "CONCATENATION": "concat",
    "TRANSPOSE": "transpose",
    "RESHAPE": "reshape",
    "RESIZE_BILINEAR": "resize",
    "RESIZE_NEAREST_NEIGHBOR": "resize",
    "RELU": "clamp",
    "RELU6": "clamp",
    "RELU_N1_TO_1": "clamp",
    "HARD_SWISH": "mul",
    "QUANTIZE": "rescale",
    "DEQUANTIZE": "rescale",
    "LOGISTIC_QUANT": "sigmoid",
    "MAXIMUM": "maximum",
    "MINIMUM": "minimum",
    "EXP": "exp",
    "RSQRT": "rsqrt",
    "ABS": "abs",
    "NEG": "negate",
}

# TOSA ops treated as structural glue rather than a layer's primary compute op.
_STRUCTURAL_TOSA_OPS = {"reshape", "transpose", "cast", "const", "const_shape"}


def _tflite_layer_id(op_name: str, op_index: int) -> str:
    """Build the layer_id used as the JSON key for a TFLite operator."""
    return f"{op_name}_{op_index}"


def _compute_tflite_layer_mac_info(case) -> Optional[Dict[str, Any]]:
    """Compute MAC count plus per-op metadata for a TFLite layer case."""
    if case is None or not hasattr(case, "data") or case.data is None:
        return None
    layer_case = case.data
    layer_model_path = getattr(layer_case, "layer_model_path", None)
    if not layer_model_path:
        return None
    try:
        return compute_tflite_model_mac_details(Path(layer_model_path))
    except Exception as e:
        _discovery_log(f"[MacCount] Failed to compute TFLite MAC details: {e}")
        return None


def _build_tflite_to_mlir_mapping(tflite_model_path: Path, mlir_file: Path) -> Dict[str, str]:
    """Build a mapping from TFLite layer_id to full-model TOSA ``line:column``.

    Runs ``tosa-converter-for-tflite`` to obtain the full-model TOSA MLIR (the
    same converter the compiler consumes), then greedily assigns each TFLite op
    to its primary compute TOSA op location.

    Returns: {layer_id -> "line:column"}
    """
    try:
        mlir_file.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tosa-converter-for-tflite", "--text", str(tflite_model_path), "-o", str(mlir_file)],
            check=True,
            capture_output=True,
        )
    except Exception as e:
        _discovery_log(f"[TFLitetoMLIR] Error converting {tflite_model_path} to TOSA: {e}")
        return {}

    tosa_ops = extract_tosa_line_numbers_from_mlir(mlir_file)
    if not tosa_ops:
        _discovery_log(f"[TFLitetoMLIR] No TOSA ops found in {mlir_file}")
        return {}

    try:
        extractor = TFLiteLayerExtractor(str(tflite_model_path))
        layers = extractor.get_layer_info()
    except Exception as e:
        _discovery_log(f"[TFLitetoMLIR] Error reading TFLite layers: {e}")
        return {}

    mapping: Dict[str, str] = {}
    pos = 0
    for layer in layers:
        op_name = layer["op_name"]
        if op_name == "DELEGATE":
            continue
        expected = TFLITE_TO_TOSA_PRIMARY.get(op_name)
        location: Optional[str] = None

        if expected:
            # Scan forward for the primary TOSA op of this layer.
            for j in range(pos, len(tosa_ops)):
                if tosa_ops[j][0] == expected:
                    location = tosa_ops[j][1]
                    pos = j + 1
                    break

        if location is None:
            # Fall back to the next unconsumed non-structural TOSA op.
            while pos < len(tosa_ops) and tosa_ops[pos][0] in _STRUCTURAL_TOSA_OPS:
                pos += 1
            if pos < len(tosa_ops):
                location = tosa_ops[pos][1]
                pos += 1

        if location is not None:
            mapping[_tflite_layer_id(op_name, layer["index"])] = location

    _discovery_log(
        f"[TFLitetoMLIR] Mapping: {len(mapping)} layer(s) mapped to TOSA locations "
        f"({len(tosa_ops)} TOSA ops)"
    )
    return mapping


def _discover_model_files(config) -> List[Path]:
    """Discover TFLite model files from --model/--model-path or default dirs."""
    model_path = _opt(config, "--model", "--model-path")
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise pytest.UsageError(f"--model-path does not exist: {model_path}")
        return [path]
    files = list_files("dev_ops", ".tflite", False) + list_files("tflite_models", ".tflite", False)
    if not files:
        pytest.skip("No TFLite models found")
    return files


def pytest_generate_tests(metafunc):
    """Generate per-layer/per-executor and full-model cases for TFLite models."""
    if "layer_executor_case" not in metafunc.fixturenames:
        return

    files = _discover_model_files(metafunc.config)
    if not files:
        return

    skipped_executors = _get_skipped_executors(metafunc.config)
    if skipped_executors:
        _discovery_log(f"[SkipExecutors] Will skip executors: {sorted(skipped_executors)}")

    # Build all cases and the per-model location maps (layer_id -> line:column).
    all_cases: List[Case] = []
    case_location_maps: Dict[str, Dict[str, str]] = {}
    for model_file in files:
        model_name = model_file.stem
        mlir_cache = Path(f".pytest_cache/tflite_full_mlir/{model_name}.mlir")
        location_map = _build_tflite_to_mlir_mapping(model_file, mlir_cache)
        cases = [
            Case(tflite_case.name, tflite_case)
            for tflite_case in generate_tflite_layer_cases(model_name, model_file, metafunc)
        ]
        all_cases.extend(cases)
        for case in cases:
            case_location_maps[case.name] = location_map

    test_cases = []
    for case in all_cases:
        # Full-model cases carry no layer_id; executors are provided by fixture.
        if not case.data.is_layer:
            test_cases.append((case, None, "discovered", None, None, False, None))
            continue

        layer_id = _tflite_layer_id(case.data.op_name, case.data.op_index)
        node_index = case.data.op_index
        full_mlir_location = case_location_maps.get(case.name, {}).get(layer_id)
        for executor in EXECUTOR_ORDER:
            if executor in skipped_executors:
                continue
            test_cases.append(
                (case, layer_id, executor, node_index, full_mlir_location, False, None)
            )

    metafunc.parametrize(
        "layer_executor_case",
        test_cases,
        indirect=True,
        ids=[f"{c.name}_{ex}" for c, _, ex, _, _, _, _ in test_cases],
    )


@pytest.fixture
def layer_executor_case(request):
    """Provide the TFLite layer case with executor info."""
    case, layer_id, executor, node_index, full_mlir_location, is_subgraph, source_layer_id = request.param
    return {
        "case": case,
        "layer_id": layer_id,
        "executor": executor,
        "node_index": node_index,
        "full_mlir_location": full_mlir_location,
        "is_subgraph": is_subgraph,
        "source_layer_id": source_layer_id,
    }


@pytest.fixture
def tflite_layer_model(request, layer_executor_case):
    """Provide the TFLite layer model, overriding the shared plugin fixture.

    Returns a versioned wrapper around the ``TFLiteLayerCase`` so the existing
    ``tflite_*`` fixture chain (model path, MLIR, layer inputs, reference) works
    unchanged.
    """
    case = layer_executor_case["case"]
    return VersionedUncachedData(data=case.data, version="tflite_layer_" + case.name)


@versioned_cached_data_fixture
def gen_config_full_model_input_data(request, tflite_layer_model):
    """Generate reproducible random inputs matching the full TFLite model's inputs."""
    import numpy as np
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(tflite_layer_model.full_model_path))
    interp.allocate_tensors()

    rng = np.random.default_rng(1234)
    inputs = []
    for detail in interp.get_input_details():
        shape = [int(s) for s in detail["shape"]]
        dtype = detail["dtype"]
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            arr = rng.integers(info.min, info.max + 1, size=shape, dtype=dtype)
        else:
            arr = rng.random(shape).astype(dtype)
        inputs.append(arr)
    return inputs


@pytest.fixture
def reference_results(request, tflite_layer_model):
    """Reference results from the TFLite interpreter."""
    return request.getfixturevalue("tflite_reference_results")


def executor_discovery_tflite(
    request,
    torq_results,
    reference_results,
    case_config,
    layer_executor_case,
    tflite_mlir_model_file,
    discovery_state: ExecutorDiscoveryState = _discovery_state,
):
    """Core TFLite executor discovery implementation."""
    layer_id = layer_executor_case["layer_id"]
    executor = layer_executor_case["executor"]
    node_index = layer_executor_case.get("node_index")
    full_mlir_location = layer_executor_case.get("full_mlir_location")
    case = layer_executor_case["case"]

    model_name = _extract_model_name_from_case(case)

    # Load existing JSON results on first test (for consistent reporting in skip mode)
    if not hasattr(executor_discovery_tflite, "_loaded_existing_json"):
        executor_discovery_tflite._loaded_existing_json = {}

    if model_name and model_name not in executor_discovery_tflite._loaded_existing_json:
        executor_discovery_tflite._loaded_existing_json[model_name] = True
        json_data = _load_json(request.config, model_name)
        if json_data:
            discovery_state.load_from_json(json_data)

    # Full model test: no layer_id, just compare and capture metrics on failure.
    if layer_id is None:
        from torq.testing.comparison import compare_test_results
        import io
        import contextlib

        capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(capture):
                compare_test_results(request, torq_results, reference_results, case_config)
        except AssertionError:
            captured = capture.getvalue()
            metrics = {}
            for line in captured.splitlines():
                if line.startswith("Max relative difference:"):
                    metrics["max_rel_diff"] = line.split(":", 1)[1].strip()
                elif line.startswith("Max absolute difference:"):
                    metrics["max_abs_diff"] = line.split(":", 1)[1].strip()
                elif line.startswith("Number of differences:"):
                    metrics["num_differences"] = line.split(":", 1)[1].strip()
            discovery_state.full_model_metrics = metrics
            raise
        return

    mlir_file_path = (
        Path(tflite_mlir_model_file.file_path)
        if tflite_mlir_model_file and hasattr(tflite_mlir_model_file, "file_path")
        else None
    )

    # Layer mode: record metadata (including MAC count) before running the test.
    mac_info = _compute_tflite_layer_mac_info(case)
    discovery_state.record_metadata(
        layer_id=layer_id,
        node_index=node_index,
        full_mlir_location=full_mlir_location,
        mlir_file=mlir_file_path,
        mac_count=mac_info["mac_count"] if mac_info else None,
        mac_details=mac_info,
    )

    json_data = _load_json(request.config, model_name) if model_name else {}
    _run_layer_test(
        request, torq_results, reference_results, case_config,
        layer_id, executor, node_index, mlir_file_path, json_data, case.name
    )
