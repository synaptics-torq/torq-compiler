# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ONNX executor discovery test cases and fixtures.

Contains ``pytest_generate_tests``, all pytest fixtures, and the core
``executor_discovery`` implementation used by ``tests/test_onnx_gen_config.py``.
"""

import contextlib
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import onnx
import pytest

from torq.testing.cases import Case
from torq.testing.iree import list_files
from torq.testing.onnx import (
    extract_onnx_subgraph,
    generate_onnx_layers_from_file,
    generate_onnx_layers_from_model,
    get_full_model,
    model_signature,
)
from torq.testing.versioned_fixtures import (
    VersionedUncachedData,
    versioned_hashable_object_fixture,
)

from torq.gen_config._state import ExecutorDiscoveryState, _discovery_state
from torq.gen_config._utils import (
    _normalize_quantized_op_type,
    extract_line_numbers_from_mlir,
    parse_diff_metrics,
)
from torq.gen_config.core import (
    DEFAULT_TOLERANCE,
    EXECUTOR_ORDER,
    TIMING_PRECISION,
    _discovery_log,
    _get_json_path,
    _load_json,
    _opt,
    build_timing_data,
    extract_model_name_from_case_name,
    get_compiler_config_path,
    get_config_path,
    get_recommended_executor,
    get_subgraph_suffix_from_case_name,
    get_tolerance,
    is_line_column_format,
    load_config,
    save_compiler_config,
    save_config,
    update_config_with_results,
)
from torq.testing.quantize_onnx import (
    is_model_quantized,
    quantize_onnx_model,
)


_TORCH_SUFFIXES = (".pt", ".pth", ".py")


def _is_torch_model(model_path: Path) -> bool:
    """Return True if the model path points to a Torch model file."""
    return model_path.suffix.lower() in _TORCH_SUFFIXES


def _verify_import_ordering(
    onnx_nodes: list, mlir_ops: List[Tuple[str, str]]
) -> Tuple[bool, List[str]]:
    """Verify that MLIR ops match ONNX nodes in order.

    Returns: (is_valid, list_of_warnings)
    """
    warnings = []

    # Filter Constant nodes from ONNX
    onnx_ops = [(n.op_type, n.name) for n in onnx_nodes if n.op_type != "Constant"]

    # Check counts match
    if len(onnx_ops) != len(mlir_ops):
        warnings.append(
            f"COUNT MISMATCH: ONNX has {len(onnx_ops)} non-Constant ops, "
            f"MLIR has {len(mlir_ops)} torch.operator ops"
        )
        return False, warnings

    # Check op types match at each position
    type_mismatches = []
    for i, ((onnx_type, onnx_name), (mlir_type, _)) in enumerate(
        zip(onnx_ops, mlir_ops)
    ):
        if onnx_type != mlir_type:
            type_mismatches.append(
                f"  Position {i}: ONNX={onnx_type} (name='{onnx_name}'), MLIR={mlir_type}"
            )

    if type_mismatches:
        warnings.append(f"OP TYPE MISMATCHES ({len(type_mismatches)}):")
        warnings.extend(type_mismatches[:5])  # Show first 5
        if len(type_mismatches) > 5:
            warnings.append(f"  ... and {len(type_mismatches) - 5} more")
        return False, warnings

    return True, warnings


def _build_onnx_to_mlir_mapping(onnx_model_path: Path, mlir_file: Path) -> Dict[str, str]:
    """Build mapping from ONNX node names to full MLIR line numbers.

    Returns: {onnx_node_name -> "line:column"}
    """
    import onnx
    import subprocess

    try:
        # Generate full model MLIR from ONNX using iree-import-onnx
        mlir_file.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "iree.compiler.tools.import_onnx",
             str(onnx_model_path), "-o", str(mlir_file), "--data-prop"],
            check=True,
            capture_output=True,
        )

        # Extract all operations from full MLIR
        all_ops = extract_line_numbers_from_mlir(mlir_file)

        # Load ONNX model to get node names
        model = onnx.load(str(onnx_model_path))

        # Verify ordering assumptions
        is_valid, warnings = _verify_import_ordering(model.graph.node, all_ops)

        if warnings:
            _discovery_log("[ONNXtoMLIR] Verification:")
            for w in warnings:
                _discovery_log(f"  ! {w}")

        if not is_valid:
            _discovery_log(
                "[ONNXtoMLIR] WARNING: Import ordering verification failed. "
                "Mapping may be incorrect. See test_onnx_import_order.py"
            )

        # Build mapping: onnx_node_name -> line:column
        # Using position-based matching (most reliable)
        mapping = {}
        op_idx = 0
        for node in model.graph.node:
            if node.op_type == "Constant":
                continue
            if op_idx < len(all_ops):
                op_type, line_col = all_ops[op_idx]
                # Use output[0] as the key to match layer_id format
                if node.output:
                    node_name = f"{node.op_type}_{node.output[0]}"
                    mapping[node_name] = line_col
                op_idx += 1

        onnx_count = len([n for n in model.graph.node if n.op_type != "Constant"])
        _discovery_log(
            f"[ONNXtoMLIR] Mapping: {len(mapping)}/{onnx_count} ops, "
            f"valid={is_valid}"
        )

        return mapping
    except Exception as e:
        _discovery_log(f"[ONNXtoMLIR] Error building mapping: {e}")
        return {}


_QUANT_WRAPPER_OPS = {"QuantizeLinear", "DequantizeLinear"}


def _lcs_alignment(a: List[str], b: List[str]) -> List[Tuple[int, int]]:
    """Return matched index pairs from a longest-common-subsequence alignment."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    alignment: List[Tuple[int, int]] = []
    i = j = 0
    while i < m and j < n:
        if a[i] == b[j]:
            alignment.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return alignment


def _build_quantized_mlir_mapping(
    quantized_model,
    mlir_file: Path,
    original_layers: List[Tuple[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Map original layer IDs to line numbers in the final quantized MLIR.

    Quantization inserts Q/DQ wrapper nodes around compute ops and may fuse
    activations (e.g. Conv+ReLU). The final executor assignment must target
    the compute op lines in the quantized MLIR, not the Q/DQ wrappers. This
    helper imports the quantized model, extracts compute-op line numbers, and
    aligns them with the original unquantized layer order using a longest
    common subsequence so fused layers are skipped automatically.

    Returns:
        (layer_id -> "line:column", layer_id -> node_index in full quantized MLIR)
    """
    mlir_file.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = mlir_file.with_suffix(".onnx")
    try:
        model_proto = _get_model_from_wrapper(quantized_model)
        onnx.save(model_proto, str(onnx_path))
        subprocess.run(
            [sys.executable, "-m", "iree.compiler.tools.import_onnx",
             str(onnx_path), "-o", str(mlir_file), "--data-prop"],
            check=True,
            capture_output=True,
        )

        all_ops = extract_line_numbers_from_mlir(mlir_file)
        compute_ops = [
            (_normalize_quantized_op_type(op_type), loc) for op_type, loc in all_ops
            if op_type not in _QUANT_WRAPPER_OPS
        ]
        # _update_discovery_json_line_numbers uses the same compute-only list
        # (no Q/DQ wrappers) so node_index points at a compute op.
        indexed_ops = compute_ops

        original_ops: List[Tuple[str, str]] = []
        for _layer_key, layer_data in original_layers:
            model = _get_model_from_wrapper(layer_data)
            if not model.graph.node:
                continue
            op_type = model.graph.node[0].op_type
            layer_id = _get_layer_id_from_case(Case("", layer_data))
            original_ops.append((layer_id, op_type))

        alignment = _lcs_alignment(
            [op_type for _, op_type in original_ops],
            [op_type for op_type, _ in indexed_ops],
        )

        mapping: Dict[str, str] = {}
        node_indices: Dict[str, int] = {}
        for orig_idx, comp_idx in alignment:
            layer_id, _ = original_ops[orig_idx]
            op_type, loc = indexed_ops[comp_idx]
            mapping[layer_id] = loc
            node_indices[layer_id] = comp_idx

        _discovery_log(
            f"[QuantizedONNXtoMLIR] Mapped {len(mapping)}/{len(original_ops)} "
            f"original layers to {len(compute_ops)} quantized compute ops"
        )
        return mapping, node_indices
    except Exception as e:
        _discovery_log(f"[QuantizedONNXtoMLIR] Error building quantized mapping: {e}")
        return {}, {}


def _extract_op_type_from_layer(mlir_file: Path, target_op_type: str) -> Optional[str]:
    """Extract operation type from layer MLIR file."""
    if not mlir_file.exists():
        return None
    try:
        content = mlir_file.read_text()
        pattern = rf'torch\.operator\s+"onnx\.({target_op_type})"'
        if re.search(pattern, content, re.IGNORECASE):
            return target_op_type
    except Exception as e:
        _discovery_log(f"[LocationExtract] Error reading {mlir_file}: {e}")
    return None





def _get_skipped_executors(config) -> set:
    """Get set of executors to skip from --skip-executors option.

    Args:
        config: pytest config object or Config object with getoption method

    Returns:
        Set of executor names to skip (e.g., {'nss', 'css'})
    """
    skip_executors_option = config.getoption("--skip-executors", default=None)
    if skip_executors_option:
        return {e.strip().lower() for e in skip_executors_option.split(",")}
    return set()

def _get_model_from_wrapper(model_wrapper):
    """Return the underlying onnx.ModelProto from a wrapper or raw model."""
    return model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper


def _set_model_metadata(model, key: str, value: str):
    """Set a string metadata property on an onnx.ModelProto."""
    model = _get_model_from_wrapper(model)
    for prop in model.metadata_props:
        if prop.key == key:
            prop.value = value
            return
    prop = model.metadata_props.add()
    prop.key = key
    prop.value = value


def _get_model_metadata(model, key: str) -> Optional[str]:
    """Get a string metadata property from an onnx.ModelProto."""
    model = _get_model_from_wrapper(model)
    for prop in model.metadata_props:
        if prop.key == key:
            return prop.value
    return None


def _get_layer_id_from_case(case) -> str:
    """Extract layer_id (opType_outputName) from a Case's ONNX model.

    If the model carries a ``torq_original_layer_id`` metadata property (set
    when quantization renames tensors), that value is returned. Otherwise the
    first non-QDQ compute node is used so quantized layers keep their original
    op identity.
    """
    model_wrapper = case.data
    model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper

    original_id = _get_model_metadata(model, "torq_original_layer_id")
    if original_id:
        return original_id

    graph = model.graph
    for node in graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if node.output:
            return f"{node.op_type}_{node.output[0]}"
    return (
        f"{graph.node[0].op_type}_{graph.node[0].output[0]}"
        if graph.node and graph.node[0].output
        else "unknown_layer"
    )


def _build_duplicate_layer_map(cases) -> Dict[str, str]:
    """Build a mapping from duplicate layer_id to source layer_id.

    Two layers are considered duplicates when they have the same ONNX model
    signature (input/output shapes, op types, initializers) within the same
    model scope. The first-seen layer for a given signature becomes the source.
    """
    scope_sig_to_source = {}
    layer_id_to_source = {}
    for case in cases:
        layer_id = _get_layer_id_from_case(case)
        scope = case.name.split("_layer_")[0]
        model = case.data.model if hasattr(case.data, 'model') else case.data
        sig = json.dumps(model_signature(model))
        key = (scope, sig)
        if key not in scope_sig_to_source:
            scope_sig_to_source[key] = layer_id
        else:
            layer_id_to_source[layer_id] = scope_sig_to_source[key]
    return layer_id_to_source


def _copy_result_from_source_layer(
    layer_id: str, executor: str, source_layer_id: str, discovery_state: ExecutorDiscoveryState
) -> None:
    """Copy executor result from source layer to duplicate layer.

    If the source layer has a result for this executor, copy it into
    *discovery_state* and log. If not, skip the test.
    """
    source_results = discovery_state.results.get(source_layer_id, {})
    if executor in source_results:
        src = source_results[executor]
        discovery_state.record_result(
            layer_id,
            executor,
            src["status"],
            src.get("tolerance_used", DEFAULT_TOLERANCE.copy()),
            src.get("max_diff"),
            src.get("failure_report"),
            src.get("timing"),
        )
        _discovery_log(
            f"\nLayer {layer_id}: {executor.upper()} = {src['status']} "
            f"(copied from {source_layer_id})"
        )
        return

    pytest.skip(
        f"Layer {layer_id} is a duplicate of {source_layer_id}, "
        f"but {source_layer_id} was not tested on {executor.upper()}"
    )


def _maybe_skip_executor(
    request,
    layer_id: str,
    executor: str,
    model_name: Optional[str],
    subgraph_suffix: Optional[str] = None,
    discovery_state: ExecutorDiscoveryState = _discovery_state,
) -> None:
    """Skip remaining executors for a layer if one already succeeded.

    This is the "skip mode" functionality that speeds up discovery by not
    testing additional executors once one works for a layer.

    Args:
        request: Pytest request object
        layer_id: ID of the layer being tested
        executor: Current executor being tested (nss/css/host)
        model_name: Name of the model
        subgraph_suffix: Optional suffix for subgraph-specific JSON
        discovery_state: Discovery state instance (defaults to global singleton).
    """
    # Check if this executor should be skipped entirely (--skip-executors option)
    skipped_executors = _get_skipped_executors(request.config)
    if executor in skipped_executors:
        pytest.skip(f"Executor '{executor}' is in --skip-executors list")

    skip_mode = _opt(request.config, "--skip-mode", "--executor-skip-mode", default=False)
    if not skip_mode or not layer_id or not model_name:
        return

    # First check in-memory results from current test session (e.g., NSS just passed)
    if layer_id in discovery_state.results:
        for exec_name, exec_result in discovery_state.results[layer_id].items():
            if exec_result.get("status") == "success":
                pytest.skip(f"Layer {layer_id} already works with {exec_name}, skipping {executor}")

    # --recompute-cache invalidates the versioned fixture cache and forces the
    # layer to be compiled/run again. Don't let a stale persisted JSON success
    # short-circuit that, otherwise --debug-ir and similar debug flags never run.
    recompute_cache = request.config.getoption("--recompute-cache", default=False)
    if recompute_cache:
        return

    # Then check persisted results from JSON file (subgraph-specific or main model)
    json_path = _get_json_path(request.config, model_name, subgraph_suffix)
    if not json_path or not json_path.exists():
        return

    existing_data = _load_json(request.config, model_name, subgraph_suffix)
    if layer_id not in existing_data.get("ops", {}):
        return

    op_data = existing_data["ops"][layer_id]
    for exec_name, exec_result in op_data.get("executors", {}).items():
        if exec_result.get("status") == "success":
            pytest.skip(f"Layer {layer_id} already works with {exec_name}, skipping {executor}")


def _save_json(config, model_name: str, data: Dict) -> None:
    """Save JSON config for a model."""
    path = _get_json_path(config, model_name)
    save_config(path, data)

def _extract_model_name_from_case(case) -> Optional[str]:
    """Extract model name from case name."""
    return extract_model_name_from_case_name(case.name)


def _is_subgraph_case(case) -> bool:
    """Check if case is a subgraph case."""
    return "_subgraph_" in case.name


def _get_subgraph_suffix(case) -> Optional[str]:
    """Extract subgraph suffix from case name (e.g., 'subgraph_0_5')."""
    return get_subgraph_suffix_from_case_name(case.name)

def _update_json_with_results(
    json_data: Dict,
    mlir_file: Optional[Path],
    recommend_by_timing: bool = False,
    discovery_state: ExecutorDiscoveryState = _discovery_state,
) -> None:
    """Update JSON data with discovery results and line numbers."""
    update_config_with_results(
        json_data,
        discovery_state.results,
        discovery_state.node_indices,
        discovery_state.locations,
        discovery_state.full_mlir_locations,
        recommend_by_timing,
        discovery_state.orig_indices,
    )


def _save_discovery_results(
    config,
    case,
    mlir_file: Optional[Path],
    is_subgraph: bool = False,
    discovery_state: ExecutorDiscoveryState = _discovery_state,
) -> None:
    """Save discovery results to JSON file.

    For subgraph cases, saves to a subgraph-specific JSON file.
    """
    model_name = _extract_model_name_from_case(case) if case else None
    if not model_name:
        return

    # Get recommend_by_timing option
    recommend_by_timing = config.getoption("--recommend-by-timing", default=False)

    # Determine JSON path (subgraph-specific or main model)
    subgraph_suffix = _get_subgraph_suffix(case) if is_subgraph else None
    json_data = _load_json(config, model_name, subgraph_suffix)
    _update_json_with_results(json_data, mlir_file, recommend_by_timing, discovery_state)

    # Clean stale executor entries when --skip-mode or --skip-executors is active.
    # Otherwise a previous run's errors for lower-priority executors stay in the
    # JSON and make the final report look like those executors were tested.
    skip_mode = _opt(config, "--skip-mode", "--executor-skip-mode", default=False)
    skipped_executors = _get_skipped_executors(config)
    if skip_mode or skipped_executors:
        for op_data in json_data.get("ops", {}).values():
            executors = op_data.get("executors", {})
            # Remove executors the user explicitly told us to skip.
            for exc in skipped_executors:
                executors.pop(exc, None)
            # In skip mode, once a higher-priority executor succeeds we no longer
            # care about lower-priority results from earlier runs.
            if skip_mode:
                first_success_idx = None
                for idx, exc in enumerate(EXECUTOR_ORDER):
                    if executors.get(exc, {}).get("status") == "success":
                        first_success_idx = idx
                        break
                if first_success_idx is not None:
                    for exc in EXECUTOR_ORDER[first_success_idx + 1:]:
                        executors.pop(exc, None)

    if model_name:
        json_data["model_name"] = model_name

    json_name = model_name if not subgraph_suffix else f"{model_name}_{subgraph_suffix}"
    _save_json(config, json_name, json_data)

    # Generate compiler-format JSON (minimal, only what C++ pass needs)
    output_dir = _opt(config, "--output-dir", "--gen-config-output")
    compiler_path = get_compiler_config_path(model_name, output_dir, subgraph_suffix)
    save_compiler_config(compiler_path, json_data, model_name)

def _extract_max_diff(error_msg: str) -> Optional[Dict[str, float]]:
    """Extract max differences from comparison error message."""
    metrics = parse_diff_metrics(error_msg)
    if not metrics:
        return None
    # Return only the float-valued diff keys
    return {
        k: v for k, v in metrics.items()
        if k in ("max_rel_diff", "max_abs_diff") and isinstance(v, float)
    } or None


def _extract_failure_report(error_msg: str, error_type: type) -> Dict[str, str]:
    """Extract failure report from error message."""
    if error_type == AssertionError:
        metrics = parse_diff_metrics(error_msg)
        parts = []
        if "max_rel_diff" in metrics:
            parts.append(f"Max relative difference: {metrics['max_rel_diff']}")
        if "max_abs_diff" in metrics:
            parts.append(f"Max absolute difference: {metrics['max_abs_diff']}")
        if "num_differences" in metrics:
            parts.append(
                f"Number of differences: {metrics['num_differences']} out of "
                f"{metrics['total_elements']} [{metrics['diff_percentage']}%]"
            )
        return {
            "type": "accuracy_failure",
            "summary": "\n".join(parts) if parts else "Accuracy check failed",
        }
    # Other errors
    return {"type": "error", "summary": f"{error_type.__name__}: Test execution failed"}


def _resolve_op_name_to_index(op_name, name_to_index, option_name):
    """Resolve an op name to its node index."""
    if op_name in name_to_index:
        return name_to_index[op_name]
    available = list(name_to_index.keys())[:5]
    raise ValueError(
        f"Could not resolve {option_name}='{op_name}'. Available: {available}..."
    )

def _discover_model_files(config) -> List[Path]:
    """Discover ONNX model files from --model-path or default directories."""
    onnx_model_path = _opt(config, "--model", "--model-path")
    if onnx_model_path:
        path = Path(onnx_model_path)
        if not path.exists():
            raise pytest.UsageError(f"--model-path does not exist: {onnx_model_path}")
        return [path]
    files = list_files("dev_ops", ".onnx", False) + list_files("onnx_models", ".onnx", False)
    if not files:
        pytest.skip("No ONNX models found")
    return files


def _precompute_mlir_mappings(files: List[Path], config) -> Dict[str, Dict[str, str]]:
    """Pre-compute ONNX-to-full-MLIR line-number mappings for each model."""
    onnx_to_mlir_map = {}
    if not hasattr(pytest_generate_tests, '_onnx_to_mlir_maps'):
        pytest_generate_tests._onnx_to_mlir_maps = {}

    for f in files:
        model_name = f.stem
        mlir_cache = Path(f".pytest_cache/onnx_full_mlir/{model_name}.mlir")

        if model_name not in pytest_generate_tests._onnx_to_mlir_maps:
            mapping = _build_onnx_to_mlir_mapping(f, mlir_cache)
            pytest_generate_tests._onnx_to_mlir_maps[model_name] = mapping

        onnx_to_mlir_map[model_name] = pytest_generate_tests._onnx_to_mlir_maps[model_name]

    return onnx_to_mlir_map

def _maybe_apply_bf16_conversion(model, f: Path, auto_convert: bool, save_path: Optional[str]) -> Any:
    """Apply BF16 conversion to a model if requested.

    Returns the (possibly converted) model.
    """
    if not auto_convert:
        return model
    from torq.testing.onnx import convert_fp32_to_bf16, is_model_bf16
    if is_model_bf16(model):
        _discovery_log(f"[BF16] Model {f.name} already in BF16 format")
        return model
    _discovery_log(f"[BF16] Converting {f.name} to BF16...")
    converted = convert_fp32_to_bf16(model)
    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(converted, str(sp))
        _discovery_log(f"[BF16] Saved converted model to: {sp}")
    return converted


def _maybe_apply_quantization(
    model,
    quantize: bool,
    per_channel: bool,
    full_integer: bool,
    quant_format: str = "qdq",
    label: str = "model",
) -> Any:
    """Apply int8 static quantization to a model if requested.

    Returns the (possibly quantized) model. Logs the operation.
    """
    if not quantize:
        return model

    # Unwrap ModelWithMetadata if needed
    wrapped = None
    if hasattr(model, "model") and hasattr(model, "_model"):
        wrapped = model
        model_proto = model.model
    else:
        model_proto = model

    if is_model_quantized(model_proto):
        _discovery_log(f"[Quantize] {label} already quantized, skipping")
        return model

    _discovery_log(
        f"[Quantize] Quantizing {label} (quant_format={quant_format}, "
        f"per_channel={per_channel}, full_integer={full_integer})..."
    )
    try:
        quantized = quantize_onnx_model(
            model_proto,
            per_channel=per_channel,
            full_integer=full_integer,
            quant_format=quant_format,
        )
    except Exception as e:
        _discovery_log(f"[Quantize] Failed to quantize {label}: {e}; using original model")
        return model

    if wrapped is not None:
        wrapped._model = quantized
        return wrapped
    return quantized

def _generate_subgraph_cases(f: Path, config) -> List[Case]:
    """Generate test cases in subgraph mode."""
    subgraph_from = config.getoption("--subgraph-from")
    subgraph_to = config.getoption("--subgraph-to")
    auto_convert_bf16 = config.getoption("--auto-convert-bf16", default=False)
    save_bf16_path = config.getoption("--save-bf16-model", default=None)
    quantize = config.getoption("--quantize", default=False)
    per_channel = config.getoption("--per-channel", default=False)
    full_integer = config.getoption("--full-integer", default=False)
    quant_format = config.getoption("--quant-format", default="qdq")
    model_name = f.stem

    if _is_torch_model(f):
        raise ValueError(f"Torch models are not supported in this branch: {f}")

    full_model = get_full_model(str(f))
    full_model = _maybe_apply_bf16_conversion(full_model, f, auto_convert_bf16, save_bf16_path)

    # Build name -> index mapping from the (possibly BF16 but not yet quantized)
    # full model so indices match the original ONNX node positions.
    all_layers = generate_onnx_layers_from_model(full_model, node_groups=None, dedup=False, quantize=quantize)
    name_to_index = {}
    for layer_data in all_layers.values():
        layer_node_index = getattr(layer_data, 'node_index', None)
        if layer_node_index is None:
            continue
        model = layer_data.model if hasattr(layer_data, 'model') else layer_data
        graph = model.graph
        if not graph.node:
            continue
        op_type = graph.node[0].op_type
        output_name = graph.node[0].output[0]
        name_to_index[f"{op_type}_{output_name}"] = layer_node_index
        name_to_index[output_name] = layer_node_index

    from_index = _resolve_op_name_to_index(subgraph_from, name_to_index, "--subgraph-from")
    to_index = _resolve_op_name_to_index(subgraph_to, name_to_index, "--subgraph-to")
    _discovery_log(f"[Subgraph] Resolved '{subgraph_from}' -> {from_index}, "
                   f"'{subgraph_to}' -> {to_index}")

    _discovery_log(f"[Subgraph] Extracting subgraph from node {from_index} to "
                   f"{to_index} from {f.name}")
    try:
        subgraph = extract_onnx_subgraph(full_model, from_index, to_index, quantize=quantize)
    except Exception as e:
        _discovery_log(f"[Subgraph] Error extracting subgraph: {e}")
        pytest.skip(f"Failed to extract subgraph: {e}")
        return []

    subgraph.source_model_path = str(f)

    subgraph_suffix = f"subgraph_{from_index}_{to_index}"
    source_model_path = str(f)

    # Build a subgraph-specific ONNX-to-MLIR mapping so the executor
    # assignment pass matches the subgraph MLIR (not the full-model MLIR).
    subgraph_onnx_path = config.cache.mkdir('subgraph_onnx') / f"{model_name}_{subgraph_suffix}.onnx"
    subgraph_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(subgraph.model, str(subgraph_onnx_path))
    subgraph_mlir_cache = Path(f".pytest_cache/onnx_full_mlir/{model_name}_{subgraph_suffix}.mlir")
    map_key = f"{model_name}_{subgraph_suffix}"
    if not hasattr(pytest_generate_tests, '_onnx_to_mlir_maps'):
        pytest_generate_tests._onnx_to_mlir_maps = {}

    # Extract layers from the UNQUANTIZED subgraph first so layer IDs stay
    # tied to the original compute ops, then quantize each layer individually.
    subgraph_layers = generate_onnx_layers_from_model(
        subgraph.model, node_groups=None, dedup=False, quantize=quantize
    )
    _discovery_log(f"[Subgraph] Extracted {len(subgraph_layers)} layers from subgraph")

    # Quantize the full subgraph for the full-model test.
    if quantize:
        subgraph = _maybe_apply_quantization(
            subgraph,
            quantize=True,
            per_channel=per_channel,
            full_integer=full_integer,
            quant_format=quant_format,
            label=f"subgraph {from_index}-{to_index}",
        )

    # Build the ONNX->MLIR mapping from the final model that the compiler will
    # see. For quantized models this must be the quantized MLIR so that executor
    # assignments target the actual compute ops (not inserted Q/DQ wrappers).
    if quantize:
        quantized_subgraph_mlir_cache = subgraph_mlir_cache.with_suffix(".quantized.mlir")
        subgraph_mapping, quantized_node_indices = _build_quantized_mlir_mapping(
            subgraph,
            quantized_subgraph_mlir_cache,
            list(subgraph_layers.items()),
        )
    else:
        subgraph_mapping = _build_onnx_to_mlir_mapping(subgraph_onnx_path, subgraph_mlir_cache)
        quantized_node_indices = {}
    pytest_generate_tests._onnx_to_mlir_maps[map_key] = subgraph_mapping
    _discovery_log(
        f"[Subgraph] Built subgraph MLIR mapping ({len(subgraph_mapping)} ops) "
        f"under key {map_key}"
    )

    node_count = len(subgraph.model.graph.node)
    _discovery_log(f"[Subgraph] Successfully extracted subgraph with {node_count} nodes")

    cases = []
    for layer_key, layer_data in subgraph_layers.items():
        case_name = f"{model_name}_{subgraph_suffix}_{layer_key}"
        original_layer_id = _get_layer_id_from_case(Case("", layer_data))
        quantized_layer = _maybe_apply_quantization(
            layer_data,
            quantize=quantize,
            per_channel=per_channel,
            full_integer=full_integer,
            quant_format=quant_format,
            label=f"subgraph layer {layer_key}",
        )
        quantized_layer.source_model_path = source_model_path
        _set_model_metadata(quantized_layer, "torq_original_layer_id", original_layer_id)
        if quantize and original_layer_id in quantized_node_indices:
            quantized_layer.node_index = quantized_node_indices[original_layer_id]
        cases.append(Case(case_name, quantized_layer))
    cases.append(Case(f"{model_name}_{subgraph_suffix}_full", subgraph))
    return cases

def _generate_layer_cases(f: Path, config) -> List[Case]:
    """Generate test cases in normal layer-extraction mode."""
    auto_convert_bf16 = config.getoption("--auto-convert-bf16", default=False)
    save_bf16_path = config.getoption("--save-bf16-model", default=None)
    quantize = config.getoption("--quantize", default=False)
    per_channel = config.getoption("--per-channel", default=False)
    full_integer = config.getoption("--full-integer", default=False)
    quant_format = config.getoption("--quant-format", default="qdq")

    if _is_torch_model(f):
        raise ValueError(f"Torch models are not supported in this branch: {f}")

    if auto_convert_bf16 and not quantize:
        # Original BF16-only path: convert full model, then extract layers.
        model = get_full_model(str(f))
        model = _maybe_apply_bf16_conversion(model, f, auto_convert_bf16, save_bf16_path)
        layers = generate_onnx_layers_from_model(model, node_groups=None, dedup=False, quantize=False)
        return [
            Case(f"{f.stem}_{key}", layer)
            for key, layer in layers.items()
        ] + [Case(f"{f.stem}_full_model", model)]

    if quantize:
        # Quantize path: extract layers from the original full model first, then
        # quantize each layer individually so per-layer tests use their own
        # calibration data and the ONNX-to-MLIR mapping stays tied to the
        # original op positions. BF16 conversion is mutually exclusive with
        # quantization at the CLI level.
        source_model_path = str(f)
        model = get_full_model(str(f))
        layers = generate_onnx_layers_from_model(model, node_groups=None, dedup=False, quantize=True)

        # Quantize the full model first so the ONNX->MLIR mapping is built from
        # the final quantized MLIR while the unquantized layer list is still
        # intact (quantizing layers in place would mutate the list below).
        quantized_full = _maybe_apply_quantization(
            model,
            quantize=True,
            per_channel=per_channel,
            full_integer=full_integer,
            quant_format=quant_format,
            label=f"full model {f.name}",
        )
        quantized_mlir_cache = Path(f".pytest_cache/onnx_full_mlir/{f.stem}.quantized.mlir")
        mapping, node_indices = _build_quantized_mlir_mapping(
            quantized_full, quantized_mlir_cache, list(layers.items())
        )
        if not hasattr(pytest_generate_tests, '_onnx_to_mlir_maps'):
            pytest_generate_tests._onnx_to_mlir_maps = {}
        pytest_generate_tests._onnx_to_mlir_maps[f.stem] = mapping

        cases = []
        for key, layer in layers.items():
            original_layer_id = _get_layer_id_from_case(Case("", layer))

            quantized_layer = _maybe_apply_quantization(
                layer,
                quantize=True,
                per_channel=per_channel,
                full_integer=full_integer,
                quant_format=quant_format,
                label=f"layer {key}",
            )
            quantized_layer.source_model_path = source_model_path
            _set_model_metadata(quantized_layer, "torq_original_layer_id", original_layer_id)
            if original_layer_id in node_indices:
                quantized_layer.node_index = node_indices[original_layer_id]
            cases.append(Case(f"{f.stem}_{key}", quantized_layer))

        cases.append(Case(f"{f.stem}_full_model", quantized_full))
        return cases

    return generate_onnx_layers_from_file(config.cache, f)


def _assemble_layer_test_cases(
    cases: List[Case],
    onnx_to_mlir_map: Dict[str, Dict[str, str]],
    skipped_executors: set,
    dedup_layers: bool,
) -> List[tuple]:
    """Build param tuples for per-layer / per-executor parametrization."""
    non_full_cases = [
        c for c in cases
        if "_full_model" not in c.name and not c.name.endswith("_full")
    ]

    layer_id_to_source = _build_duplicate_layer_map(non_full_cases) if dedup_layers else {}

    test_cases = []
    for orig_index, case in enumerate(non_full_cases):
        model_wrapper = case.data
        node_index = getattr(model_wrapper, 'node_index', None)

        if "_subgraph_" in case.name:
            model_name = case.name.split("_subgraph_")[0]
            subgraph_suffix = _get_subgraph_suffix(case)
            map_key = f"{model_name}_{subgraph_suffix}" if subgraph_suffix else model_name
            is_subgraph = True
        else:
            model_name = case.name.split("_layer_")[0]
            map_key = model_name
            is_subgraph = False

        layer_id = _get_layer_id_from_case(case)
        # Subgraph cases use a subgraph-specific MLIR mapping built during
        # case generation; full-model cases use the precomputed full mapping.
        global_maps = getattr(pytest_generate_tests, '_onnx_to_mlir_maps', {})
        full_mlir_location = (
            global_maps.get(map_key, onnx_to_mlir_map.get(model_name, {})).get(layer_id)
            if model_name else None
        )
        source_layer_id = layer_id_to_source.get(layer_id)

        for executor in EXECUTOR_ORDER:
            if executor in skipped_executors:
                continue
            test_cases.append(
                (case, layer_id, executor, node_index, full_mlir_location, is_subgraph, source_layer_id, orig_index)
            )

    # Full model / full subgraph tests
    for case in [c for c in cases if "_full_model" in c.name or c.name.endswith("_full")]:
        is_subgraph = "_subgraph_" in case.name
        test_cases.append((case, None, "discovered", None, None, is_subgraph, None, None))

    return test_cases

def pytest_generate_tests(metafunc):
    """Generate test cases for each layer, subgraph, or full model."""
    files = _discover_model_files(metafunc.config)
    if not files:
        return

    skipped_executors = _get_skipped_executors(metafunc.config)
    if skipped_executors:
        _discovery_log(f"[SkipExecutors] Will skip executors: {sorted(skipped_executors)}")

    onnx_to_mlir_map = _precompute_mlir_mappings(files, metafunc.config)

    subgraph_mode = (
        metafunc.config.getoption("--subgraph-from", default=None) is not None
        and metafunc.config.getoption("--subgraph-to", default=None) is not None
    )

    cases = []
    for f in files:
        if subgraph_mode:
            cases += _generate_subgraph_cases(f, metafunc.config)
        else:
            cases += _generate_layer_cases(f, metafunc.config)

    dedup_layers = metafunc.config.getoption("--dedup-layers", default=False)
    test_cases = _assemble_layer_test_cases(cases, onnx_to_mlir_map, skipped_executors, dedup_layers)

    metafunc.parametrize(
        "layer_executor_case",
        test_cases,
        indirect=True,
        ids=[f"{c.name}_{exec}" for c, _, exec, _, _, _, _, _ in test_cases],
    )

@pytest.fixture
def reference_results(request, onnx_layer_model):
    return request.getfixturevalue("composite_reference_results")


@pytest.fixture

def layer_executor_case(request):
    """Provide the layer case with executor info."""
    case, layer_id, executor, node_index, full_mlir_location, is_subgraph, source_layer_id, orig_index = request.param
    return {
        "case": case,
        "layer_id": layer_id,
        "executor": executor,
        "node_index": node_index,
        "orig_index": orig_index,
        "full_mlir_location": full_mlir_location,
        "is_subgraph": is_subgraph,
        "source_layer_id": source_layer_id,
    }

@pytest.fixture
def onnx_layer_model(request, layer_executor_case):
    """Provide the layer model."""
    # Early skip for duplicate layers: if the source layer has already been
    # tested on this executor, copy its result and skip all expensive fixture
    # setup (compilation, reference generation, runtime).
    source_layer_id = layer_executor_case.get("source_layer_id")
    if source_layer_id:
        layer_id = layer_executor_case["layer_id"]
        executor = layer_executor_case["executor"]
        source_results = _discovery_state.results.get(source_layer_id, {})
        if executor in source_results:
            _copy_result_from_source_layer(layer_id, executor, source_layer_id, _discovery_state)
            node_index = layer_executor_case.get("node_index")
            orig_index = layer_executor_case.get("orig_index")
            full_mlir_location = layer_executor_case.get("full_mlir_location")
            _discovery_state.record_metadata(
                layer_id,
                node_index=node_index,
                orig_index=orig_index,
                full_mlir_location=full_mlir_location,
            )
            pytest.skip(
                f"Duplicate of {source_layer_id}, result copied"
            )

    case = layer_executor_case["case"]
    version = "onnx_layer_" + case.name
    return VersionedUncachedData(data=case.data, version=version)

@versioned_hashable_object_fixture
def torq_compiler_options(request, case_config):
    """Override torq_compiler_options to add executor map for full model tests."""
    # Start with the compiler options from case_config
    cmds = case_config.get("torq_compiler_options", [])

    if request.config.getoption("--extra-torq-compiler-options"):
        cmds.extend(request.config.getoption("--extra-torq-compiler-options").split(" "))

    if request.config.getoption("--trace-buffers"):
        cmds.append("--torq-enable-buffer-debug-info")

    gdb_port = request.config.getoption("--debug-torq-compiler")
    if gdb_port > 0:
        cmds = ["gdbserver", "localhost:" + str(gdb_port)] + cmds

    # Only add executor map for full model tests (discovered executor mode)
    # and only if case_config doesn't already provide one
    # (Layer tests provide their own temporary executor maps)
    has_executor_map = any("--torq-executor-map=" in str(opt) for opt in cmds)
    executor = request.getfixturevalue("layer_executor_case")["executor"]

    if not has_executor_map and executor == "discovered":
        try:
            torq_torq_gen_config_json = request.getfixturevalue(
                "torq_torq_gen_config_json"
            )
            if (
                torq_torq_gen_config_json
                and torq_torq_gen_config_json.file_path.exists()
            ):
                cmds.append(
                    f"--torq-executor-map={torq_torq_gen_config_json.file_path}"
                )
        except pytest.FixtureLookupError:
            pass

    return cmds

@pytest.fixture
def comparison_config_for_executor_discovery(request, layer_executor_case):
    """Provide comparison config with per-op tolerances."""
    layer_id = layer_executor_case["layer_id"]
    case = layer_executor_case["case"]
    is_subgraph = layer_executor_case.get("is_subgraph", False)
    model_name = _extract_model_name_from_case(case)
    subgraph_suffix = _get_subgraph_suffix(case) if is_subgraph else None
    json_data = _load_json(request.config, model_name, subgraph_suffix) if model_name else {}
    tolerance = get_tolerance(layer_id, json_data)

    config = {
        "int_tol": 1,
        "int_thld": 1,
        "fp_avg_tol": tolerance.get("fp_avg_tol", 0.01),
        "fp_max_tol": tolerance.get("fp_max_tol", 0.01),
        "epsilon": 1e-6,
        "allow_all_zero": False,
        "skip_nan_check": False,
    }

    return config

@pytest.fixture(autouse=True)
def save_progress(request, layer_executor_case):
    """Save discovery progress after each layer test.

    Clears in-memory discovery state when switching to a different model
    so that each model's JSON only contains its own layers.
    """
    case = layer_executor_case.get("case") if layer_executor_case else None
    if case:
        current_model = _extract_model_name_from_case(case)
        prev_model = getattr(save_progress, "_last_model", None)
        if prev_model and prev_model != current_model:
            # Model changed — reset discovery state to prevent cross-model
            # pollution in the JSON output.
            _discovery_state.results.clear()
            _discovery_state.locations.clear()
            _discovery_state.node_indices.clear()
            _discovery_state.full_mlir_locations.clear()
            _discovery_state.recommended_executors.clear()
        save_progress._last_model = current_model

    yield

    # Skip full model tests — they don't perform discovery and should not
    # overwrite the discovery JSON or compiler JSON.
    is_subgraph = layer_executor_case.get("is_subgraph", False)
    if case and "_full_model" not in case.name and "_full" not in case.name:
        _save_discovery_results(request.config, case, None, is_subgraph)

def _run_layer_test(
    request, torq_results, reference_results, case_config,
    layer_id, executor, node_index, mlir_file_path, json_data, case_name,
    discovery_state: ExecutorDiscoveryState = _discovery_state,
):
    """Run a single layer test with result recording to JSON."""
    from torq.testing.comparison import compare_test_results
    import time

    tolerance_used = get_tolerance(layer_id, json_data)
    collect_timing = request.config.getoption("--collect-timing", default=False)
    timing_runs = request.config.getoption("--timing-runs", default=1)

    def _record(status, max_diff=None, failure_report=None, timing=None):
        discovery_state.record_result(
            layer_id, executor, status, tolerance_used, max_diff, failure_report, timing
        )
        if node_index is not None:
            discovery_state.record_metadata(layer_id, node_index=node_index)

    def _format_timing_str(timing: Optional[Dict]) -> str:
        """Format timing for display."""
        if timing and "runtime_ms" in timing:
            return f" ({timing['runtime_ms']}ms)"
        return ""

    runtime_times = []

    try:
        for _ in range(timing_runs if collect_timing else 1):
            run_start = time.perf_counter()
            try:
                compare_test_results(request, torq_results, reference_results, case_config)
            except AssertionError:
                raise
            finally:
                if collect_timing:
                    runtime_times.append((time.perf_counter() - run_start) * 1000)

        timing = build_timing_data(runtime_times) if collect_timing else None

        _record("success", timing=timing)

        op_type = case_name.split("_")[-2]
        loc = _extract_op_type_from_layer(mlir_file_path, op_type)
        if loc:
            discovery_state.record_metadata(layer_id, mlir_location=loc)

        _discovery_log(f"\nLayer {layer_id}: {executor.upper()} = success{_format_timing_str(timing)}")

    except AssertionError as e:
        max_diff = _extract_max_diff(str(e))
        failure_report = _extract_failure_report(str(e), AssertionError)
        timing = build_timing_data(runtime_times) if collect_timing else None

        _record("difference", max_diff, failure_report, timing)
        _discovery_log(f"\nLayer {layer_id}: {executor.upper()} = difference{_format_timing_str(timing)}")
        if failure_report.get("summary"):
            _discovery_log(f"  {failure_report['summary']}")
        pytest.fail(f"Layer {layer_id} failed on {executor.upper()} with difference")

    except Exception as e:
        failure_report = _extract_failure_report(str(e), type(e))
        timing = build_timing_data(runtime_times) if collect_timing else None

        _record("error", failure_report=failure_report, timing=timing)
        _discovery_log(f"\nLayer {layer_id}: {executor.upper()} = error")
        # Re-raise the original exception so pytest marks it as ERROR
        # (pytest.fail() would produce FAILED, which is for assertion failures)
        raise

def executor_discovery(
    request,
    torq_results,
    reference_results,
    case_config,
    layer_executor_case,
    onnx_mlir_model_file,
    discovery_state: ExecutorDiscoveryState = _discovery_state,
):
    """Core executor discovery implementation."""
    layer_id = layer_executor_case["layer_id"]
    executor = layer_executor_case["executor"]
    node_index = layer_executor_case.get("node_index")
    orig_index = layer_executor_case.get("orig_index")
    full_mlir_location = layer_executor_case.get("full_mlir_location")
    is_subgraph = layer_executor_case.get("is_subgraph", False)
    case = layer_executor_case["case"]

    # Determine which JSON to load (subgraph-specific or main model)
    subgraph_suffix = _get_subgraph_suffix(case) if is_subgraph else None
    model_name = _extract_model_name_from_case(case)

    # Load existing JSON results on first test (for consistent reporting in skip mode)
    if not hasattr(executor_discovery, "_loaded_existing_json"):
        executor_discovery._loaded_existing_json = {}

    json_key = f"{model_name}_{subgraph_suffix}" if subgraph_suffix else model_name
    if json_key not in executor_discovery._loaded_existing_json:
        executor_discovery._loaded_existing_json[json_key] = True
        if model_name:
            json_data = _load_json(request.config, model_name, subgraph_suffix)
            if json_data:
                discovery_state.load_from_json(json_data)
                if subgraph_suffix:
                    discovery_state.subgraph_suffix = subgraph_suffix

    # Full model / full subgraph test: no layer_id, just compare
    if layer_id is None:
        from torq.testing.comparison import compare_test_results
        import io
        import contextlib

        capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(capture):
                compare_test_results(request, torq_results, reference_results, case_config)
        finally:
            # Always print the comparison report and store the metrics so the
            # final run report is visible regardless of pass/fail.
            captured = capture.getvalue()
            if captured:
                print(captured, end="")
            metrics = {}
            for line in captured.splitlines():
                if line.startswith("Max relative difference:"):
                    metrics["max_rel_diff"] = line.split(":", 1)[1].strip()
                elif line.startswith("Max absolute difference:"):
                    metrics["max_abs_diff"] = line.split(":", 1)[1].strip()
                elif line.startswith("Number of differences:"):
                    metrics["num_differences"] = line.split(":", 1)[1].strip()
            discovery_state.full_model_metrics = metrics
        return

    # Layer mode: record metadata first so duplicates also get correct
    # node_index, orig_index and mlir_location in the JSON output.
    discovery_state.record_metadata(
        layer_id=layer_id,
        node_index=node_index,
        orig_index=orig_index,
        full_mlir_location=full_mlir_location,
        mlir_file=(
            Path(onnx_mlir_model_file.file_path)
            if onnx_mlir_model_file and hasattr(onnx_mlir_model_file, 'file_path')
            else None
        ),
    )

    # Short-circuit duplicate layers: copy results from source layer
    source_layer_id = layer_executor_case.get("source_layer_id")
    if source_layer_id:
        _copy_result_from_source_layer(layer_id, executor, source_layer_id, discovery_state)
        return

    json_data = _load_json(request.config, model_name, subgraph_suffix) if model_name else {}

    mlir_file_path = (
        Path(onnx_mlir_model_file.file_path)
        if onnx_mlir_model_file and hasattr(onnx_mlir_model_file, 'file_path')
        else None
    )

    _run_layer_test(
        request, torq_results, reference_results, case_config,
        layer_id, executor, node_index, mlir_file_path, json_data, case.name
    )
