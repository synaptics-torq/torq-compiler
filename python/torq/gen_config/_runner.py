# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Standalone executor discovery and full-model run engine.

This module is the runtime heart of ``torq-gen-config discover`` /
``torq-gen-config run``: case generation, skip/dedup logic, per-layer
outcome recording, and discovery/compiler JSON resolution, all driven by a
plain :class:`~torq.gen_config._options.DiscoveryConfig`.  It builds on:

- :class:`~torq.gen_config._cache.Cache` for the content-versioned artifact
  cache and small-value store (the ``get_or_generate_*`` cache drivers);
- :class:`torq.lab.pipeline.ModelPipeline` to drive ``torq-compile`` /
  ``torq-run-module``;
- :func:`torq.lab.compare.compare_outputs` for the numeric core, keeping
  the exact per-tensor metric printout the report parsing relies on;
- a reference-output chain (ONNXRuntime -> numpy -> IREE llvm-cpu) built on
  ``torq.lab.reference``.  There is no torch fallback (torch is a test-only
  dependency); if all three references fail, the error propagates.

Behavior notes:

- Timing (``--collect-timing``): every timing iteration performs a fresh
  ``torq-run-module`` subprocess run plus comparison, so ``runtime_ms``
  reflects real inference wall time.  The JSON schema is
  ``runtime_ms``/``runs``.
- ``--debug-ir=<dir>`` (string form): the IR is dumped into the per-case
  cache directory (``<compile dir>/debug/ir``) and then copied to ``<dir>``
  relative to the current working directory.
- ``--debug-ir`` (bare/bool form): IR lands in ``<compile dir>/debug/ir``
  instead of ``<compile dir>/ir``.
- Reference outputs are cached on disk keyed by MLIR content hash.  All
  reference backends used here are deterministic, so this only saves time.
- Outside a source checkout, chip group files (``extras/chips``,
  ``tests/testdata/chips``) are unavailable: ``default.group`` falls back to
  the built-in ``default`` chip (target ``SL2610``); other groups raise a
  clear error.

Exit code: 0 when every case passed or was skipped, 1 on any
difference/error.

This module must not import ``torq.testing`` or the test framework.
"""

import contextlib
import hashlib
import io
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import onnx
import numpy as np

from torq.gen_config._cache import (
    Cache,
    _hash_data,
    get_or_compute_data,
    get_or_generate_directory,
    get_or_generate_file,
)
from torq.lab.onnx import (
    convert_onnx_to_mlir,
    extract_onnx_subgraph,
    generate_onnx_layers_from_file,
    generate_onnx_layers_from_model,
    get_full_model,
    model_signature,
)
from torq.lab.types import Case
from torq.gen_config._options import DiscoveryConfig
from torq.gen_config._report import _print_final_report, _save_detailed_report
from torq.gen_config._state import ExecutorDiscoveryState, _discovery_state
from torq.gen_config._utils import (
    _normalize_quantized_op_type,
    extract_line_numbers_from_mlir,
    parse_diff_metrics,
)
from torq.gen_config._utils_mac import compute_model_mac_details
from torq.gen_config.core import (
    DEFAULT_TOLERANCE,
    EXECUTOR_ORDER,
    _discovery_log,
    _discovery_vlog,
    _get_json_path,
    _load_json,
    build_timing_data,
    extract_model_name_from_case_name,
    generate_compiler_config,
    get_compiler_config_path,
    get_mac_debug_config_path,
    get_subgraph_suffix_from_case_name,
    get_tolerance,
    is_verbose,
    load_config,
    save_compiler_config,
    save_config,
    save_mac_debug_config,
    set_verbose,
    update_config_with_results,
)
from torq.lab.quantize_onnx import (
    is_model_quantized,
    quantize_onnx_model,
)

logger = logging.getLogger("torq.gen_config.runner")

# Repo root when running from a source checkout (python/torq/gen_config/ ->
# repo root).  Chip group/JSON files are looked up relative to this, matching
# TOPDIR in torq.testing.iree.
_TOPDIR = Path(__file__).resolve().parents[3]


class CaseGenSkip(Exception):
    """Abort case generation for a model; the run loop records it as a skip."""


class LayerDifference(AssertionError):
    """A layer ran but its outputs diverged beyond tolerance; recorded as a difference."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

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
        config: DiscoveryConfig with the discovery options

    Returns:
        Set of executor names to skip (e.g., {'nss', 'css'})
    """
    skip_executors_option = config.skip_executors
    if skip_executors_option:
        return {e.strip().lower() for e in skip_executors_option.split(",")}
    return set()


def _get_skipped_ops(config) -> set:
    """Get set of ONNX op types to skip from --skip-ops option."""
    skip_ops_option = config.skip_ops
    if skip_ops_option:
        return {op.strip().lower() for op in skip_ops_option.split(",")}
    return set()


def _get_op_type_from_layer_id(layer_id: str) -> Optional[str]:
    """Extract ONNX op type from a layer_id such as 'MaxPool_/foo/MaxPool_output_0'."""
    if not layer_id:
        return None
    parts = layer_id.split("_", 1)
    return parts[0] if parts else None


# Tracks layers skipped by --skip-ops so their JSON entries are written once.
_skipped_op_layers: Dict[str, Dict[str, Any]] = {}


def _save_json(config, model_name: str, data: Dict) -> None:
    """Save JSON config for a model."""
    path = _get_json_path(config, model_name)
    save_config(path, data)


def _record_skipped_op_in_json(
    config,
    layer_id: str,
    model_name: str,
    subgraph_suffix: Optional[str],
    node_index: Optional[int] = None,
    orig_index: Optional[int] = None,
    full_mlir_location: Optional[str] = None,
) -> None:
    """Ensure a skipped op has a JSON entry with recommended_executor=None."""
    json_data = _load_json(config, model_name, subgraph_suffix)
    ops = json_data.setdefault("ops", {})
    if layer_id not in ops:
        ops[layer_id] = {"executors": {}}
    op_data = ops[layer_id]
    op_data["recommended_executor"] = None

    if node_index is not None:
        op_data["_node_index"] = node_index
    if orig_index is not None:
        op_data["_orig_index"] = orig_index
    if full_mlir_location and re.match(r"^\d+:\d+$", full_mlir_location):
        op_data["mlir_location"] = full_mlir_location

    json_name = model_name if not subgraph_suffix else f"{model_name}_{subgraph_suffix}"
    _save_json(config, json_name, json_data)

    output_dir = config.output_dir
    compiler_path = get_compiler_config_path(model_name, output_dir, subgraph_suffix)
    save_compiler_config(compiler_path, json_data, model_name)


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
    model = _get_model_from_wrapper(case.data)

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


def _compute_layer_mac_info(case) -> Optional[Dict[str, Any]]:
    """Compute MAC count plus per-node metadata for a layer Case's ONNX model."""
    if case is None or not hasattr(case, "data") or case.data is None:
        return None
    model = _get_model_from_wrapper(case.data)
    try:
        return compute_model_mac_details(model)
    except Exception as e:
        _discovery_log(f"[MacCount] Failed to compute MAC details: {e}")
        return None


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
        model = _get_model_from_wrapper(case.data)
        sig = json.dumps(model_signature(model))
        key = (scope, sig)
        if key not in scope_sig_to_source:
            scope_sig_to_source[key] = layer_id
        else:
            layer_id_to_source[layer_id] = scope_sig_to_source[key]
    return layer_id_to_source


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
        discovery_state.mac_counts,
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
    recommend_by_timing = config.recommend_by_timing

    # Determine JSON path (subgraph-specific or main model)
    subgraph_suffix = _get_subgraph_suffix(case) if is_subgraph else None
    json_data = _load_json(config, model_name, subgraph_suffix)
    _update_json_with_results(json_data, mlir_file, recommend_by_timing, discovery_state)

    # Clean stale executor entries when --skip-mode or --skip-executors is active.
    # Otherwise a previous run's errors for lower-priority executors stay in the
    # JSON and make the final report look like those executors were tested.
    skip_mode = config.skip_mode
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

    # Per-layer extraction strips QDQ around MaxPool, so the layer appears as
    # float32 and discovery cannot recommend a valid executor. Leave it unset
    # and let the full-model C++ pattern fuse the MaxPool QDQ chain instead.
    if not is_subgraph and config.quantize:
        for layer_id, op_data in json_data.get("ops", {}).items():
            if layer_id.startswith("MaxPool_"):
                op_data["recommended_executor"] = None

    if model_name:
        json_data["model_name"] = model_name

    json_name = model_name if not subgraph_suffix else f"{model_name}_{subgraph_suffix}"
    _save_json(config, json_name, json_data)

    # Generate compiler-format JSON (minimal, only what C++ pass needs)
    output_dir = config.output_dir
    compiler_path = get_compiler_config_path(model_name, output_dir, subgraph_suffix)
    save_compiler_config(compiler_path, json_data, model_name)

    # Generate MAC-count debug JSON with per-node metadata used to compute MACs
    mac_debug_path = get_mac_debug_config_path(model_name, output_dir, subgraph_suffix)
    save_mac_debug_config(
        mac_debug_path,
        discovery_state.mac_counts,
        discovery_state.mac_details,
        model_name,
    )


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


def _maybe_apply_bf16_conversion(model, f: Path, auto_convert: bool, save_path: Optional[str]) -> Any:
    """Apply BF16 conversion to a model if requested.

    Returns the (possibly converted) model.
    """
    if not auto_convert:
        return model
    from torq.lab.convert_onnx import convert_fp32_to_bf16, is_model_bf16
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


def _maybe_apply_int32_conversion(model, f: Path, auto_convert: bool) -> Any:
    """Apply INT64->INT32 conversion to a model if requested.

    Mirrors _maybe_apply_bf16_conversion. Call it after the BF16 step so the
    order is bf16 first, then int32, when both conversions are enabled.
    Returns the (possibly converted) model.
    """
    if not auto_convert:
        return model
    from torq.lab.convert_onnx import convert_int64_to_int32, is_model_int32
    if is_model_int32(model):
        _discovery_log(f"[INT32] Model {f.name} has no INT64 tensors")
        return model
    _discovery_log(f"[INT32] Converting INT64 tensors in {f.name} to INT32...")
    return convert_int64_to_int32(model)


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


# ---------------------------------------------------------------------------
# Skip / dedup / layer-outcome cores (skip/fail surfaced as values)
# ---------------------------------------------------------------------------


def _check_skip_executor(
    config,
    layer_id: str,
    executor: str,
    model_name: Optional[str],
    subgraph_suffix: Optional[str] = None,
    discovery_state: ExecutorDiscoveryState = _discovery_state,
    layer_executor_case: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Check whether a (layer, executor) case should be skipped.

    Runs the checks (--skip-executors, --skip-ops with JSON
    recording, --skip-mode against in-memory and persisted results) and
    returns the skip reason, or None when the case should run.  The caller
    decides how to surface the skip (a logged skip in the standalone
    runner).
    """
    # Check if this executor should be skipped entirely (--skip-executors option)
    skipped_executors = _get_skipped_executors(config)
    if executor in skipped_executors:
        return f"Executor '{executor}' is in --skip-executors list"

    # Skip whole op types on request (--skip-ops option).  Record the layer in
    # the JSON with recommended_executor=None so the compiler leaves it
    # unassigned and the full-model pattern can handle it.
    skipped_ops = _get_skipped_ops(config)
    op_type = _get_op_type_from_layer_id(layer_id)
    if op_type and op_type.lower() in skipped_ops:
        if layer_id not in _skipped_op_layers:
            _skipped_op_layers[layer_id] = {}
            case_meta = layer_executor_case or {}
            _record_skipped_op_in_json(
                config,
                layer_id,
                model_name,
                subgraph_suffix,
                node_index=case_meta.get("node_index"),
                orig_index=case_meta.get("orig_index"),
                full_mlir_location=case_meta.get("full_mlir_location"),
            )
        return f"Layer {layer_id} op type '{op_type}' is in --skip-ops list"

    skip_mode = config.skip_mode
    if not skip_mode or not layer_id or not model_name:
        return None

    # First check in-memory results from current test session (e.g., NSS just passed)
    if layer_id in discovery_state.results:
        for exec_name, exec_result in discovery_state.results[layer_id].items():
            if exec_result.get("status") == "success":
                return f"Layer {layer_id} already works with {exec_name}, skipping {executor}"

    # --recompute-cache invalidates the versioned fixture cache and forces the
    # layer to be compiled/run again. Don't let a stale persisted JSON success
    # short-circuit that, otherwise --debug-ir and similar debug flags never run.
    recompute_cache = config.recompute_cache
    if recompute_cache:
        return None

    # Then check persisted results from JSON file (subgraph-specific or main model)
    json_path = _get_json_path(config, model_name, subgraph_suffix)
    if not json_path or not json_path.exists():
        return None

    existing_data = _load_json(config, model_name, subgraph_suffix)
    if layer_id not in existing_data.get("ops", {}):
        return None

    op_data = existing_data["ops"][layer_id]
    for exec_name, exec_result in op_data.get("executors", {}).items():
        if exec_result.get("status") == "success":
            return f"Layer {layer_id} already works with {exec_name}, skipping {executor}"

    return None


def _copy_result_from_source_layer_core(
    layer_id: str, executor: str, source_layer_id: str, discovery_state: ExecutorDiscoveryState
) -> Optional[str]:
    """Copy executor result from source layer to duplicate layer.

    Returns None after copying; returns the skip reason when the source layer
    has no result for this executor (the caller surfaces it as a skip).
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
        return None

    return (
        f"Layer {layer_id} is a duplicate of {source_layer_id}, "
        f"but {source_layer_id} was not tested on {executor.upper()}"
    )


def _run_layer_test_core(
    config,
    compare_fn: Callable[[], None],
    layer_id: str,
    executor: str,
    node_index: Optional[int],
    mlir_file_path: Optional[Path],
    json_data: Dict,
    case_name: str,
    discovery_state: ExecutorDiscoveryState = _discovery_state,
) -> None:
    """Run one layer comparison and record the outcome.

    ``compare_fn`` performs one model run plus comparison and must raise
    ``AssertionError`` on a tolerance mismatch; see the module docstring for
    what a timing iteration measures.

    Raises ``LayerDifference`` on accuracy failure (recorded as "difference");
    re-raises any other exception (recorded as "error").
    """
    tolerance_used = get_tolerance(layer_id, json_data)
    collect_timing = config.collect_timing
    timing_runs = config.timing_runs

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
                if is_verbose():
                    compare_fn()
                else:
                    # Keep the console quiet without --verbose; per-layer diff
                    # metrics are still recorded from the comparison result.
                    with contextlib.redirect_stdout(io.StringIO()):
                        compare_fn()
            except AssertionError:
                raise
            finally:
                if collect_timing:
                    runtime_times.append((time.perf_counter() - run_start) * 1000)

        timing = build_timing_data(runtime_times) if collect_timing else None

        _record("success", timing=timing)

        op_type = case_name.split("_")[-2]
        loc = _extract_op_type_from_layer(mlir_file_path, op_type) if mlir_file_path else None
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
        raise LayerDifference(f"Layer {layer_id} failed on {executor.upper()} with difference")

    except Exception as e:
        failure_report = _extract_failure_report(str(e), type(e))
        timing = build_timing_data(runtime_times) if collect_timing else None

        _record("error", failure_report=failure_report, timing=timing)
        _discovery_log(f"\nLayer {layer_id}: {executor.upper()} = error")
        raise


# ---------------------------------------------------------------------------
# Case generation cores
# ---------------------------------------------------------------------------


def _resolve_model_files(cfg: DiscoveryConfig) -> List[Path]:
    """Resolve the model files to run discovery on.

    A model path is required: without a source checkout there are no default
    testdata directories to scan.
    """
    if not cfg.model_path:
        raise ValueError(
            "discovery requires --model: no default testdata directories "
            "are available without a source checkout"
        )
    path = Path(cfg.model_path)
    if not path.exists():
        raise ValueError(f"--model does not exist: {cfg.model_path}")
    return [path]


def _precompute_mlir_mappings_core(
    files: List[Path], maps: Dict[str, Dict[str, str]], mlir_cache_dir: Path
) -> Dict[str, Dict[str, str]]:
    """Pre-compute ONNX-to-full-MLIR line-number mappings for each model.

    ``maps`` is the caller-owned per-model stash; ``mlir_cache_dir`` holds
    the cached full-model MLIR files the mappings are built from.
    """
    onnx_to_mlir_map = {}

    for f in files:
        model_name = f.stem
        mlir_cache = Path(mlir_cache_dir) / f"{model_name}.mlir"

        if model_name not in maps:
            maps[model_name] = _build_onnx_to_mlir_mapping(f, mlir_cache)

        onnx_to_mlir_map[model_name] = maps[model_name]

    return onnx_to_mlir_map


def _generate_subgraph_cases_core(
    f: Path, config, maps: Dict[str, Dict[str, str]], mlir_cache_dir: Path
) -> List[Case]:
    """Generate test cases in subgraph mode."""
    subgraph_from = config.subgraph_from
    subgraph_to = config.subgraph_to
    auto_convert_bf16 = config.auto_convert_bf16
    auto_convert_int32 = config.auto_convert_int32
    save_bf16_path = config.save_bf16_model
    quantize = config.quantize
    per_channel = config.per_channel
    full_integer = config.full_integer
    quant_format = config.quant_format
    model_name = f.stem

    if _is_torch_model(f):
        raise ValueError(f"Torch models are not supported in this branch: {f}")

    full_model = get_full_model(str(f))
    full_model = _maybe_apply_bf16_conversion(full_model, f, auto_convert_bf16, save_bf16_path)
    full_model = _maybe_apply_int32_conversion(full_model, f, auto_convert_int32)

    # Build name -> index mapping from the (possibly BF16 but not yet quantized)
    # full model so indices match the original ONNX node positions.
    all_layers = generate_onnx_layers_from_model(full_model, node_groups=None, dedup=False, quantize=quantize)
    name_to_index = {}
    for layer_data in all_layers.values():
        layer_node_index = getattr(layer_data, 'node_index', None)
        if layer_node_index is None:
            continue
        model = _get_model_from_wrapper(layer_data)
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
        raise CaseGenSkip(f"Failed to extract subgraph: {e}")

    subgraph.source_model_path = str(f)

    subgraph_suffix = f"subgraph_{from_index}_{to_index}"
    source_model_path = str(f)

    # Build a subgraph-specific ONNX-to-MLIR mapping so the executor
    # assignment pass matches the subgraph MLIR (not the full-model MLIR).
    subgraph_onnx_path = config.cache.mkdir('subgraph_onnx') / f"{model_name}_{subgraph_suffix}.onnx"
    subgraph_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(subgraph.model, str(subgraph_onnx_path))
    subgraph_mlir_cache = Path(mlir_cache_dir) / f"{model_name}_{subgraph_suffix}.mlir"
    map_key = f"{model_name}_{subgraph_suffix}"

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
    maps[map_key] = subgraph_mapping
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


def _generate_layer_cases_core(
    f: Path, config, maps: Dict[str, Dict[str, str]], mlir_cache_dir: Path
) -> List[Case]:
    """Generate test cases in normal layer-extraction mode."""
    auto_convert_bf16 = config.auto_convert_bf16
    auto_convert_int32 = config.auto_convert_int32
    save_bf16_path = config.save_bf16_model
    quantize = config.quantize
    per_channel = config.per_channel
    full_integer = config.full_integer
    quant_format = config.quant_format

    if _is_torch_model(f):
        raise ValueError(f"Torch models are not supported in this branch: {f}")

    if (auto_convert_bf16 or auto_convert_int32) and not quantize:
        # Non-quantized conversion path: convert the full model (bf16 first,
        # then int32), then extract layers.
        model = get_full_model(str(f))
        model = _maybe_apply_bf16_conversion(model, f, auto_convert_bf16, save_bf16_path)
        model = _maybe_apply_int32_conversion(model, f, auto_convert_int32)
        layers = generate_onnx_layers_from_model(model, node_groups=None, dedup=False, quantize=False)
        return [
            Case(f"{f.stem}_{key}", layer)
            for key, layer in layers.items()
        ] + [Case(f"{f.stem}_full_model", model)]

    if quantize:
        # Quantize path: extract layers from the original full model first, then
        # quantize each layer individually so per-layer tests use their own
        # calibration data and the ONNX-to-MLIR mapping stays tied to the
        # original op positions. BF16/INT32 conversion is mutually exclusive with
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
        quantized_mlir_cache = Path(mlir_cache_dir) / f"{f.stem}.quantized.mlir"
        mapping, node_indices = _build_quantized_mlir_mapping(
            quantized_full, quantized_mlir_cache, list(layers.items())
        )
        maps[f.stem] = mapping

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


def _assemble_layer_test_cases_core(
    cases: List[Case],
    onnx_to_mlir_map: Dict[str, Dict[str, str]],
    skipped_executors: set,
    dedup_layers: bool,
    global_maps: Dict[str, Dict[str, str]],
) -> List[tuple]:
    """Build param tuples for per-layer / per-executor parametrization.

    ``global_maps`` holds the per-model (and per-subgraph) mappings built
    during case generation.
    """
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


def _generate_test_cases(
    cfg: DiscoveryConfig, cache: Cache
) -> Tuple[List[tuple], Dict[str, Dict[str, str]]]:
    """Generate (case, layer_id, executor, ...) tuples."""
    files = _resolve_model_files(cfg)

    _skipped_op_layers.clear()

    skipped_executors = _get_skipped_executors(cfg)
    if skipped_executors:
        _discovery_log(f"[SkipExecutors] Will skip executors: {sorted(skipped_executors)}")

    skipped_ops = _get_skipped_ops(cfg)
    if skipped_ops:
        _discovery_log(f"[SkipOps] Will skip op types: {sorted(skipped_ops)}")

    maps: Dict[str, Dict[str, str]] = {}
    mlir_cache_dir = cache.mkdir("onnx_full_mlir")
    onnx_to_mlir_map = _precompute_mlir_mappings_core(files, maps, mlir_cache_dir)

    cases: List[Case] = []
    for f in files:
        if cfg.subgraph_mode:
            cases += _generate_subgraph_cases_core(f, cfg, maps, mlir_cache_dir)
        else:
            cases += _generate_layer_cases_core(f, cfg, maps, mlir_cache_dir)

    test_cases = _assemble_layer_test_cases_core(
        cases, onnx_to_mlir_map, skipped_executors, cfg.dedup_layers, maps
    )
    return test_cases, onnx_to_mlir_map


# ---------------------------------------------------------------------------
# Gather-indices input range (moved from tests/test_onnx_gen_config.py)
# ---------------------------------------------------------------------------


def _gather_indices_input_range(case):
    """Valid index range for Gather layers with a constant table.

    Discovery feeds random integer inputs drawn from (-40, 40) by default. For
    a Gather whose constant table has fewer rows along the gathered axis (e.g.
    BERT token_type_embeddings with 2 rows) that produces out-of-bounds
    indices, which is undefined behaviour - the compiled host gather reads out
    of bounds and can segfault. When every runtime (non-initializer) input of
    the layer is the indices tensor of a Gather with a constant table,
    restrict the generated indices to [0, table_dim). Returns None when the
    range should not be overridden.
    """
    try:
        model = case.data.model
    except Exception:
        return None

    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}

    # The table of an extracted Gather layer is typically listed both as an
    # initializer and as a graph input; only non-initializer inputs are
    # actually driven with random data.
    runtime_inputs = {inp.name for inp in graph.input if inp.name not in initializers}
    if not runtime_inputs:
        return None

    table_dim = None
    for node in graph.node:
        if node.op_type != "Gather" or len(node.input) < 2:
            continue
        data_name, indices_name = node.input[0], node.input[1]
        # Table must be a constant and indices must be a runtime input.
        if data_name not in initializers or indices_name not in runtime_inputs:
            return None
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
        dims = initializers[data_name].dims
        if not dims:
            return None
        dim = dims[axis % len(dims)]
        table_dim = dim if table_dim is None else min(table_dim, dim)
        runtime_inputs.discard(indices_name)

    if table_dim is None or table_dim <= 0:
        return None

    # Do not clamp if some runtime input is not a gather indices tensor:
    # the single range would apply to that input as well.
    if runtime_inputs:
        return None

    return (0, table_dim)


# ---------------------------------------------------------------------------
# Chip resolution (mirrors testing/iree.py's chip handling)
# ---------------------------------------------------------------------------


def _get_latest_chips() -> List[str]:
    chips = set()
    for configs_dir in [_TOPDIR / 'extras' / 'chips', _TOPDIR / 'tests' / 'testdata' / 'chips']:
        if not configs_dir.exists():
            continue
        chips.update([f.resolve().stem for f in configs_dir.iterdir() if f.stem.endswith('latest')])
    return sorted(chips)


def _load_chip_config(chip_name: str) -> Dict[str, Any]:
    """Port of the ``chip_config`` fixture: merge ``<chip>.json`` if present."""
    config: Dict[str, Any] = {"chip_name": chip_name, "target": chip_name}
    if chip_name == 'default':
        config["target"] = "SL2610"
    for root_dir in [_TOPDIR / 'tests' / 'testdata' / 'chips', _TOPDIR / 'extras' / 'chips']:
        config_file = root_dir / f'{chip_name}.json'
        if not config_file.exists():
            continue
        with open(config_file, 'r') as f:
            config.update(json.load(f))
    return config


def _resolve_chip_configs(torq_hw: str) -> List[Dict[str, Any]]:
    """Resolve a --torq-hw value to a list of chip_config dicts."""
    chips = set()

    if torq_hw == 'all':
        chips.update(_get_latest_chips())
    else:
        for chip in torq_hw.split(","):
            if chip.endswith(".group"):
                found = False
                for root_dir in [_TOPDIR / 'extras' / 'chips', _TOPDIR / 'tests' / 'testdata' / 'chips']:
                    group_file = root_dir / chip
                    if not group_file.exists():
                        continue
                    found = True
                    with open(group_file, 'r') as f:
                        chips.update([x.strip() for x in f.read().splitlines() if x.strip() != ""])
                if not found:
                    if chip == "default.group":
                        # Standalone fallback outside a source checkout:
                        # default.group ships sl2610-v1, whose JSON maps to the
                        # built-in SL2610 target; "default" resolves to the same.
                        chips.add("default")
                    else:
                        raise ValueError(
                            f"Chip group file not found: {chip} (looked under "
                            f"{_TOPDIR}/extras/chips and {_TOPDIR}/tests/testdata/chips)"
                        )
            else:
                chips.add(chip)

    chips = sorted(chips)
    if len(chips) == 0:
        raise ValueError("No chips found for torq tests")

    return [_load_chip_config(c) for c in chips]


def _torq_hw_arg(chip_config: Dict[str, Any]) -> str:
    """Return the value for ``--torq-hw=`` (handles the custom chip string format)."""
    target = chip_config.get("target", "SL2610")
    if target == "custom":
        dma_tp = chip_config.get("dma_theoretical_bytes_per_cycle", 8)
        dma_factor = chip_config.get("dma_factor", 1.0)
        return (
            f'{chip_config["hw_id"]}:{chip_config["lram_size"]}:{chip_config["slice_count"]}:'
            f'{chip_config.get("css_features","")}:{chip_config.get("nss_features","")}:{dma_tp}:{dma_factor}:'
            f'{chip_config.get("host_triple","native")}:{chip_config.get("host_cpu","host")}:{chip_config.get("host_cpu_features","host")}'
        )
    return target


# ---------------------------------------------------------------------------
# Discovery/compiler JSON resolution
# ---------------------------------------------------------------------------


def _find_discovery_json(
    config, model_name: str, subgraph_suffix: Optional[str] = None
) -> Optional[Path]:
    """Find existing executor discovery JSON file for the model.

    For subgraph tests, looks for subgraph-specific JSON file.
    """
    if not model_name:
        return None

    output_dir = config.output_dir
    search_dir = Path(output_dir) if output_dir else Path(".")

    if not search_dir.exists():
        return None

    # For subgraph tests, look for subgraph-specific JSON first
    if subgraph_suffix:
        json_path = search_dir / f"torq_gen_config_{model_name}_{subgraph_suffix}.json"
        if json_path.exists():
            return json_path

    # Look for torq_gen_config_<model_name>.json
    json_path = search_dir / f"torq_gen_config_{model_name}.json"
    if json_path.exists():
        return json_path

    # Search by model_name field inside JSON files (exclude sidecar JSONs)
    for json_file in search_dir.glob("torq_gen_config_*.json"):
        if json_file.name.endswith(("_compiler.json", "_mac_debug.json")):
            continue
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            # A discovery report always carries an "ops" section; this keeps
            # sidecar files from being misidentified as the report.
            if data.get("model_name") == model_name and "ops" in data:
                return json_file
        except Exception:
            continue

    return None


def _find_compiler_json(
    config, model_name: str, subgraph_suffix: Optional[str] = None
) -> Optional[Path]:
    """Find existing compiler-format JSON file for the model.

    Searches for torq_gen_config_<model>_compiler.json files. If not found
    by exact name, falls back to searching by model_name field inside JSON.
    """
    if not model_name:
        return None

    output_dir = config.output_dir
    search_dir = Path(output_dir) if output_dir else Path(".")

    if not search_dir.exists():
        return None

    # For subgraph tests, look for subgraph-specific compiler JSON first
    if subgraph_suffix:
        json_path = search_dir / f"torq_gen_config_{model_name}_{subgraph_suffix}_compiler.json"
        if json_path.exists():
            return json_path

    # Look for torq_gen_config_<model_name>_compiler.json
    json_path = search_dir / f"torq_gen_config_{model_name}_compiler.json"
    if json_path.exists():
        return json_path

    # Search by model_name field inside compiler JSON files
    for json_file in search_dir.glob("torq_gen_config_*_compiler.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            if data.get("model_name") == model_name:
                return json_file
        except Exception:
            continue

    return None


def _update_discovery_json_line_numbers(
    discovery_json_path: Path, mlir_file: Path
) -> None:
    """Update discovery JSON with correct line numbers from full model MLIR.

    Layer tests save mlir_location as op_type (e.g., "Tanh"). This updates
    it to line:column format (e.g., "10:10") from full model MLIR.
    Uses node_index (position) for matching.

    For quantized models, Q/DQ wrapper nodes are ignored so that executor
    assignments target compute ops, not inserted wrappers.
    """
    if not discovery_json_path.exists() or not mlir_file.exists():
        return

    # Get compute operations in order from full model MLIR, skipping Q/DQ wrappers.
    # Normalize qoperator op names (QLinearConv -> Conv, etc.) so they match the
    # original layer op types stored in the discovery JSON.
    all_ops = extract_line_numbers_from_mlir(mlir_file)
    compute_ops = [
        (_normalize_quantized_op_type(op_type), loc) for op_type, loc in all_ops
        if op_type not in _QUANT_WRAPPER_OPS
    ]
    if not compute_ops:
        return

    with open(discovery_json_path, "r") as f:
        data = json.load(f)

    updated = False
    for op_name in sorted(data.get("ops", {}).keys()):
        op_data = data["ops"][op_name]
        expected_op_type = op_name.split("_")[0] if op_name else None

        # Use node_index if available for precise matching
        node_index = op_data.get("_node_index")
        if node_index is not None and 0 <= node_index < len(compute_ops):
            op_type, new_location = compute_ops[node_index]

            # If the stored index points at the wrong op type (stale/fused
            # data), fall back to op_type matching instead.  This is expected
            # for quantized models where Q/DQ wrappers are stripped or where
            # activations like Relu are fused into the preceding compute op.
            if expected_op_type and op_type == expected_op_type:
                old_location = op_data.get("mlir_location", "")
                if old_location != new_location:
                    op_data["mlir_location"] = new_location
                    updated = True
                    logger.info(
                        f"[ExecutorAssignments] Updated {op_name}: {old_location} -> "
                        f"{new_location} (node_index={node_index})"
                    )
                continue

        # Warn if node_index is out of bounds and the expected op type is still
        # present in the MLIR (indicates a stale index). If the op was removed
        # by quantization, fallback silently leaves the op_type string.
        if node_index is not None and node_index >= len(compute_ops):
            if expected_op_type and any(ot == expected_op_type for ot, _ in compute_ops):
                logger.warning(
                    f"[ExecutorAssignments] node_index {node_index} out of bounds for {op_name} "
                    f"(max: {len(compute_ops) - 1}), but {expected_op_type} still exists in MLIR. "
                    f"Using fallback matching by op_type."
                )

        # Fallback: try to match by op_type (first occurrence)
        matched = False
        for ot, loc in compute_ops:
            if ot == expected_op_type:
                old_location = op_data.get("mlir_location", "")
                if old_location != loc:
                    op_data["mlir_location"] = loc
                    updated = True
                    logger.info(f"[ExecutorAssignments] Updated {op_name} (fallback): -> {loc}")
                matched = True
                break

        # Fused or otherwise unmapped layers get their op_type string back so
        # the compiler config excludes them (only line:column keys are valid).
        if not matched and expected_op_type:
            old_location = op_data.get("mlir_location", "")
            if old_location != expected_op_type:
                op_data["mlir_location"] = expected_op_type
                updated = True
                logger.info(f"[ExecutorAssignments] Cleared stale location for {op_name}: -> {expected_op_type}")

    if updated:
        with open(discovery_json_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[ExecutorAssignments] Updated discovery JSON: {discovery_json_path}")


def _write_versioned_json(
    cache: Cache, name: str, data: Dict[str, Any], recompute: bool, extra_version: str = ""
) -> Path:
    """Write *data* as JSON into a content-versioned cache file and return its path."""
    text = json.dumps(data, indent=2)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    vf = get_or_generate_file(
        cache,
        name,
        "json",
        [extra_version, digest],
        lambda p: p.write_text(text),
        recompute=recompute,
    )
    return vf.file_path


def _resolve_executor_assignments(
    cfg: DiscoveryConfig,
    cache: Cache,
    model_name: Optional[str],
    subgraph_suffix: Optional[str],
    mlir_path: Path,
) -> Path:
    """Port of the ``torq_torq_gen_config_json`` fixture.

    Returns the path of a JSON file with ``{"op_assignments": ...}`` for the
    compiler's ``--torq-executor-map``.
    """
    discovery_json = _find_discovery_json(cfg, model_name, subgraph_suffix)

    if discovery_json:
        logger.info(f"[ExecutorAssignments] Using discovery JSON: {discovery_json}")

        # Update discovery JSON with correct line numbers from full model MLIR
        _update_discovery_json_line_numbers(discovery_json, mlir_path)

        # Generate compiler-format JSON from the updated discovery data
        discovery_data = load_config(discovery_json)
        compiler_data = generate_compiler_config(discovery_data, model_name)

        versioned_path = _write_versioned_json(
            cache, "executor_assignments", compiler_data, cfg.recompute_cache, str(mlir_path)
        )

        # Also persist compiler JSON next to discovery JSON for user reference
        output_dir = cfg.output_dir
        compiler_path = get_compiler_config_path(model_name, output_dir, subgraph_suffix)
        save_config(compiler_path, compiler_data)

        logger.info(
            f"[ExecutorAssignments] Compiler JSON "
            f"({len(compiler_data.get('op_assignments', {}))} ops) -> {versioned_path}"
        )
        return versioned_path

    # Option 2: Use compiler JSON directly (no report JSON needed)
    compiler_json = _find_compiler_json(cfg, model_name, subgraph_suffix)
    if compiler_json:
        logger.info(f"[ExecutorAssignments] Using compiler JSON: {compiler_json}")
        with open(compiler_json, "r") as f:
            compiler_data = json.load(f)
        versioned_path = _write_versioned_json(
            cache, "executor_assignments", compiler_data, cfg.recompute_cache, str(mlir_path)
        )
        logger.info(
            f"[ExecutorAssignments] Compiler JSON "
            f"({len(compiler_data.get('op_assignments', {}))} ops) -> {versioned_path}"
        )
        return versioned_path

    # Option 3: Default empty assignments
    logger.warning(
        f"[ExecutorAssignments] No discovery JSON found for model '{model_name}'."
    )
    logger.warning(
        "[ExecutorAssignments] Run discovery first: "
        "python3 -m torq.gen_config discover --model=<model>.onnx"
    )
    return _write_versioned_json(
        cache, "executor_assignments", {"op_assignments": {}}, cfg.recompute_cache, str(mlir_path)
    )


# ---------------------------------------------------------------------------
# Artifact build / compile / run / reference / compare (torq.lab based)
# ---------------------------------------------------------------------------


def _case_model_version(case) -> str:
    """Content-addressed version string for a case's ONNX model."""
    model = _get_model_from_wrapper(case.data)
    digest = hashlib.sha256(model.SerializeToString()).hexdigest()[:16]
    return f"{case.name}.{digest}"


def _case_onnx_and_mlir(cfg: DiscoveryConfig, cache: Cache, case) -> Tuple[Path, Path, str]:
    """Save the case's ONNX model and convert it to MLIR (both content-versioned).

    Mirrors the ``onnx_model_file`` + ``onnx_mlir_model_file`` fixtures.
    Returns (onnx_path, mlir_path, mlir_version).
    """
    model = _get_model_from_wrapper(case.data)

    onnx_vf = get_or_generate_file(
        cache,
        "onnx_model",
        "onnx",
        [_case_model_version(case)],
        lambda p: onnx.save(model, str(p)),
        recompute=cfg.recompute_cache,
    )
    mlir_vf = get_or_generate_file(
        cache,
        "onnx_mlir",
        "mlir",
        [onnx_vf.version],
        lambda p: convert_onnx_to_mlir(
            str(onnx_vf.file_path), str(p), timeout=cfg.compiler_timeout
        ),
        recompute=cfg.recompute_cache,
    )
    return onnx_vf.file_path.resolve(), mlir_vf.file_path.resolve(), mlir_vf.version


def _case_inputs(case, mlir_path: Path):
    """Parse the MLIR IO spec and generate the deterministic random inputs.

    Mirrors ``tweaked_random_input_data``: seed 1234; per-input range from the
    Gather-indices guard when it fires, else the legacy defaults of (0, 80)
    for uint8 and (-40, 40) for everything else.
    """
    from torq.lab.io import generate_random_inputs, get_dtype, parse_mlir_io_spec

    spec = parse_mlir_io_spec(mlir_path)
    gather_range = _gather_indices_input_range(case)
    ranges = {}
    for i, input_spec in enumerate(spec.inputs):
        if gather_range is not None:
            ranges[str(i)] = tuple(gather_range)
        elif get_dtype(input_spec.fmt) == np.uint8:
            ranges[str(i)] = (0, 80)
        else:
            ranges[str(i)] = (-40, 40)
    inputs = generate_random_inputs(spec, seed=None, ranges=ranges)
    return spec, ranges, inputs


def _compute_reference_outputs(
    cfg: DiscoveryConfig, onnx_path: Path, mlir_path: Path, inputs: list, work_dir: Path
) -> list:
    """Port of the ``composite_reference_results`` fixture chain (minus torch)."""
    import onnxruntime

    from torq.lab.reference import (
        execute_onnx_model_numpy,
        has_bf16_einsum,
        has_bf16_matmul,
        llvmcpu_reference_outputs,
    )

    try:
        onnx_model = onnx.load(str(onnx_path))

        # 1. Try ONNXRuntime first (same quirky guard as the legacy fixture)
        if not has_bf16_matmul(onnx_model) or not has_bf16_einsum(onnx_model):
            try:
                ort_session = onnxruntime.InferenceSession(str(onnx_path))
                ort_inputs = {inp.name: inputs[i] for i, inp in enumerate(ort_session.get_inputs())}
                return ort_session.run(None, ort_inputs)
            except Exception:
                pass

        # 2. Try numpy fallback
        try:
            return execute_onnx_model_numpy(onnx_model, inputs)
        except Exception:
            pass
    except Exception:
        pass

    # 3. llvmcpu fallback (IREE reference compilation).  The legacy chain's
    # last-resort torch fallback is intentionally not ported.
    return llvmcpu_reference_outputs(
        Path(mlir_path).resolve(),
        inputs,
        Path(work_dir).resolve(),
        timeout=cfg.compiler_timeout,
    )


def _reference_outputs(
    cfg: DiscoveryConfig,
    cache: Cache,
    onnx_path: Path,
    mlir_path: Path,
    mlir_version: str,
    inputs: list,
    ranges: Optional[Dict[str, Tuple[float, float]]],
) -> list:
    """Reference outputs, cached on disk keyed by MLIR content + input ranges."""
    ranges_key = json.dumps(sorted((ranges or {}).items()))
    vd = get_or_compute_data(
        cache,
        "reference_outputs",
        [mlir_version, ranges_key],
        compute_fn=lambda: _compute_reference_outputs(
            cfg,
            onnx_path,
            mlir_path,
            inputs,
            cache.mkdir("llvmcpu_ref") / hashlib.sha256(mlir_version.encode()).hexdigest()[:16],
        ),
        recompute=cfg.recompute_cache,
    )
    return vd.data


def _extra_compiler_options(cfg: DiscoveryConfig) -> List[str]:
    """User-supplied extra compiler options (``--compiler-option``), after case_config's."""
    cmds: List[str] = []
    cmds.extend(cfg.compiler_options)
    if cfg.trace_buffers:
        cmds.append("--torq-enable-buffer-debug-info")
    return cmds


def _layer_compiler_options(
    cfg: DiscoveryConfig, cache: Cache, layer_id: str, executor: str
) -> List[str]:
    """Per-executor compiler options for layer tests (from the ``case_config`` fixture)."""
    digest = hashlib.sha256(json.dumps([layer_id, executor]).encode()).hexdigest()[:16]
    map_path = cache.mkdir("executor_maps") / f"torq_gen_config_{executor}_{digest}.json"
    assignment = {"op_assignments": {layer_id: {"executor": executor}}}
    with open(map_path, "w") as f:
        json.dump(assignment, f, indent=2)

    options = [f"--torq-executor-map={map_path}"]
    if executor == "nss":
        options.extend(["--torq-disable-css", "--torq-disable-host"])
    elif executor == "css":
        options.extend(["--torq-disable-slices", "--torq-disable-host"])
    elif executor == "host":
        options.extend(["--torq-disable-slices", "--torq-disable-css"])

    return options + _extra_compiler_options(cfg)


def _runtime_options(cfg: DiscoveryConfig) -> List[str]:
    """User-supplied extra runtime options (``--runtime-option``)."""
    cmds: List[str] = []
    cmds.extend(cfg.runtime_options)
    return cmds


def _compile_model(
    cfg: DiscoveryConfig,
    cache: Cache,
    mlir_path: Path,
    mlir_version: str,
    tag: str,
    chip_arg: str,
    compiler_options: List[str],
) -> Path:
    """Compile the MLIR into a versioned directory; return the directory path.

    A phases dump is always emitted.  Honors --debug-ir: the IR tree is
    written under ``<dir>/debug/ir``; a string value additionally copies it
    to the given directory (relative to the CWD).
    """
    from torq.lab.pipeline import ModelPipeline
    from torq.lab.types import PipelineConfig

    versions = [
        mlir_version,
        tag,
        json.dumps(compiler_options),
        chip_arg,
        cfg.runtime_hw_type,
        repr(cfg.debug_ir),
    ]

    def _generate(work_dir: Path) -> None:
        pipe = ModelPipeline(
            PipelineConfig(
                model_path=Path(mlir_path).resolve(),
                work_dir=work_dir,
                chip=chip_arg,
                runtime_hw_type=cfg.runtime_hw_type,
                compiler_options=list(compiler_options),
                dump_ir=bool(cfg.debug_ir),
                dump_phases=True,
                timeout=cfg.compiler_timeout,
            )
        )
        pipe.compile()

    vdir = get_or_generate_directory(
        cache, "torq_compiled_model", versions, _generate, recompute=cfg.recompute_cache
    )

    if isinstance(cfg.debug_ir, str):
        ir_src = vdir.dir_path / "debug" / "ir"
        if ir_src.exists():
            shutil.copytree(str(ir_src), str(Path(cfg.debug_ir)), dirs_exist_ok=True)

    return vdir.dir_path


def _make_compare_fn(
    cfg: DiscoveryConfig,
    mlir_path: Path,
    compile_dir: Path,
    ranges: Optional[Dict[str, Tuple[float, float]]],
    reference: list,
    comparison_config: Dict[str, Any],
) -> Callable[[], None]:
    """One fresh model run + output comparison; raises AssertionError on mismatch."""

    def compare_fn() -> None:
        from torq.lab.pipeline import ModelPipeline
        from torq.lab.types import PipelineConfig

        pipe = ModelPipeline(
            PipelineConfig(
                model_path=Path(mlir_path).resolve(),
                work_dir=compile_dir,
                runtime_hw_type=cfg.runtime_hw_type,
                runtime_options=_runtime_options(cfg),
                random_inputs=True,
                input_ranges=ranges,
                timeout=cfg.runtime_timeout,
                vmfb_path=Path(compile_dir) / "model.vmfb",
            )
        )
        run_result = pipe.run()

        from torq.lab.compare import compare_outputs

        result = compare_outputs(run_result.outputs, reference, config=comparison_config)

        # Exact per-tensor printout of testing/comparison.compare_results;
        # the full-model metrics parsing below relies on these lines.
        for tensor in result.tensors:
            if tensor.max_rel_diff is not None:
                print(f"Max relative difference: {tensor.max_rel_diff}")
            print(f"Max absolute difference: {tensor.max_abs_diff}")
            pct = (tensor.num_diffs / tensor.size * 100) if tensor.size else 0.0
            print(f"Number of differences: {tensor.num_diffs} out of {tensor.size} [{pct:.2f}%]")

        if not result.passed:
            raise AssertionError(result.reason)

    return compare_fn


def _comparison_config_dict(tolerance: Dict[str, float]) -> Dict[str, Any]:
    """The ``comparison_config_for_executor_discovery`` fixture value."""
    return {
        "int_tol": 1,
        "int_thld": 1,
        "fp_avg_tol": tolerance.get("fp_avg_tol", 0.01),
        "fp_max_tol": tolerance.get("fp_max_tol", 0.01),
        "epsilon": 1e-6,
        "allow_all_zero": False,
        "skip_nan_check": False,
    }


# ---------------------------------------------------------------------------
# Runner state helpers
# ---------------------------------------------------------------------------


def _reset_state(state: ExecutorDiscoveryState) -> None:
    """Clear all per-model state (a superset of the ``save_progress`` fixture's reset).

    Also clears ``orig_indices``, ``full_model_metrics`` and ``subgraph_suffix``
    so back-to-back runs for different models in one process stay isolated.
    """
    state.results.clear()
    state.locations.clear()
    state.node_indices.clear()
    state.orig_indices.clear()
    state.mlir_files.clear()
    state.full_mlir_locations.clear()
    state.recommended_executors.clear()
    state.mac_counts.clear()
    state.mac_details.clear()
    state.full_model_metrics = None
    state.subgraph_suffix = None


def _load_existing_json_once(
    cfg: DiscoveryConfig,
    state: ExecutorDiscoveryState,
    loaded_keys: set,
    model_name: Optional[str],
    subgraph_suffix: Optional[str],
) -> None:
    """One-time merge of persisted JSON results (``executor_discovery._loaded_existing_json``)."""
    json_key = f"{model_name}_{subgraph_suffix}" if subgraph_suffix else model_name
    if json_key in loaded_keys:
        return
    loaded_keys.add(json_key)
    if model_name:
        json_data = _load_json(cfg, model_name, subgraph_suffix)
        if json_data:
            state.load_from_json(json_data)
            if subgraph_suffix:
                state.subgraph_suffix = subgraph_suffix


def _finalize_report(cfg: DiscoveryConfig, state: ExecutorDiscoveryState) -> None:
    """Merge persisted JSON, save + print the report."""
    try:
        model_name = Path(cfg.model_path).stem if cfg.model_path else "unknown"
        subgraph_suffix = state.subgraph_suffix

        if cfg.model_path:
            json_data = _load_json(cfg, model_name, subgraph_suffix)
            if json_data:
                state.load_from_json(json_data)
                if subgraph_suffix:
                    state.subgraph_suffix = subgraph_suffix

        if not (state.results or state.full_model_metrics):
            return

        _save_detailed_report(cfg, model_name, state, subgraph_suffix)
    except Exception as e:
        _discovery_log(f"\nWarning: failed to finalize discovery report: {e}")
        return

    try:
        _print_final_report(cfg, state)
    except Exception as e:
        _discovery_log(f"\nWarning: failed to print final report: {e}")


class _Tee:
    """Minimal tee stream: writes go to every wrapped stream."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return self._streams[0].isatty()


_log_detail_stream = None


@contextlib.contextmanager
def _maybe_tee_log(cfg: DiscoveryConfig):
    """Tee stdout/stderr into ``cfg.log_file``; full error details go to the file.

    The console keeps the concise per-layer lines; multi-line diagnostics
    (e.g. compiler stderr embedded in tool errors) are routed to the log
    file via :func:`_log_error_details` instead of being printed whole.
    """
    global _log_detail_stream
    if not cfg.log_file:
        yield
        return
    log_path = Path(cfg.log_file)
    if str(log_path.parent) not in ("", "."):
        log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        _log_detail_stream = log_file
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_stdout, log_file)
        sys.stderr = _Tee(old_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            _log_detail_stream = None


def _summarize_exception(e: BaseException) -> str:
    """First line of an exception's message, for concise console output."""
    message = str(e)
    return message.splitlines()[0] if message else "(no message)"


def _log_error_details(header: str, e: BaseException) -> None:
    """Route full multi-line error diagnostics away from the normal console path.

    Written to the active log file when one is configured; otherwise only
    shown on stderr with --verbose.
    """
    details = f"{header}\n{type(e).__name__}: {e}"
    if _log_detail_stream is not None:
        _log_detail_stream.write(details + "\n")
        _log_detail_stream.flush()
    elif is_verbose():
        print(details, file=sys.stderr)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _run_one_layer_case(
    cfg: DiscoveryConfig,
    cache: Cache,
    chip_arg: str,
    test_case: tuple,
    state: ExecutorDiscoveryState,
    loaded_keys: set,
) -> int:
    """Execute one (layer, executor) case; returns 0 on success/skip, 1 on failure."""
    (case, layer_id, executor, node_index, full_mlir_location, is_subgraph,
     source_layer_id, orig_index) = test_case

    model_name = _extract_model_name_from_case(case)
    subgraph_suffix = _get_subgraph_suffix(case) if is_subgraph else None
    label = f"{case.name}_{executor}"

    _load_existing_json_once(cfg, state, loaded_keys, model_name, subgraph_suffix)

    # Early skip check (mirrors the case_config fixture's skip-before-compile).
    case_meta = {
        "node_index": node_index,
        "orig_index": orig_index,
        "full_mlir_location": full_mlir_location,
    }
    reason = _check_skip_executor(
        cfg, layer_id, executor, model_name, subgraph_suffix, state, case_meta
    )
    if reason:
        _discovery_vlog(f"{label} skipped ({reason})")
        return 0

    # Duplicate layer: copy the source layer's result and skip all expensive
    # work (mirrors the early dedup in the onnx_layer_model fixture).
    if source_layer_id:
        reason = _copy_result_from_source_layer_core(layer_id, executor, source_layer_id, state)
        if reason:
            _discovery_vlog(f"{label} skipped ({reason})")
            return 0
        state.record_metadata(
            layer_id,
            node_index=node_index,
            orig_index=orig_index,
            full_mlir_location=full_mlir_location,
        )
        _discovery_vlog(f"{label} skipped (duplicate of {source_layer_id}, result copied)")
        return 0

    # Build the artifacts (ONNX file, MLIR, inputs, reference, compile).  A
    # failure here is a setup error.
    try:
        onnx_path, mlir_path, mlir_version = _case_onnx_and_mlir(cfg, cache, case)
        spec, ranges, inputs = _case_inputs(case, mlir_path)
        reference = _reference_outputs(
            cfg, cache, onnx_path, mlir_path, mlir_version, inputs, ranges
        )
        compiler_options = _layer_compiler_options(cfg, cache, layer_id, executor)
        compile_dir = _compile_model(
            cfg, cache, mlir_path, mlir_version, executor, chip_arg, compiler_options
        )
    except Exception as e:
        state.record_result(
            layer_id,
            executor,
            "error",
            DEFAULT_TOLERANCE.copy(),
            failure_report={
                "type": "error",
                "summary": f"Fixture/setup failed: {str(e)[:200]}",
            },
        )
        _discovery_log(
            f"\nLayer {layer_id}: {executor.upper()} = error "
            f"(setup: {type(e).__name__}: {_summarize_exception(e)})"
        )
        _log_error_details(f"[{label}] setup error details:", e)
        return 1

    # Record metadata before running (mirrors executor_discovery; duplicates
    # already returned above).
    mac_info = _compute_layer_mac_info(case)
    state.record_metadata(
        layer_id=layer_id,
        node_index=node_index,
        orig_index=orig_index,
        full_mlir_location=full_mlir_location,
        mlir_file=Path(mlir_path),
        mac_count=mac_info["mac_count"] if mac_info else None,
        mac_details=mac_info,
    )

    json_data = _load_json(cfg, model_name, subgraph_suffix) if model_name else {}
    tolerance = get_tolerance(layer_id, json_data)
    compare_fn = _make_compare_fn(
        cfg, mlir_path, compile_dir, ranges, reference, _comparison_config_dict(tolerance)
    )

    try:
        _run_layer_test_core(
            cfg, compare_fn, layer_id, executor, node_index,
            Path(mlir_path), json_data, case.name, state,
        )
        return 0
    except LayerDifference:
        return 1
    except Exception as e:
        _discovery_log(f"  ({type(e).__name__}: {_summarize_exception(e)})")
        _log_error_details(f"[{label}] run error details:", e)
        return 1


def run_discovery(cfg: DiscoveryConfig) -> int:
    """Run per-layer executor discovery for ``cfg.model_path``.

    Returns a process exit code: 0 when every executed (layer, executor) case
    passed or was skipped, 1 on any difference/error.
    """
    set_verbose(cfg.verbose)
    state = _discovery_state
    _reset_state(state)
    cache = Cache(cfg.cache_dir)

    with _maybe_tee_log(cfg):
        try:
            test_cases, _ = _generate_test_cases(cfg, cache)
            chip_configs = _resolve_chip_configs(cfg.torq_hw)
        except CaseGenSkip as e:
            _discovery_log(f"Case generation skipped: {e}")
            return 0
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        layer_cases = [t for t in test_cases if t[1] is not None]
        total = len(layer_cases) * len(chip_configs)
        failures = 0
        completed = 0
        loaded_keys: set = set()

        for chip_config in chip_configs:
            chip_arg = _torq_hw_arg(chip_config)
            _discovery_log(
                f"[Chip] chip={chip_config.get('chip_name')} target={chip_config.get('target')}"
            )
            for test_case in layer_cases:
                failures += _run_one_layer_case(
                    cfg, cache, chip_arg, test_case, state, loaded_keys
                )
                completed += 1
                pct = int(100 * completed / total) if total else 100
                _discovery_log(f"[progress] {completed}/{total} [{pct:3d}%]")
                # Persist after each case (mirrors the save_progress fixture).
                case, _, _, _, _, is_subgraph, _, _ = test_case
                _save_discovery_results(cfg, case, None, is_subgraph, state)

        _finalize_report(cfg, state)

    if failures:
        _discovery_log(f"\nDiscovery finished with {failures} failing case(s)")
        return 1
    return 0


def _run_one_full_case(
    cfg: DiscoveryConfig,
    cache: Cache,
    chip_arg: str,
    test_case: tuple,
    state: ExecutorDiscoveryState,
    loaded_keys: set,
) -> int:
    """Run the full model (or full subgraph) with discovered executor assignments."""
    case, _, _, _, _, is_subgraph, _, _ = test_case
    model_name = _extract_model_name_from_case(case)
    subgraph_suffix = _get_subgraph_suffix(case) if is_subgraph else None

    _load_existing_json_once(cfg, state, loaded_keys, model_name, subgraph_suffix)

    try:
        onnx_path, mlir_path, mlir_version = _case_onnx_and_mlir(cfg, cache, case)
        spec, ranges, inputs = _case_inputs(case, mlir_path)
        reference = _reference_outputs(
            cfg, cache, onnx_path, mlir_path, mlir_version, inputs, ranges
        )
        assignments_path = _resolve_executor_assignments(
            cfg, cache, model_name, subgraph_suffix, mlir_path
        )
        # The executor map goes last, after the extra options (mirrors the
        # torq_compiler_options fixture's append order).
        compiler_options = _extra_compiler_options(cfg) + [
            f"--torq-executor-map={assignments_path}"
        ]
        compile_dir = _compile_model(
            cfg, cache, mlir_path, mlir_version, "discovered", chip_arg, compiler_options
        )
    except Exception as e:
        _discovery_log(
            f"\nFull model setup failed: {type(e).__name__}: {_summarize_exception(e)}"
        )
        _log_error_details(f"[full-model {model_name}] setup error details:", e)
        return 1

    json_data = _load_json(cfg, model_name, subgraph_suffix) if model_name else {}
    tolerance = get_tolerance(None, json_data)
    compare_fn = _make_compare_fn(
        cfg, mlir_path, compile_dir, ranges, reference, _comparison_config_dict(tolerance)
    )

    # Full model / full subgraph test: capture the comparison printout and
    # store the metrics for the final report (mirrors executor_discovery).
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture):
            compare_fn()
    finally:
        captured = capture.getvalue()
        if captured and is_verbose():
            print(captured, end="")
        metrics = {}
        for line in captured.splitlines():
            if line.startswith("Max relative difference:"):
                metrics["max_rel_diff"] = line.split(":", 1)[1].strip()
            elif line.startswith("Max absolute difference:"):
                metrics["max_abs_diff"] = line.split(":", 1)[1].strip()
            elif line.startswith("Number of differences:"):
                metrics["num_differences"] = line.split(":", 1)[1].strip()
        state.full_model_metrics = metrics

    _discovery_log(f"\nFull model {model_name}: comparison passed")
    return 0


def run_full_model(cfg: DiscoveryConfig) -> int:
    """Run the full model with the discovered executor assignments.

    Returns 0 when the full-model comparison passed, 1 otherwise.
    """
    set_verbose(cfg.verbose)
    state = _discovery_state
    _reset_state(state)
    cache = Cache(cfg.cache_dir)

    with _maybe_tee_log(cfg):
        try:
            test_cases, _ = _generate_test_cases(cfg, cache)
            chip_configs = _resolve_chip_configs(cfg.torq_hw)
        except CaseGenSkip as e:
            _discovery_log(f"Case generation skipped: {e}")
            return 0
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        full_cases = [t for t in test_cases if t[1] is None]
        if not full_cases:
            print("Error: no full-model case generated for this model", file=sys.stderr)
            return 1

        failures = 0
        loaded_keys: set = set()
        for chip_config in chip_configs:
            chip_arg = _torq_hw_arg(chip_config)
            for test_case in full_cases:
                try:
                    failures += _run_one_full_case(
                        cfg, cache, chip_arg, test_case, state, loaded_keys
                    )
                except AssertionError as e:
                    _discovery_log(f"\nFull model comparison failed: {e}")
                    failures += 1
                except Exception as e:
                    _discovery_log(
                        f"\nFull model run failed: {type(e).__name__}: {_summarize_exception(e)}"
                    )
                    _log_error_details("[full-model] run error details:", e)
                    failures += 1

        _finalize_report(cfg, state)

    return 1 if failures else 0
