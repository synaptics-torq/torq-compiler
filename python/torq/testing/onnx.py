"""
Fixtures and utilities for testing ONNX models.
"""

import json
from dataclasses import dataclass
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
import pytest
from onnx import helper, numpy_helper, shape_inference, TensorProto

from torq.testing.cases import Case
from torq.testing.hf import get_hf_model_file

from .versioned_fixtures import (
    versioned_cached_data_fixture,
    versioned_generated_file_fixture,
    versioned_hashable_object_fixture,
    versioned_static_file_fixture,
    versioned_unhashable_object_fixture,
    VersionedUncachedData,
)


class ModelWithMetadata:
    """Wrapper for ONNX model with metadata like node_index.

    ONNX protobuf objects don't allow setting arbitrary attributes,
    so we use this wrapper to store additional metadata.
    """

    def __init__(self, model: Any, node_index: int = None):
        self.model = model
        self.node_index = node_index

    def __getattr__(self, name):
        # Delegate attribute access to wrapped model
        return getattr(self.model, name)

    def __getitem__(self, key):
        # Support dict-style access (e.g. layer["model"]) for backward compatibility
        return getattr(self, key)


@dataclass
class OnnxLayerCase(Case):
    """Represents a generated ONNX layer test case plus source node name."""

    node_name: str = ""
    is_full_model: bool = False


def pytest_addoption(parser):
    parser.addoption(
        "--onnx-print-layer-info",
        action="store_true",
        default=False,
        help="Print original ONNX node names for generated ONNX layer tests during setup",
    )


def _source_node_name(node, node_index):
    if node.name:
        return node.name

    outputs = [output_name for output_name in node.output if output_name]
    if outputs:
        return outputs[0]

    return f"{node.op_type}@{node_index}"


def _format_layer_node_name(case: OnnxLayerCase) -> str:
    if not case.node_name:
        return "ONNX node: <unknown>"

    label = "nodes" if " -> " in case.node_name else "node"
    return f"ONNX {label}: {case.node_name}"


def _vi_dtype_shape(vi):
    try:
        tt = vi.type.tensor_type
    except Exception:
        return None, None

    elem = getattr(tt, 'elem_type', None)

    shape = None
    try:
        dims = []
        for d in getattr(tt, 'shape').dim:
            val = getattr(d, 'dim_value', None)
            if val is None:
                val = getattr(d, 'dim_param', None)
            dims.append(val)
        shape = tuple(dims)
    except Exception:
        shape = None

    return int(elem) if elem is not None else None, shape


def _inputs_outputs_signature(graph):
    inputs = []
    for vi in graph.input:
        inputs.append(_vi_dtype_shape(vi))

    outputs = []
    for vi in graph.output:
        outputs.append(_vi_dtype_shape(vi))

    return inputs, outputs


def _init_signature(init):
    try:
        arr = numpy_helper.to_array(init)
        dtype = str(arr.dtype)
        shape = tuple(int(d) for d in arr.shape)
        return [dtype, shape]
    except Exception:
        return [init.data_type, tuple(getattr(init, 'dims', ()))]

def model_signature(model):
    graph = getattr(model, 'graph', model)
    inputs, outputs = _inputs_outputs_signature(graph)
    inits = [_init_signature(init) for init in graph.initializer]

    # signature_list = [inputs, outputs, len(node), node_op_type_list, initializer]
    return [inputs, outputs, len(list(graph.node)), [n.op_type for n in graph.node], inits]


def _build_model_from_node_subset(nodes_subset, all_nodes, model, graph_name,
                                   init_name_fn, log_prefix=""):
    """Build a standalone ONNX model from a subset of nodes.

    Shared logic between layer extraction and subgraph extraction.

    Args:
        nodes_subset: List of nodes to include in the new model.
        all_nodes: Full list of nodes from the original model (for determining
                   which outputs are consumed outside the subset).
        model: Original ONNX model (ModelProto or ModelWithMetadata).
        graph_name: Name for the new graph.
        init_name_fn: Callable(base_name, idx_init) -> str for renaming
                      initializers to avoid collisions.
        log_prefix: Prefix string for error messages.

    Returns:
        The inferred/checked ONNX model, or a deep copy of the raw model if
        inference fails.
    """
    import copy as _copy

    # Unwrap ModelWithMetadata if needed
    if hasattr(model, 'model'):
        model = model.model

    graph = model.graph

    # Determine tensors produced and used by the subset
    produced = set()
    used = set()
    for n in nodes_subset:
        for o in n.output:
            produced.add(o)
        for inp in n.input:
            if inp:
                used.add(inp)

    external_inputs = used - produced

    # External outputs: produced names consumed by nodes outside the subset
    # or that are original model outputs
    external_outputs = set()
    orig_output_names = {o.name for o in graph.output}
    for name in produced:
        consumed_outside = False
        for other in all_nodes:
            if other in nodes_subset:
                continue
            if name in other.input:
                consumed_outside = True
                break
        if consumed_outside or (name in orig_output_names):
            external_outputs.add(name)

    # Build metadata maps
    orig_inputs = {vi.name: vi for vi in graph.input}
    orig_value_info = {vi.name: vi for vi in graph.value_info}
    orig_initializers = {init.name: init for init in graph.initializer}

    # Helper to synthesize a simple value_info from an initializer
    def _value_info_from_initializer(init):
        try:
            arr = numpy_helper.to_array(init)
            shape = list(arr.shape)
            return helper.make_tensor_value_info(init.name, init.data_type, shape)
        except Exception:
            return helper.make_tensor_value_info(init.name, TensorProto.FLOAT, [])

    # Build new inputs/outputs/initializers/value_info
    new_inputs = []
    new_outputs = []
    new_initializers = []
    added_initializer_names = set()

    for name in sorted(external_inputs):
        if name in orig_inputs:
            new_inputs.append(_copy.deepcopy(orig_inputs[name]))
        elif name in orig_value_info:
            new_inputs.append(_copy.deepcopy(orig_value_info[name]))
        elif name in orig_initializers:
            init = orig_initializers[name]
            new_initializers.append(_copy.deepcopy(init))
            added_initializer_names.add(name)
            new_inputs.append(_value_info_from_initializer(init))
        else:
            new_inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, []))

    for name in sorted((external_inputs | used)):
        if name in orig_initializers and name not in added_initializer_names:
            new_initializers.append(_copy.deepcopy(orig_initializers[name]))

    for name in sorted(external_outputs):
        matched = False
        for out in graph.output:
            if out.name == name:
                new_outputs.append(_copy.deepcopy(out))
                matched = True
                break
        if not matched and name in orig_value_info:
            new_outputs.append(_copy.deepcopy(orig_value_info[name]))
        elif not matched and name in orig_initializers:
            new_outputs.append(_value_info_from_initializer(orig_initializers[name]))
        elif not matched:
            new_outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, []))

    new_value_info = []
    for name in sorted(produced | used):
        if name in orig_value_info:
            new_value_info.append(_copy.deepcopy(orig_value_info[name]))

    # Deep-copy nodes
    nodes_copy = [_copy.deepcopy(n) for n in nodes_subset]

    # Ensure initializer names are unique
    name_map = {}
    existing_names = set()
    for n in nodes_copy:
        existing_names.update([s for s in n.input if s])
        existing_names.update([s for s in n.output if s])
    existing_names.update([vi.name for vi in new_inputs])
    existing_names.update([vi.name for vi in new_outputs])
    existing_names.update([vi.name for vi in new_value_info])

    for idx_init, init in enumerate(new_initializers):
        base = init.name
        candidate = init_name_fn(base, idx_init)
        i = 0
        while candidate in existing_names:
            i += 1
            candidate = f"{candidate}_{i}"
        name_map[base] = candidate
        existing_names.add(candidate)
        init.name = candidate

    if name_map:
        for n in nodes_copy:
            for ii, v in enumerate(list(n.input)):
                if v in name_map:
                    n.input[ii] = name_map[v]
        for vi in new_inputs:
            if vi.name in name_map:
                vi.name = name_map[vi.name]
        for vi in new_outputs:
            if vi.name in name_map:
                vi.name = name_map[vi.name]
        for vi in new_value_info:
            if vi.name in name_map:
                vi.name = name_map[vi.name]

    # Create new graph and model
    new_graph = helper.make_graph(
        nodes_copy,
        graph_name,
        inputs=new_inputs,
        outputs=new_outputs,
        initializer=new_initializers
    )
    if new_value_info:
        new_graph.value_info.extend(new_value_info)

    new_model = helper.make_model(new_graph)
    new_model.ir_version = model.ir_version
    new_model.opset_import.extend(model.opset_import)

    try:
        inferred = shape_inference.infer_shapes(new_model)
        onnx.checker.check_model(inferred)
        return inferred
    except Exception as e:
        if log_prefix:
            print(f'{log_prefix}: inference/check failed: {e}; using raw model')
        return _copy.deepcopy(new_model)


def generate_onnx_layers_from_model(model, node_groups=None, dedup=True):

    existing_cases = set()
    layer_configs = {}

    graph = model.graph
    nodes = graph.node
    node_count = len(nodes)

    def match_group(start_index):
        if node_groups is None:
            return 0
        for group in node_groups:
            if start_index + len(group) <= node_count:
                if all(nodes[start_index + i].op_type == group[i] for i in range(len(group))):
                    return len(group)
        return 0

    index = 0
    part_num = 0
    # Track the original node index for matching with full model MLIR.
    # This counts only non-Constant nodes since those are what torch-mlir converts.
    original_node_index = 0
    while index < node_count:

        if (nodes[index].op_type.lower() == 'constant'):
            index += 1
            continue

        # Record the original index BEFORE incrementing for this layer
        start_node_index = original_node_index

        group_size = match_group(index)
        if group_size > 0:
            part_nodes = nodes[index:index + group_size]
            part_node_indices = list(range(index, index + group_size))
            index += group_size
            original_node_index += group_size
        else:
            part_nodes = [nodes[index]]
            part_node_indices = [index]
            index += 1
            original_node_index += 1

        source_node_names = [
            _source_node_name(node, node_index)
            for node_index, node in zip(part_node_indices, part_nodes)
        ]
        node_name = " -> ".join(source_node_names)

        layer_name = f"layer_{part_nodes[0].op_type}_{part_num}"
        part_num += 1

        final_model = _build_model_from_node_subset(
            part_nodes,
            nodes,
            model,
            graph_name=f'part{part_num - 1}_graph',
            init_name_fn=lambda base, idx, pn=part_num - 1: f"{base}_part{pn}_init{idx}",
            log_prefix=f'layer {layer_name}',
        )

        # check duplication: too heavy to compare model, just compare signature
        if dedup:
            m_signature_json = json.dumps(model_signature(final_model))
            found_existing = False
            for c in existing_cases:
                if c == m_signature_json:
                    found_existing = True
                    break

            if not found_existing:
                existing_cases.add(m_signature_json)
            else:
                continue

        # Wrap model with metadata (node_index for matching with full model MLIR)
        layer_configs[layer_name] = ModelWithMetadata(final_model, start_node_index)

    return layer_configs


def extract_onnx_subgraph(model, from_index, to_index):
    """
    Extract a subgraph from an ONNX model given start and end node indices.

    The indices refer to non-Constant nodes in the model graph (same indexing
    as used in executor discovery JSON's _node_index field).

    Args:
        model: ONNX model (ModelProto or ModelWithMetadata)
        from_index: Start node index (inclusive, refers to non-Constant nodes)
        to_index: End node index (inclusive, refers to non-Constant nodes)

    Returns:
        ModelWithMetadata containing the extracted subgraph
    """
    import copy as _copy

    # Unwrap ModelWithMetadata if needed
    if hasattr(model, 'model'):
        model = model.model

    graph = model.graph
    nodes = list(graph.node)

    # Filter out Constant nodes to get the same indexing as layer extraction
    non_constant_nodes = []
    non_constant_indices = []  # Maps from filtered index -> original index
    for i, node in enumerate(nodes):
        if node.op_type.lower() != 'constant':
            non_constant_nodes.append(node)
            non_constant_indices.append(i)

    total_non_constant = len(non_constant_nodes)

    # Validate indices
    if from_index < 0 or from_index >= total_non_constant:
        raise ValueError(f"from_index {from_index} out of range [0, {total_non_constant})")
    if to_index < 0 or to_index >= total_non_constant:
        raise ValueError(f"to_index {to_index} out of range [0, {total_non_constant})")
    if from_index > to_index:
        raise ValueError(f"from_index {from_index} must be <= to_index {to_index}")

    # Map to original node indices
    original_from_idx = non_constant_indices[from_index]
    original_to_idx = non_constant_indices[to_index]

    # Get all nodes in range (including any Constant nodes between them)
    subgraph_nodes = nodes[original_from_idx:original_to_idx + 1]

    final_model = _build_model_from_node_subset(
        subgraph_nodes,
        nodes,
        model,
        graph_name=f"subgraph_{from_index}_to_{to_index}",
        init_name_fn=lambda base, idx: f"{base}_subgraph_init{idx}",
        log_prefix="[Subgraph]",
    )

    return ModelWithMetadata(final_model, from_index)


def _build_onnx_layer_cases(model_prefix: str, model, layers):
    cases = [
        OnnxLayerCase(
            name=f"{model_prefix}_{key}",
            data=layer,
        )
        for key, layer in layers.items()
    ]
    cases.append(OnnxLayerCase(name=f"{model_prefix}_full_model", data=model, is_full_model=True))
    return cases


def generate_onnx_layers_from_hf(cache, repo_id, filename, node_groups=None, dedup=True):
    model = get_full_model(get_hf_model_file(cache, repo_id, filename))
    layers = generate_onnx_layers_from_model(model, node_groups, dedup)
    return _build_onnx_layer_cases(Path(filename).stem, model, layers)


def generate_onnx_layers_from_file(filepath:Path, node_groups=None, dedup=False):
    model = get_full_model(str(filepath))
    layers = generate_onnx_layers_from_model(model, node_groups, dedup)
    return _build_onnx_layer_cases(filepath.stem, model, layers)


@pytest.fixture
def onnx_model(request, case_config):
    return request.getfixturevalue(case_config['onnx_model'])


@versioned_generated_file_fixture("onnx")
def onnx_model_file(request, versioned_file, onnx_model):
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, versioned_file)


def is_model_bf16(model: onnx.ModelProto) -> bool:
    """Check if model is already in BF16 format."""
    return any(
        init.data_type == TensorProto.BFLOAT16
        for init in model.graph.initializer
    )


@versioned_hashable_object_fixture
def onnx_bf16_config(request):
    """Return BF16 conversion config for version hashing.

    This ensures cache invalidation when --auto-convert-bf16 flag changes.
    """
    return {
        "auto_convert_bf16": request.config.getoption("--auto-convert-bf16", default=False)
    }


@versioned_generated_file_fixture("onnx_bf16")
def onnx_bf16_model_file(request, versioned_file, onnx_model_file, onnx_bf16_config):
    """Convert FP32 ONNX to BF16 if --auto-convert-bf16 is enabled.

    Always saves to versioned_file location (the decorator always returns versioned_file).

    Note: onnx_model_file is a Path (versioned_generated_file_fixture unwraps VersionedFile).
    """
    import shutil
    use_bf16 = request.config.getoption("--auto-convert-bf16", default=False)

    if not use_bf16:
        # No conversion - copy original model to versioned location
        print(f"[BF16] Conversion disabled, copying original model to {versioned_file}")
        shutil.copy(str(onnx_model_file), str(versioned_file))
        return versioned_file

    # Load the model
    model = onnx.load(str(onnx_model_file))

    # Check if already BF16
    if is_model_bf16(model):
        print(f"[BF16] Model already in BF16 format, copying to {versioned_file}")
        shutil.copy(str(onnx_model_file), str(versioned_file))
        return versioned_file

    # Convert to BF16
    print(f"[BF16] Converting {onnx_model_file.name} to BF16...")
    converted_model = convert_fp32_to_bf16(model)

    # Save converted model
    onnx.save(converted_model, str(versioned_file))
    print(f"[BF16] Saved to: {versioned_file}")

    return versioned_file


@versioned_generated_file_fixture("mlir")
def onnx_mlir_model_file(request, versioned_file, onnx_model_file, onnx_bf16_model_file, onnx_bf16_config):
    """Convert ONNX model to MLIR.

    Uses BF16 model if --auto-convert-bf16 is enabled, otherwise uses original model.
    This ensures the compiler receives the correctly converted model based on user options.

    Note: Both onnx_model_file and onnx_bf16_model_file are Path objects
    (versioned_generated_file_fixture unwraps VersionedFile to Path).
    """
    use_bf16 = request.config.getoption("--auto-convert-bf16", default=False)
    model_path = onnx_bf16_model_file if use_bf16 else onnx_model_file

    if use_bf16:
        print(f"[BF16] Using BF16 model for MLIR conversion: {model_path}")

    subprocess.check_output(
        [sys.executable, "-m", "iree.compiler.tools.import_onnx",
          str(model_path), "-o", str(versioned_file), "--data-prop"])


@versioned_hashable_object_fixture
def onnx_params(case_config):
    return {"opset": case_config.get("opset", 20)}


@versioned_cached_data_fixture
def onnx_reference_results(request, onnx_model_file):
    onnx_model = onnx.load(str(onnx_model_file))
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(str(onnx_model_file))
    ort_inputs = {ort_session.get_inputs()[0].name: sample_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs


def get_full_model(model_file):

    model = onnx.load(model_file)
    # Run shape inference on the original model to get value_info with shapes
    inferred_model = None
    try:
        inferred_model = shape_inference.infer_shapes(model)
    except Exception as e:
        inferred_model = model

    return inferred_model


def _float32_to_bfloat16(arr: np.ndarray) -> np.ndarray:
    """Convert float32 numpy array to bfloat16 (stored as uint16)."""
    arr_uint32 = arr.view(np.uint32)
    arr_bf16 = (arr_uint32 >> 16).astype(np.uint16)
    return arr_bf16


def _fix_batch_dimension_to_one(model: onnx.ModelProto) -> int:
    """Fix dynamic batch dimensions (?, -1) to 1 for all inputs."""
    modified_count = 0
    for input_tensor in model.graph.input:
        tensor_type = input_tensor.type.tensor_type
        if tensor_type.HasField('shape') and len(tensor_type.shape.dim) > 0:
            first_dim = tensor_type.shape.dim[0]
            if first_dim.HasField('dim_param'):
                first_dim.ClearField('dim_param')
                first_dim.dim_value = 1
                modified_count += 1
            elif not first_dim.HasField('dim_value'):
                first_dim.dim_value = 1
                modified_count += 1
    return modified_count


def convert_fp32_to_bf16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert FP32 ONNX model to BF16 format.

    Returns the converted model and prints accuracy metrics.
    """
    import copy
    model = copy.deepcopy(model)  # Don't modify original

    # Fix batch dimension
    batch_fixed = _fix_batch_dimension_to_one(model)
    if batch_fixed > 0:
        print(f"[BF16] Fixed {batch_fixed} input(s) to have batch=1")

    total_count = 0
    max_error = 0.0

    # Convert initializers (weights)
    for init in model.graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue

        # Get FP32 data
        if init.raw_data:
            fp32_data = np.frombuffer(init.raw_data, dtype=np.float32).copy()
        elif init.float_data:
            fp32_data = np.array(init.float_data, dtype=np.float32)
        else:
            continue

        # Convert to BF16
        bf16_data = _float32_to_bfloat16(fp32_data)

        # Track error
        total_count += len(fp32_data)
        fp32_back = (bf16_data.astype(np.uint32) << 16).view(np.float32)
        max_error = max(max_error, np.max(np.abs(fp32_data - fp32_back)))

        # Replace data
        init.raw_data = bf16_data.tobytes()
        init.float_data[:] = []
        init.data_type = TensorProto.BFLOAT16

    # Run shape inference
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    # Update input/output types
    for value_info in list(model.graph.input) + list(model.graph.output):
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            value_info.type.tensor_type.elem_type = TensorProto.BFLOAT16

    # Update intermediate value_info types
    for value_info in model.graph.value_info:
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            value_info.type.tensor_type.elem_type = TensorProto.BFLOAT16

    print(f"[BF16] Converted {total_count} weight values, max error: {max_error:.6f}")
    return model


@pytest.fixture
def onnx_layer_model(request):
    case_name = request.param.name
    model_data = request.param.data
    version = "onnx_layer_model_" + case_name

    if isinstance(request.param, OnnxLayerCase):
        if request.config.getoption("--onnx-print-layer-info") and not request.param.is_full_model:
            print(_format_layer_node_name(request.param))

    return VersionedUncachedData(data=model_data, version=version)


# Re-export numpy executor functions and fixture for backward compatibility
# These are defined in .numpy to avoid circular dependencies with .torch
from .numpy import (
    _has_bf16_matmul,
    _has_bf16_einsum,
    _numpy_maxpool,
    _execute_onnx_model_numpy,
    numpy_reference_results,
)


@versioned_unhashable_object_fixture
def composite_reference_results(request, onnx_model_file, input_data):
    """
    Generate reference using a chained fallback strategy:
    1. ONNXRuntime (fastest, most accurate for f32)
    2. numpy fallback (for bf16 MatMul/Einsum/MaxPool)
    3. llvmcpu fallback (IREE reference compilation)
    4. torch fallback (last resort for bf16 models with unsupported ops)
    """
    onnx_model = onnx.load(str(onnx_model_file))

    # 1. Try ONNXRuntime first
    if not _has_bf16_matmul(onnx_model) or not _has_bf16_einsum(onnx_model):
        try:
            ort_session = onnxruntime.InferenceSession(str(onnx_model_file))
            ort_inputs = {inp.name: input_data[i] for i, inp in enumerate(ort_session.get_inputs())}
            return ort_session.run(None, ort_inputs)
        except Exception:
            pass

    # 2. Try numpy fallback
    try:
        return _execute_onnx_model_numpy(onnx_model, input_data)
    except Exception:
        pass

    # 3. Try llvmcpu fallback
    try:
        return request.getfixturevalue("llvmcpu_reference_results").data
    except Exception:
        pass

    # 4. Last resort: torch fallback
    print("Warning: All previous references (ONNXRuntime, numpy, llvmcpu) failed, falling back to torch reference")
    return request.getfixturevalue("torch_reference_results").data
