import pytest

import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import numpy as np
import onnxruntime
import subprocess
import sys

import json
from pathlib import Path

from torq.testing.hf import get_hf_model_file
from torq.testing.cases import Case

from .versioned_fixtures import (versioned_generated_file_fixture,
  versioned_cached_data_fixture,
  versioned_hashable_object_fixture,
  versioned_unhashable_object_fixture,
  versioned_static_file_fixture,
  VersionedUncachedData
 )


"""
This module provides fixtures and utilities for testing onnx models.
"""

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
    while index < node_count:

        if (nodes[index].op_type.lower() == 'constant'):
            index += 1
            continue

        group_size = match_group(index)
        if group_size > 0:
            part_nodes = nodes[index:index + group_size]
            index += group_size
        else:
            part_nodes = [nodes[index]]
            index += 1
        # Determine tensors produced and used by the part
        produced = set()
        used = set()
        for n in part_nodes:
            for o in n.output:
                produced.add(o)
            for inp in n.input:
                if inp:
                    used.add(inp)

        external_inputs = used - produced

        # external outputs: produced names that are consumed by nodes outside
        external_outputs = set()
        orig_output_names = {o.name for o in graph.output}
        for name in produced:
            consumed_outside = False
            for idx_node, other in enumerate(nodes):
                # if this other node is part of the part_nodes, skip
                # compare by identity of node object index ranges
                # simpler: check if other is in part_nodes
                if other in part_nodes:
                    continue
                if name in other.input:
                    consumed_outside = True
                    break
            if consumed_outside or (name in orig_output_names):
                external_outputs.add(name)

        # Build maps for metadata; prefer inferred_model when available
        meta = model

        orig_inputs = {vi.name: vi for vi in meta.graph.input}
        orig_value_info = {vi.name: vi for vi in meta.graph.value_info}
        orig_initializers = {init.name: init for init in graph.initializer}

        # Build new input/value_info/initializer lists for the part
        new_inputs = []
        new_outputs = []
        new_initializers = []

        import copy as _copy

        # Helper to synthesize a simple value_info from an initializer
        def _value_info_from_initializer(init):
            try:
                arr = numpy_helper.to_array(init)
                shape = list(arr.shape)
                return helper.make_tensor_value_info(init.name, init.data_type, shape)
            except Exception:
                # fallback to scalar unknown-shape float
                return helper.make_tensor_value_info(init.name, TensorProto.FLOAT, [])

        # Populate inputs
        for name in sorted(external_inputs):
            if name in orig_inputs:
                new_inputs.append(_copy.deepcopy(orig_inputs[name]))
            elif name in orig_value_info:
                # convert value_info entry into an input
                new_inputs.append(_copy.deepcopy(orig_value_info[name]))
            elif name in orig_initializers:
                init = orig_initializers[name]
                new_initializers.append(_copy.deepcopy(init))
                new_inputs.append(_value_info_from_initializer(init))
            else:
                # unknown; synthesize a placeholder value_info (float, scalar)
                new_inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, []))

        # Include initializers referenced by used names
        for name in sorted((external_inputs | used)):
            if name in orig_initializers:
                new_initializers.append(_copy.deepcopy(orig_initializers[name]))

        # Populate outputs
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

        # Also add any value_info for produced/used tensors to aid shape inference
        new_value_info = []
        for name in sorted(produced | used):
            if name in orig_value_info:
                new_value_info.append(_copy.deepcopy(orig_value_info[name]))

        # Deep-copy nodes so we don't mutate original model
        part_nodes_copy = [_copy.deepcopy(n) for n in part_nodes]

        # Ensure initializer names are unique for this part and update
        # references in nodes and value infos accordingly.
        name_map = {}
        # collect existing names used in this part to avoid collisions
        existing_names = set()
        for n in part_nodes_copy:
            existing_names.update([s for s in n.input if s])
            existing_names.update([s for s in n.output if s])
        existing_names.update([vi.name for vi in new_inputs])
        existing_names.update([vi.name for vi in new_outputs])
        existing_names.update([vi.name for vi in new_value_info])

        for idx_init, init in enumerate(new_initializers):
            base = init.name
            # generate a candidate new name
            candidate = f"{base}_part{part_num}_init{idx_init}"
            # avoid accidental collisions
            i = 0
            while candidate in existing_names:
                i += 1
                candidate = f"{base}_part{part_num}_init{idx_init}_{i}"
            # record mapping and update the initializer name
            name_map[base] = candidate
            existing_names.add(candidate)
            init.name = candidate

        # apply renaming to node inputs
        if name_map:
            for n in part_nodes_copy:
                for ii, v in enumerate(list(n.input)):
                    if v in name_map:
                        n.input[ii] = name_map[v]

            # update new_inputs / new_outputs / new_value_info names where they reference initializers
            for vi in new_inputs:
                if vi.name in name_map:
                    vi.name = name_map[vi.name]
            for vi in new_outputs:
                if vi.name in name_map:
                    vi.name = name_map[vi.name]
            for vi in new_value_info:
                if vi.name in name_map:
                    vi.name = name_map[vi.name]

        # Create new graph for the part using the synthesized inputs/outputs
        part_graph = helper.make_graph(
            part_nodes_copy,
            f'part{part_num}_graph',
            inputs=new_inputs,
            outputs=new_outputs,
            initializer=new_initializers
        )

        # attach value_info entries if present
        if new_value_info:
            part_graph.value_info.extend(new_value_info)

        part_model = helper.make_model(part_graph)
        part_model.ir_version = model.ir_version
        part_model.opset_import.extend(model.opset_import)

        layer_name = f"layer_{part_nodes[0].op_type}_{part_num}"

        final_model = part_model

        # to trace exact layer_number in model
        part_num += 1

        # Try to run shape inference and checker on the part. If inference
        # succeeds, save the inferred model; otherwise try checker on raw model.
        try:
            inferred_part = shape_inference.infer_shapes(part_model)
            onnx.checker.check_model(inferred_part)
            final_model = inferred_part
        except Exception as e:
            print(f'layer {layer_name}: inference/check failed: {e}; saving raw part for debugging')

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
                # print(f"Skipping duplicate model for layer {layer_name}")
                continue

        layer_configs[layer_name] = final_model

    return layer_configs


def generate_onnx_layers_from_hf(cache, repo_id, filename, node_groups=None, dedup=True):
    model = get_full_model(get_hf_model_file(cache, repo_id, filename))
    layers = generate_onnx_layers_from_model(model, node_groups, dedup)
    model_prefix = Path(filename).stem
    return [Case(f"{model_prefix}_{key}", layer) for key, layer in layers.items()] + [ Case(f"{model_prefix}_full_model", model) ]


def generate_onnx_layer_from_file(filepath:Path, node_groups=None, dedup=True):
    model = get_full_model(str(filepath))
    layers = generate_onnx_layers_from_model(model, node_groups, dedup)
    return [Case(f"{filepath.stem}_{key}", layer) for key, layer in layers.items()] + [ Case(f"{filepath.stem}_full_model", model) ]


@pytest.fixture
def onnx_model(request, case_config):
    return request.getfixturevalue(case_config['onnx_model'])


@versioned_generated_file_fixture("onnx")
def onnx_model_file(request, versioned_file, onnx_model):
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, versioned_file)


@versioned_generated_file_fixture("mlir")
def onnx_mlir_model_file(request, versioned_file, onnx_model_file):
    subprocess.check_output(
        [sys.executable, "-m", "iree.compiler.tools.import_onnx",
          str(onnx_model_file), "-o", str(versioned_file), "--data-prop"])


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


@pytest.fixture
def onnx_layer_model(request):
    case_name = request.param.name
    model_data = request.param.data
    version = "onnx_layer_model_" + case_name

    return VersionedUncachedData(data=model_data, version=version)


def _has_bf16_matmul(model):
    """Check if model contains MatMul with bf16."""
    graph = model.graph
    value_info = {vi.name: vi.type.tensor_type.elem_type 
                  for vi in list(graph.value_info) + list(graph.input) + list(graph.output)
                  if hasattr(vi.type, 'tensor_type')}
    for node in graph.node:
        if node.op_type == 'MatMul':
            node_inputs = list(node.input)
            node_outputs = list(node.output)
            if any(value_info.get(name) == TensorProto.BFLOAT16 for name in node_inputs + node_outputs):
                return True
            if any(init.data_type == TensorProto.BFLOAT16 for init in graph.initializer 
                   if init.name in node_inputs):
                return True
    return False


def _execute_onnx_model_numpy(model, input_data):
    """
    Execute ONNX model using numpy, with special handling for bf16 MatMul operations.
    NumPy's matmul accepts bf16 arrays directly and handles promotion to float32 internally,
    then we convert the result back to bf16 if needed. Uses manual execution with numpy
    for MatMul and onnxruntime for other operations.
    """
    graph = model.graph
    
    # Build value info map
    value_info_map = {vi.name: vi for vi in list(graph.value_info) + list(graph.input) + list(graph.output)}
    
    # Initialize tensor values
    tensor_values = {}
    
    # Load initializers
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        tensor_values[init.name] = arr
    
    # Set input values
    input_names = [inp.name for inp in graph.input]
    for i, inp_name in enumerate(input_names):
        if i < len(input_data):
            tensor_values[inp_name] = input_data[i]
    
    # Execute nodes in order
    for node in graph.node:
        inputs = [tensor_values.get(inp) for inp in node.input if inp and inp in tensor_values]
        
        if node.op_type == 'MatMul' and len(inputs) >= 2:
            result = np.matmul(inputs[0], inputs[1])
            # Convert to bf16 if output type requires it
            if node.output and node.output[0] in value_info_map:
                vi = value_info_map[node.output[0]]
                if hasattr(vi.type, 'tensor_type') and vi.type.tensor_type.elem_type == TensorProto.BFLOAT16:
                    result = result.astype('bfloat16')
            tensor_values[node.output[0]] = result
        else:
            # For other operations, try to use onnxruntime
            try:
                # Create a temporary model with just this node and required inputs/outputs
                node_inputs = [helper.make_tensor_value_info(inp, TensorProto.FLOAT, []) 
                               for inp in node.input if inp in tensor_values]
                node_outputs = [helper.make_tensor_value_info(out, TensorProto.FLOAT, []) 
                                for out in node.output]
                node_inits = [init for init in graph.initializer if init.name in node.input]
                
                temp_graph = helper.make_graph([node], 'temp_graph', node_inputs, node_outputs, node_inits)
                temp_model = helper.make_model(temp_graph)
                ort_session = onnxruntime.InferenceSession(temp_model.SerializeToString())
                ort_inputs = {inp: tensor_values[inp] for inp in node.input if inp in tensor_values}
                outputs = ort_session.run(None, ort_inputs)
                for i, output_name in enumerate(node.output):
                    tensor_values[output_name] = outputs[i]
            except Exception as e:
                # If onnxruntime fails, raise to fall back to llvmcpu
                print(f"Warning: Could not execute {node.op_type} with onnxruntime: {e}")
                raise
    
    # Collect output values
    outputs = []
    for output in graph.output:
        if output.name in tensor_values:
            outputs.append(tensor_values[output.name])
    
    return outputs


@versioned_unhashable_object_fixture
def numpy_reference_results(request, onnx_model_file, input_data, llvmcpu_reference_results):
    """Generate reference using numpy for bf16 MatMul (onnxruntime doesn't support it).
    Falls back to llvmcpu if numpy cannot handle an operation."""
    onnx_model = onnx.load(str(onnx_model_file))
    if not _has_bf16_matmul(onnx_model):
        try:
            ort_session = onnxruntime.InferenceSession(str(onnx_model_file))
            ort_inputs = {inp.name: input_data[i] for i, inp in enumerate(ort_session.get_inputs())}
            return ort_session.run(None, ort_inputs)
        except Exception:
            pass
    try:
        return _execute_onnx_model_numpy(onnx_model, input_data)
    except Exception as e:
        # Fall back to llvmcpu if numpy cannot handle the operation
        print(f"Warning: Numpy reference failed, falling back to llvmcpu: {e}")
        return llvmcpu_reference_results
