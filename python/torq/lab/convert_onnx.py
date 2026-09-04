"""ONNX model conversions: FP32->BF16 and INT64->INT32.

These are kept in a separate module because the rewrites are non-trivial
(schema repair, value-range verification, type propagation) and onnx.py is
already large; convert_onnx.py is the single place to reason about them.
"""

import copy

import numpy as np
import onnx
from onnx import TensorProto

# ---- BF16 conversion ----


def is_model_bf16(model: onnx.ModelProto) -> bool:
    """Check if model is already in BF16 format."""
    return any(
        init.data_type == TensorProto.BFLOAT16
        for init in model.graph.initializer
    )


def _float32_to_bfloat16(arr: np.ndarray) -> np.ndarray:
    """Convert float32 numpy array to bfloat16 (stored as uint16)."""
    arr_uint32 = arr.view(np.uint32)
    arr_bf16 = (arr_uint32 >> 16).astype(np.uint16)
    return arr_bf16


def _fix_batch_dimension_to_one(model: onnx.ModelProto) -> int:
    """Fix dynamic batch dimensions (?, -1) to 1 for all inputs, outputs, and value_info."""
    modified_count = 0
    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        tensor_type = value_info.type.tensor_type
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

    # Convert FP32 tensors carried by Constant nodes (e.g. rotary-embedding
    # tables in gptneox). The value_info relabel above marks every float
    # tensor BF16, so an unconverted f32 Constant would leave the graph
    # type-inconsistent (Gather output annotated bf16 while its table is
    # f32 -> "cast incompatible" at import).
    const_count = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name != "value" or not attr.HasField("t"):
                continue
            tensor = attr.t
            if tensor.data_type != TensorProto.FLOAT:
                continue
            if tensor.raw_data:
                fp32_data = np.frombuffer(tensor.raw_data, dtype=np.float32).copy()
            elif tensor.float_data:
                fp32_data = np.array(tensor.float_data, dtype=np.float32)
            else:
                continue
            # Skip scalars (epsilon and friends): the importer handles f32
            # scalars fine, and narrowing them would perturb numerics.
            if fp32_data.size <= 1:
                continue
            bf16_data = _float32_to_bfloat16(fp32_data)
            const_count += len(fp32_data)
            tensor.raw_data = bf16_data.tobytes()
            tensor.float_data[:] = []
            tensor.data_type = TensorProto.BFLOAT16
    if const_count:
        print(f"[BF16] Converted {const_count} Constant-node values")

    # LayerNormalization with stash_type=FLOAT (1) would decompose to an
    # internal f32 cone (mean/variance reductions + normalize) that the NSS
    # executor cannot lower. Retarget the stash to BFLOAT16 (16) so
    # torch-mlir computes it directly in bf16 and no f32 cone is created.
    # stash_type defaults to FLOAT when the attribute is absent (e.g. bert's
    # LNs carry no explicit stash_type), so add it in that case too.
    ln_count = 0
    for node in model.graph.node:
        if node.op_type != "LayerNormalization":
            continue
        stash = None
        for attr in node.attribute:
            if attr.name == "stash_type":
                stash = attr
                break
        if stash is None:
            stash = node.attribute.add()
            stash.name = "stash_type"
            stash.type = onnx.AttributeProto.INT  # required by the ONNX checker
            stash.i = TensorProto.FLOAT
        if stash.i == TensorProto.FLOAT:
            stash.i = TensorProto.BFLOAT16
            ln_count += 1
    if ln_count:
        print(f"[BF16] Retargeted {ln_count} LayerNormalization stash_type to bf16")

    return model


# ---- INT64 -> INT32 conversion ----


def is_model_int32(model: onnx.ModelProto) -> bool:
    """Check if the model has no INT64 tensors left to convert."""
    if any(init.data_type == TensorProto.INT64 for init in model.graph.initializer):
        return False
    for value_info in list(model.graph.input) + list(model.graph.output):
        if value_info.type.tensor_type.elem_type == TensorProto.INT64:
            return False
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t") and attr.t.data_type == TensorProto.INT64:
                return False
    return True


def _convert_int64_tensor_data(tensor: TensorProto, name: str, stats: dict) -> bool:
    """Convert an INT64 tensor's payload to INT32 in place.

    Fails (raises ValueError) if any value falls outside the int32 range:
    silently truncating out-of-range values would corrupt the model, so this
    is a hard correctness gate rather than a warning. The sole exception is
    the INT64_MIN/INT64_MAX sentinel pair: transformer attention masks use
    them as -inf/+inf proxies, and clamping them to INT32_MIN/INT32_MAX
    preserves that role exactly. Returns True when the tensor was converted.
    """
    if tensor.data_type != TensorProto.INT64:
        return False

    if tensor.raw_data:
        values = np.frombuffer(tensor.raw_data, dtype=np.int64).copy()
    elif tensor.int64_data:
        values = np.array(tensor.int64_data, dtype=np.int64)
    else:
        # No inline payload (e.g. external data); nothing we can safely convert.
        return False

    if values.size:
        # Clamp exact int64 sentinels before the range gate (see docstring).
        sentinel_mask = (values == np.int64(2**63 - 1)) | (values == np.int64(-(2**63)))
        if sentinel_mask.any():
            values = values.copy()
            values[values == np.int64(2**63 - 1)] = np.int64(2**31 - 1)
            values[values == np.int64(-(2**63))] = np.int64(-(2**31))
            stats["sentinel_clamped_tensors"] = stats.get("sentinel_clamped_tensors", 0) + 1
            print(
                f"[INT32] tensor '{name}': clamped {int(sentinel_mask.sum())} "
                f"INT64_MIN/MAX sentinel value(s) to INT32_MIN/MAX"
            )
        v_min, v_max = int(values.min()), int(values.max())
        if v_max >= 2**31 or v_min < -(2**31):
            raise ValueError(
                f"[INT32] INT64->INT32 conversion failed: tensor '{name}' contains "
                f"values outside the int32 range (min={v_min}, max={v_max})"
            )
        stats["max_abs"] = max(stats["max_abs"], abs(v_min), abs(v_max))
        stats["value_count"] += values.size

    # Per-tensor round-trip verification (mirrors the BF16 max-error check):
    # narrowing int64 -> int32 -> int64 must reproduce every value exactly.
    narrowed = values.astype(np.int32)
    if not np.array_equal(narrowed.astype(np.int64), values):
        raise ValueError(
            f"[INT32] INT64->INT32 round-trip mismatch for tensor '{name}'"
        )

    tensor.raw_data = narrowed.tobytes()
    tensor.int64_data[:] = []
    tensor.data_type = TensorProto.INT32
    stats["tensor_count"] += 1
    return True


def _collect_subgraph_input_names(graph: onnx.GraphProto) -> set:
    """Collect tensor names referenced by nodes inside control-flow subgraphs.

    Outer-scope values captured by If/Loop/Scan bodies are referenced by name;
    a Cast repair node inserted in the main graph would not be visible there,
    so tensors in this set must be left untouched.
    """
    names = set()
    for node in graph.node:
        for attr in node.attribute:
            sub_graphs = []
            if attr.HasField("g"):
                sub_graphs.append(attr.g)
            sub_graphs.extend(attr.graphs)
            for sub in sub_graphs:
                for sub_node in sub.node:
                    names.update(n for n in sub_node.input if n)
                names.update(_collect_subgraph_input_names(sub))
    return names


def _propagate_i64_status(graph: onnx.GraphProto, status: dict, opset: int) -> dict:
    """Forward-propagate int64-ness through the main graph (fixpoint).

    ``status`` maps tensor name -> "i64" | "not_i64" | "unknown" and is
    updated in place. Only tracks whether a tensor is int64 after conversion;
    control-flow ops and unknown schemas are left at their inferred status.
    """
    for _ in range(len(graph.node) + 1):
        changed = False
        for node in graph.node:
            if any(attr.HasField("g") or attr.graphs for attr in node.attribute):
                continue
            try:
                schema = onnx.defs.get_schema(node.op_type, opset, "")
            except Exception:
                continue
            ins, outs = schema.inputs, schema.outputs
            if not outs:
                continue
            for oidx, out_name in enumerate(node.output):
                if not out_name:
                    continue
                oparam = outs[min(oidx, len(outs) - 1)]
                # Inputs bound to the same type parameter as this output.
                group = [
                    nm for i, nm in enumerate(node.input)
                    if nm and ins
                    and ins[min(i, len(ins) - 1)].type_str == oparam.type_str
                ]
                if not group:
                    # Fixed-type output (e.g. Equal -> bool, Shape -> int64).
                    if len(oparam.types) != 1:
                        continue
                    new = "i64" if next(iter(oparam.types)) == "tensor(int64)" else "not_i64"
                else:
                    gstat = [status.get(nm, "unknown") for nm in group]
                    if all(s == "not_i64" for s in gstat):
                        new = "not_i64"
                    elif all(s == "i64" for s in gstat):
                        new = "i64"
                    else:
                        continue
                if status.get(out_name, "unknown") != new:
                    status[out_name] = new
                    changed = True
        if not changed:
            break
    return status


def _needs_int64_repair(node: onnx.NodeProto, input_idx: int, converted_names: set,
                        status: dict, opset: int) -> bool:
    """Check whether a consumer input must keep receiving tensor(int64).

    True when the op schema does not allow tensor(int32) for this input (e.g.
    the shape input of Reshape/Unsqueeze/ConstantOfShape), when the schema is
    unknown (custom ops), or when a sibling input sharing the same type
    parameter still carries int64 (type parameters must unify within a node).
    """
    try:
        schema = onnx.defs.get_schema(node.op_type, opset, "")
    except Exception:
        return True
    inputs = schema.inputs
    if not inputs:
        return True
    param = inputs[min(input_idx, len(inputs) - 1)]
    if "tensor(int32)" not in param.types:
        return True
    for j, sibling in enumerate(node.input):
        if j == input_idx or not sibling or sibling in converted_names:
            continue
        sib_param = inputs[min(j, len(inputs) - 1)]
        if sib_param.type_str != param.type_str:
            continue
        # Unknown sibling types are treated as int64: casting the converted
        # input back to int64 then restores the original (valid) edge type.
        if status.get(sibling, "unknown") != "not_i64":
            return True
    return False


def convert_int64_to_int32(model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert INT64 tensors to INT32 in an ONNX model.

    Conservative conversion, analogous to convert_fp32_to_bf16:
      - INT64 initializers and Constant node value tensors (including
        ConstantOfShape fill values) are re-typed and their payloads narrowed
        (hard failure on out-of-range values).
      - Cast nodes with attribute to=INT64 are retargeted to INT32.
      - Tensor element type annotations (graph inputs, plus outputs/value_info
        entries that refer to converted tensors) are updated to INT32.
      - Consumers that must keep receiving tensor(int64) — either because the
        op schema rejects int32 for that input (e.g. the shape input of
        Reshape/Unsqueeze/ConstantOfShape) or because a sibling input sharing
        the same type parameter still carries int64 — are repaired by
        inserting a Cast back to INT64, so the model stays valid.
    Non-tensor attributes (axes etc.) are left untouched.

    Returns the converted model and prints a summary.
    """
    model = copy.deepcopy(model)  # Don't modify original
    graph = model.graph

    opset = next(
        (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 18
    )
    # Tensors captured by control-flow subgraphs (If/Loop/Scan) are referenced
    # by name inside those bodies; leave them untouched so the conversion never
    # changes a type those subgraphs still read.
    protected = _collect_subgraph_input_names(graph)

    # Static element-type map of the ORIGINAL model, used to detect sibling
    # inputs that stay int64 and force a converted input back to int64.
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    type_map = {}
    for vi in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info):
        type_map[vi.name] = vi.type.tensor_type.elem_type
    for init in inferred.graph.initializer:
        type_map[init.name] = init.data_type
    for node in inferred.graph.node:
        if not node.output:
            continue
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    type_map[node.output[0]] = attr.t.data_type
        elif node.op_type == "ConstantOfShape":
            # Output type follows the value attribute; default is int64 zero.
            dt = TensorProto.INT64
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    dt = attr.t.data_type
            type_map[node.output[0]] = dt

    stats = {"tensor_count": 0, "value_count": 0, "max_abs": 0}
    converted_names = set()

    # Convert initializers (weights / constant tables)
    for init in graph.initializer:
        if init.name in protected:
            continue
        if _convert_int64_tensor_data(init, init.name, stats):
            converted_names.add(init.name)

    # Convert Constant node tensor attributes (and ConstantOfShape fill values)
    for node in graph.node:
        if not node.output or node.output[0] in protected:
            continue
        if node.op_type not in ("Constant", "ConstantOfShape"):
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                if _convert_int64_tensor_data(attr.t, f"{node.op_type}:{node.output[0]}", stats):
                    converted_names.add(node.output[0])

    # Retarget Cast nodes whose target type is INT64
    cast_count = 0
    for node in graph.node:
        if node.op_type != "Cast" or not node.output:
            continue
        if node.output[0] in protected:
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == TensorProto.INT64:
                attr.i = TensorProto.INT32
                cast_count += 1
                converted_names.add(node.output[0])

    # Update graph input element type annotations
    type_count = 0
    for value_info in graph.input:
        if value_info.name in protected:
            continue
        if value_info.type.tensor_type.elem_type == TensorProto.INT64:
            value_info.type.tensor_type.elem_type = TensorProto.INT32
            converted_names.add(value_info.name)
            type_count += 1

    # Post-conversion int64 status of every tensor: original inferred types,
    # overridden for converted tensors, then propagated through the graph so
    # that e.g. an elementwise chain rooted at a converted ConstantOfShape is
    # known to be int32 (not int64) downstream.
    status = {
        name: ("i64" if t == TensorProto.INT64 else "not_i64")
        for name, t in type_map.items()
    }
    for name in converted_names:
        status[name] = "not_i64"
    status = _propagate_i64_status(graph, status, opset)

    # Tensors whose element type changed from int64 to int32: converted
    # payloads/annotations plus tensors re-typed through propagation (e.g. an
    # elementwise chain rooted at a converted ConstantOfShape fill value).
    changed_names = set(converted_names)
    for name, t in type_map.items():
        if t == TensorProto.INT64 and status.get(name) == "not_i64":
            changed_names.add(name)

    # Update output / value_info annotations for every re-typed tensor
    for value_info in list(graph.output) + list(graph.value_info):
        if value_info.name in changed_names and \
                value_info.type.tensor_type.elem_type == TensorProto.INT64:
            value_info.type.tensor_type.elem_type = TensorProto.INT32
            type_count += 1

    # Repair consumers that must keep receiving tensor(int64): feed them a
    # Cast back to INT64 of the converted tensor. One cast per tensor, reused
    # by all such consumers; emitted before its first consumer to preserve
    # topological order.
    repairs = {}  # tensor name -> (cast output name, cast node or None once emitted)
    new_nodes = []
    for node in graph.node:
        for idx, name in enumerate(list(node.input)):
            if name not in changed_names:
                continue
            if not _needs_int64_repair(node, idx, converted_names, status, opset):
                continue
            repair = repairs.get(name)
            if repair is None:
                cast_out = f"{name}_as_int64"
                cast_node = onnx.helper.make_node(
                    "Cast", [name], [cast_out],
                    name=f"{name}_as_int64_cast", to=TensorProto.INT64,
                )
                new_nodes.append(cast_node)
                repairs[name] = (cast_out, None)
                node.input[idx] = cast_out
            else:
                cast_out, cast_node = repair
                if cast_node is not None:
                    new_nodes.append(cast_node)
                    repairs[name] = (cast_out, None)
                node.input[idx] = cast_out
        new_nodes.append(node)
    if repairs:
        del graph.node[:]
        graph.node.extend(new_nodes)
    repair_count = len(repairs)

    sentinel_count = stats.get("sentinel_clamped_tensors", 0)
    sentinel_note = f", {sentinel_count} sentinel-clamped" if sentinel_count else ""
    print(f"[INT32] Converted {stats['tensor_count']} tensors, max |value|: {stats['max_abs']}{sentinel_note}")
    print(f"[INT32] Retargeted {cast_count} Cast node(s), "
          f"updated {type_count} tensor type annotation(s), "
          f"inserted {repair_count} Cast-to-int64 repair node(s)")
    return model
