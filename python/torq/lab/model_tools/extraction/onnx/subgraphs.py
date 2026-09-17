# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""General ONNX subgraph matching and extraction.

Node-index range extraction (:func:`extract_onnx_subgraph`) and op-chain
boundary matching (:func:`extract_boundary_tensors` / :func:`extract_subgraphs`).
Decoder-specific representative components live in
:mod:`torq.lab.model_tools.extraction.onnx.decoder_components`.
"""

import hashlib
import os
from pathlib import Path
from shutil import rmtree

import onnx
import onnx_graphsurgeon as gs

from .layers import ModelWithMetadata, _build_model_from_node_subset, _fix_batch_dimension_to_one


def extract_onnx_subgraph(model, from_index, to_index, quantize=False):
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
    # Unwrap ModelWithMetadata if needed
    if hasattr(model, 'model'):
        model = model.model

    # Run shape inference on the full model so value_info is populated.
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

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
        quantize=quantize,
    )

    # Fix dynamic batch dimensions to static 1 for subgraph testing
    _fix_batch_dimension_to_one(final_model)

    return ModelWithMetadata(final_model, from_index)

def extract_boundary_tensors(
    model: onnx.ModelProto,
    ops_chain: list[str]
) -> list[dict[str, list | str]]:

    def _unique_subgraph_id(inputs: list[str], outputs: list[str], hash_length: int = 8) -> str:
        id_str = "|".join(inputs) + ">>" + "|".join(outputs) + ">>" + "|".join(ops_chain)
        return hashlib.sha256(id_str.encode()).hexdigest()[:hash_length]

    def _filter_tensors(tensors: list[gs.Constant | gs.Variable]) -> list[str]:
        tensor_names: list[str] = []
        for t in tensors:
            if isinstance(t, gs.Variable) and t.name:
                tensor_names.append(t.name)
        return tensor_names

    def _find_matches(curr: gs.Node, top: gs.Node, remaining: list[str]):
        if not remaining:
            inputs: list[str]  = _filter_tensors(top.inputs)
            outputs: list[str] = _filter_tensors(curr.outputs)
            if not inputs or not outputs:
                return
            if (subgraph_id := _unique_subgraph_id(inputs, outputs)) not in found_subgraph_ids:
                boundary_tensors.append(
                    {
                        "subgraph_id": subgraph_id,
                        "ops_chain": ops_chain,
                        "inputs": inputs,
                        "outputs": outputs
                    }
                )
                found_subgraph_ids.add(subgraph_id)
            return

        for out_t in curr.outputs:
            for consumer in out_t.outputs:
                if consumer.op == remaining[0]:
                    _find_matches(consumer, top, remaining[1:])

    if not ops_chain:
        raise ValueError("`ops` must contain at least one op type")
    boundary_tensors = []
    found_subgraph_ids: set[str] = set()
    graph = gs.import_onnx(model)
    for node in graph.nodes:
        if node.op == ops_chain[0]:
            _find_matches(node, node, ops_chain[1:])
    return boundary_tensors


def extract_subgraphs(
    model_path: str | os.PathLike,
    ops_chains: list[list[str]],
    save_dir: str | os.PathLike,
    limit: int | None = None
) -> list[Path]:
    model = onnx.load(model_path)
    subgraphs_dirs: list[Path] = []
    for ops_chain in ops_chains:
        chain_name = "-".join(ops_chain)
        subgraphs_dir = Path(save_dir) / chain_name
        subgraphs_dir.mkdir(exist_ok=True, parents=True)
        for f in subgraphs_dir.iterdir():
            if f.is_file() and f.suffix == ".onnx" and chain_name in f.name:
                f.unlink()
            if f.is_dir() and chain_name in f.name:
                rmtree(f, ignore_errors=True)
        matches = extract_boundary_tensors(model, ops_chain)
        for i, match in enumerate(matches):
            if isinstance(limit, int) and i >= limit:
                break
            output_path = subgraphs_dir / f"{chain_name}_{i + 1}.onnx"
            onnx.utils.extract_model(model_path, output_path, match["inputs"], match["outputs"])
            graph = gs.import_onnx(onnx.load(output_path))
            graph.name = "main"
            graph = graph.cleanup(
                remove_unused_graph_inputs=True,
                remove_unused_node_outputs=True
            ).toposort()
            extracted = gs.export_onnx(graph)
            extracted = onnx.shape_inference.infer_shapes(extracted, check_type=True, strict_mode=True)
            onnx.checker.check_model(extracted, full_check=True)
            onnx.save(extracted, output_path)
        if matches:
            subgraphs_dirs.append(subgraphs_dir)
    return subgraphs_dirs
