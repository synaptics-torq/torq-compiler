"""Extract representative ONNX subgraphs from decoder-only LLMs.

Splits a model into its architectural sections (embedding, decoder layer,
final norm, lm head) by discovering section boundaries from the graph
topology rather than relying on hardcoded tensor names.

Designed for Gemma3 but should work for most HuggingFace-exported
decoder-only transformer models that follow the standard naming convention:
  /model/embed_tokens/...
  /model/layers.{i}/...
  /model/norm/...
  /lm_head/...
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if __name__ == "__main__":
    # prevent shadowing ./onnx.py and ./numpy.py
    _script_dir = str(Path(__file__).resolve().parent)
    sys.path = [p for p in sys.path if str(Path(p).resolve()) != _script_dir]

import onnx
import onnx_graphsurgeon as gs


LAYER_NODE_RE = re.compile(r"^/model/layers\.(\d+)/")


@dataclass(frozen=True)
class _SectionBoundaries:
    """Tensor names at the boundaries between architectural sections."""

    embed_output: str  # tensor flowing from embedding into layer 0
    layer_residual_template: str  # f-string pattern for layer residual output
    final_norm_output: str  # tensor flowing from final norm into lm_head
    logits_output: str  # final model output tensor
    last_layer_index: int
    kv_output_template: str  # e.g. "present.{i}.key_value"


def _discover_boundaries(graph: gs.Graph) -> _SectionBoundaries:
    """Discover section boundaries from the graph structure."""
    layer_indices: set[int] = set()
    for node in graph.nodes:
        m = LAYER_NODE_RE.match(node.name)
        if m:
            layer_indices.add(int(m.group(1)))
    if not layer_indices:
        raise ValueError("No decoder layers found (expected nodes named /model/layers.{i}/...)")

    last_layer = max(layer_indices)

    # Find embedding output: the tensor produced by /model/embed_tokens/* that
    # feeds into layer 0 nodes. If no embed_tokens section exists, the graph
    # input (token_embedding) feeds directly into layer 0.
    embed_output = None
    for node in graph.nodes:
        if node.name.startswith("/model/embed_tokens/"):
            for out in node.outputs:
                if any(LAYER_NODE_RE.match(c.name) and c.name.startswith("/model/layers.0/")
                       for c in out.outputs):
                    embed_output = out.name
                    break
        if embed_output:
            break
    if not embed_output:
        # No embed_tokens section; check if a graph input feeds layer 0 directly
        for inp in graph.inputs:
            if isinstance(inp, gs.Variable) and any(
                c.name.startswith("/model/layers.0/") for c in inp.outputs
            ):
                embed_output = inp.name
                break
    if not embed_output:
        raise ValueError("Could not find embedding output tensor feeding into layer 0")

    # Find the residual output pattern: the Add node output from layer 0 that
    # feeds into layer 1's layernorm (the main residual stream, not shared
    # attention masks or rotary embeddings).
    layer_residual_template = None
    for node in graph.nodes:
        if not node.name.startswith("/model/layers.0/") or node.op != "Add":
            continue
        for out in node.outputs:
            consumers_in_next = [c for c in out.outputs if c.name.startswith("/model/layers.1/")]
            if any("layernorm" in c.name.lower() or "norm" in c.name.lower()
                   for c in consumers_in_next):
                layer_residual_template = out.name.replace("/layers.0/", "/layers.{i}/")
                break
        if layer_residual_template:
            break
    if not layer_residual_template:
        raise ValueError("Could not discover inter-layer residual tensor pattern")

    # Find final norm output: the tensor that feeds the first /lm_head/ node.
    # This handles both standard /model/norm/ and architectures where final norm
    # lives inside the last layer's namespace.
    final_norm_output = None
    for node in graph.nodes:
        if not node.name.startswith("/lm_head/"):
            continue
        # First non-constant input to the first lm_head node
        for inp in node.inputs:
            if isinstance(inp, gs.Variable) and inp.inputs:
                final_norm_output = inp.name
                break
        if final_norm_output:
            break
    if not final_norm_output:
        raise ValueError("Could not find the tensor feeding into lm_head")

    # Find logits output
    logits_output = None
    for out in graph.outputs:
        if out.name == "logits" or "logit" in out.name.lower():
            logits_output = out.name
            break
    if not logits_output:
        # Fall back to first non-kv, non-conv output
        for out in graph.outputs:
            if "key_value" not in out.name and "present" not in out.name:
                logits_output = out.name
                break
    if not logits_output:
        raise ValueError("Could not identify logits output tensor")

    # Discover KV cache output pattern from graph outputs
    kv_output_template = ""
    for out in graph.outputs:
        if re.match(r"present\.0\.", out.name):
            kv_output_template = out.name.replace(".0.", ".{i}.")
            break

    return _SectionBoundaries(
        embed_output=embed_output,
        layer_residual_template=layer_residual_template,
        final_norm_output=final_norm_output,
        logits_output=logits_output,
        last_layer_index=last_layer,
        kv_output_template=kv_output_template,
    )


def _extract_subgraph(
    graph: gs.Graph,
    *,
    graph_name: str,
    input_renames: dict[str, str],
    output_renames: dict[str, str],
) -> gs.Graph:
    """Extract subgraph by cutting the graph at specified input/output tensors."""
    tensors = graph.tensors()
    original_inputs = {t.name: t for t in graph.inputs if isinstance(t, gs.Variable)}

    # Convert specified tensors into graph inputs (sever their producers)
    new_inputs: list[gs.Variable] = []
    for source_name, export_name in input_renames.items():
        tensor = tensors.get(source_name)
        if tensor is None:
            raise ValueError(f"Input tensor '{source_name}' not found in graph")
        if original_inputs.get(source_name) is tensor:
            # Already a graph input, just rename
            tensor.name = export_name
            new_inputs.append(tensor)
        else:
            # Internal tensor - replace with a new graph input variable
            replacement = gs.Variable(name=export_name, dtype=tensor.dtype, shape=tensor.shape)
            for consumer in list(tensor.outputs):
                for idx, inp in enumerate(consumer.inputs):
                    if inp is tensor:
                        consumer.inputs[idx] = replacement
            new_inputs.append(replacement)

    # Set outputs
    new_outputs: list[gs.Variable] = []
    for source_name, export_name in output_renames.items():
        tensor = tensors.get(source_name)
        if tensor is None:
            raise ValueError(f"Output tensor '{source_name}' not found in graph")
        tensor.name = export_name
        new_outputs.append(tensor)

    graph.name = graph_name
    graph.inputs = new_inputs
    graph.outputs = new_outputs
    graph.cleanup(
        remove_unused_graph_inputs=True,
        remove_unused_node_outputs=True,
    ).toposort()

    return graph


def _save_subgraph(
    graph: gs.Graph,
    output_path: Path,
    ir_version: int,
    metadata: dict[str, str] | None = None,
) -> Path:
    extracted_model = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
    extracted_model.ir_version = ir_version
    if metadata:
        for key, value in metadata.items():
            prop = extracted_model.metadata_props.add()
            prop.key = key
            prop.value = value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(extracted_model)
    onnx.save(extracted_model, output_path)
    return output_path


def _load_graph(model_path: Path) -> tuple[gs.Graph, int]:
    model = onnx.load(model_path)
    return gs.import_onnx(model), model.ir_version


def extract_representative_components(
    input_model_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Extract representative architectural components: embed_scale, decoder_block, final_norm, lm_head."""
    input_model_path = Path(input_model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load once without external data to discover boundaries
    probe_model = onnx.load(input_model_path, load_external_data=False)
    probe_graph = gs.import_onnx(probe_model)
    boundaries = _discover_boundaries(probe_graph)
    del probe_model

    outputs: list[Path] = []

    # 1. Embedding scale (gathered token embedding -> scaled hidden states).
    # This isolates the embedding scaling (Gemma multiplies the gathered token
    # embedding by sqrt(d_model)); the token-id lookup is cut off so the harness
    # can drive it with random float input. Skip if embed_output IS a graph
    # input (no separate embedding section).
    graph_input_names = {inp.name for inp in probe_graph.inputs}
    if boundaries.embed_output not in graph_input_names:
        graph, ir_version = _load_graph(input_model_path)
        _extract_subgraph(
            graph,
            graph_name="embed_scale",
            input_renames={"token_embedding": "token_embedding"},
            output_renames={boundaries.embed_output: "hidden_states"},
        )
        outputs.append(_save_subgraph(
            graph, output_dir / "embed_scale.onnx", ir_version,
            {"subgraph_type": "embed_scale"},
        ))

    # 2. Decoder block (layer 0 as representative)
    graph, ir_version = _load_graph(input_model_path)
    # Start with embed_output as a cut point (becomes a graph input)
    layer_inputs: dict[str, str] = {boundaries.embed_output: "hidden_states"}
    # Include all original graph inputs that might be needed; cleanup will prune
    for inp in graph.inputs:
        if isinstance(inp, gs.Variable) and inp.name not in layer_inputs:
            if inp.name.startswith("past_key_values.0."):
                layer_inputs[inp.name] = "past_key_value"
            elif not inp.name.startswith("past_key_values."):
                layer_inputs[inp.name] = inp.name

    layer_outputs: dict[str, str] = {
        boundaries.layer_residual_template.format(i=0): "hidden_states_out"
    }
    if boundaries.kv_output_template:
        kv_out = boundaries.kv_output_template.format(i=0)
        layer_outputs[kv_out] = "present_key_value"

    _extract_subgraph(
        graph,
        graph_name="decoder_block_00",
        input_renames=layer_inputs,
        output_renames=layer_outputs,
    )
    outputs.append(_save_subgraph(
        graph, output_dir / "decoder_block.onnx", ir_version,
        {"subgraph_type": "decoder_block", "source_layer": "0"},
    ))

    # 3. Final norm (last layer output -> norm output)
    graph, ir_version = _load_graph(input_model_path)
    # Find the correct cut point: try residual template from last_layer downward
    tensors = graph.tensors()
    last_residual = None
    for i in range(boundaries.last_layer_index, -1, -1):
        candidate = boundaries.layer_residual_template.format(i=i)
        if candidate in tensors:
            last_residual = candidate
            break
    if not last_residual:
        raise ValueError("Could not find a valid residual tensor for final norm input")
    _extract_subgraph(
        graph,
        graph_name="final_norm",
        input_renames={last_residual: "hidden_states"},
        output_renames={boundaries.final_norm_output: "hidden_states_out"},
    )
    outputs.append(_save_subgraph(
        graph, output_dir / "final_norm.onnx", ir_version,
        {"subgraph_type": "final_norm"},
    ))

    # 4. LM head (norm output -> logits)
    graph, ir_version = _load_graph(input_model_path)
    _extract_subgraph(
        graph,
        graph_name="lm_head",
        input_renames={boundaries.final_norm_output: "hidden_states"},
        output_renames={boundaries.logits_output: "logits"},
    )
    outputs.append(_save_subgraph(
        graph, output_dir / "lm_head.onnx", ir_version,
        {"subgraph_type": "lm_head"},
    ))

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract representative architectural components from a decoder-only LLM ONNX model.",
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Path to the source ONNX model",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        help="Output directory (default: source model directory)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else args.model.parent / args.model.stem
    for path in extract_representative_components(args.model, output_dir):
        print(path)


if __name__ == "__main__":
    main()
