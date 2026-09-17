# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Per-layer quantization sensitivity analysis (model-agnostic).

Uses teacher-forced autoregressive evaluation: for each MatMul weight layer,
the weight is quantised → dequantised at a given bit-width, the model is
run step-by-step feeding baseline tokens, and output logit divergence is
measured against the fp32 baseline.

Auto-detects model architecture (KV cache shape, input names, number of
layers) from the ONNX graph.  Chat template is configurable via a format
string with ``{system}`` and ``{user}`` placeholders.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from torq.lab.logging import add_logging_args, configure_logging
from torq.lab.verification.compare import (
    classify_severity,
    cosine_similarity,
    kl_divergence,
)
from ._config import SensitivityResult, SensitivityResults
from ._quantization import dequantize_weight, quantize_weight

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bf16_roundtrip(weight: np.ndarray) -> np.ndarray:
    """Truncate fp32 weight to bf16 precision and back to fp32."""
    u32 = weight.view(np.uint32)
    u16 = ((u32 + 0x8000) >> 16).astype(np.uint16)
    fp32 = np.zeros(weight.shape, dtype=np.float32)
    fp32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return fp32


# Default chat template (Gemma-3 format).
# Use {system} and {user} placeholders.
_DEFAULT_CHAT_TEMPLATE = (
    "<start_of_turn>system\n{system}<end_of_turn>\n"
    "<start_of_turn>user\n{user}<end_of_turn>\n"
    "<start_of_turn>model\n"
)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer in 1-2 sentences."
)

_DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "How does photosynthesis work in plants?",
    "What is the speed of light in meters per second?",
    "Who wrote Romeo and Juliet?",
    "What is the largest planet in our solar system?",
    "How does a computer processor execute instructions?",
    "What causes earthquakes to occur?",
    "How does the water cycle work in nature?",
    "What is Einstein's theory of relativity about?",
    "How do vaccines help prevent infectious diseases?",
    "What is machine learning and how does it work?",
    "How does the law of supply and demand affect prices?",
    "What are the main layers that make up the Earth?",
    "How does electricity flow through a circuit?",
    "What is DNA and what role does it play in living organisms?",
    "How do airplanes generate enough lift to fly?",
    "What is the greenhouse effect and why does it matter?",
    "How does the human immune system fight infections?",
    "What are prime numbers and why are they important in mathematics?",
    "How does a neural network learn from training data?",
]


# ---------------------------------------------------------------------------
# Model architecture auto-detection
# ---------------------------------------------------------------------------
# NOTE: The inference loop below intentionally does NOT reuse the model-specific
# runners (e.g. SmolLM2Base, Gemma3Static). Those require model config files and
# are tied to a specific architecture. The sensitivity analyzer must work with
# *any* static decoder ONNX model using only the session I/O metadata.
# ---------------------------------------------------------------------------


def _detect_model_arch(sess) -> dict:
    """Auto-detect model architecture from an ORT session.

    Returns a dict with:
        embedding_input : name of the embedding/token input
        position_input  : name of the position_ids input (or None)
        kv_inputs       : dict mapping kv input name → shape
        kv_output_map   : dict mapping kv output name → kv input name
        num_layers      : number of transformer layers
        eos_token_ids   : list of token IDs that stop generation
    """
    inputs = {i.name: i for i in sess.get_inputs()}
    outputs = {o.name: o for o in sess.get_outputs()}

    # Find embedding input (float, 3D: [batch, seq, hidden])
    embedding_input = None
    position_input = None
    kv_inputs = {}

    for name, inp in inputs.items():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        dtype_str = inp.type.replace("tensor(", "").replace(")", "")

        if any(k in name.lower() for k in ("embed", "token_embed", "input_embed")):
            embedding_input = name
        elif "position" in name.lower():
            position_input = name
        elif any(k in name.lower() for k in ("past", "cache", "kv")):
            kv_inputs[name] = shape

    # If no explicit embedding input found, look for a 3D float input
    if embedding_input is None:
        for name, inp in inputs.items():
            shape = inp.shape
            dtype_str = inp.type.replace("tensor(", "").replace(")", "")
            if len(shape) == 3 and dtype_str in ("float", "float16", "bfloat16"):
                if name not in kv_inputs:
                    embedding_input = name
                    break

    # Map KV outputs back to inputs
    # Common patterns: "present.0.key_value" → "past_key_values.0.key_value"
    kv_output_map = {}
    for out_name in outputs:
        out_lower = out_name.lower()
        if not any(k in out_lower for k in ("present", "cache", "kv", "past")):
            continue
        # Try common renaming patterns
        for pattern, replacement in [
            ("present.", "past_key_values."),
            ("present_", "past_"),
            ("new_", ""),
        ]:
            candidate = out_name.replace(pattern, replacement)
            if candidate in kv_inputs:
                kv_output_map[out_name] = candidate
                break
        else:
            # Try matching by layer index
            m = re.search(r"(\d+)", out_name)
            if m:
                idx = m.group(1)
                for kv_name in kv_inputs:
                    if idx in kv_name and kv_name not in kv_output_map.values():
                        kv_output_map[out_name] = kv_name
                        break

    num_layers = len(kv_inputs)

    logger.info(
        "Auto-detected model: embedding=%s, position=%s, %d KV layers",
        embedding_input, position_input, num_layers,
    )

    return {
        "embedding_input": embedding_input,
        "position_input": position_input,
        "kv_inputs": kv_inputs,
        "kv_output_map": kv_output_map,
        "num_layers": num_layers,
    }


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


class LayerSensitivityAnalyzer:
    """Analyse per-layer quantization sensitivity of an ONNX model.

    Model-agnostic: auto-detects architecture from ONNX graph.

    Parameters
    ----------
    model_path : path to the fp32/bf16 ONNX model
    embeddings_path : path to token_embeddings.npy
    tokenizer_path : path to tokenizer.json
    token_lut_path : path to token_id_lut.npy (for reduced-vocab models)
    calibration_prompts : list of text prompts; ``None`` → use defaults
    pre_tokenized_file : path to JSON file with pre-tokenized token ID lists
                         (skips tokenization entirely)
    chat_template : format string with {system} and {user} placeholders
    system_prompt : system prompt text
    eos_token_ids : list of token IDs that stop generation
    num_tokens : number of generation tokens to evaluate per prompt
    skip_layers : layer-name substrings to skip
    """

    def __init__(
        self,
        model_path: str | Path,
        embeddings_path: str | Path,
        tokenizer_path: str | Path | None = None,
        token_lut_path: str | Path | None = None,
        calibration_prompts: list[str] | None = None,
        pre_tokenized_file: str | Path | None = None,
        chat_template: str | None = None,
        system_prompt: str | None = None,
        eos_token_ids: list[int] | None = None,
        num_tokens: int = 20,
        skip_layers: list[str] | None = None,
    ):
        self.model_path = Path(model_path)
        self.embeddings_path = Path(embeddings_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        self.token_lut_path = Path(token_lut_path) if token_lut_path else None
        self.prompts = calibration_prompts or _DEFAULT_PROMPTS
        self.pre_tokenized_file = (
            Path(pre_tokenized_file) if pre_tokenized_file else None
        )
        self.chat_template = chat_template or _DEFAULT_CHAT_TEMPLATE
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self.eos_token_ids = set(eos_token_ids) if eos_token_ids else None
        self.num_tokens = num_tokens
        self.skip_layers = skip_layers or []

        self._embeddings: np.ndarray | None = None
        self._token_lut: np.ndarray | None = None
        self._reverse_lut: dict[int, int] | None = None
        self._tokenizer = None
        self._arch: dict | None = None
        self._detected_eos: set[int] | None = None

    # --- public API ----------------------------------------------------------

    def analyze(
        self,
        bits_options: list[int] | None = None,
        output_path: str | Path | None = None,
        config_output_path: str | Path | None = None,
        bf16_threshold: float = 0.1,
        int8_threshold: float = 0.01,
    ) -> SensitivityResults:
        """Run sensitivity analysis."""
        bits_options = bits_options or [4, 8, 16]
        import onnxruntime as ort

        # Load embeddings
        emb = np.load(str(self.embeddings_path))
        if emb.dtype != np.float32:
            u16 = emb.view(np.uint16)
            fp32 = np.zeros(u16.shape, dtype=np.float32)
            fp32.view(np.uint32)[...] = u16.astype(np.uint32) << 16
            emb = fp32
        self._embeddings = emb

        # Load token LUT for reduced-vocab models
        if self.token_lut_path and self.token_lut_path.exists():
            self._token_lut = np.load(str(self.token_lut_path)).astype(np.int64)
            self._reverse_lut = {
                int(full_id): idx for idx, full_id in enumerate(self._token_lut)
            }
            logger.info(
                "Loaded token LUT: %d reduced vocab entries", len(self._token_lut)
            )

        # Tokenize prompts
        token_sequences = self._tokenize_prompts()
        logger.info("Tokenized %d prompts", len(token_sequences))

        # Load and convert model to fp32 for CPU inference
        logger.info("Loading model %s", self.model_path)
        model_proto = onnx.load(str(self.model_path), load_external_data=True)
        fp32_model = self._convert_to_fp32(model_proto)

        # Create baseline session
        logger.info("Creating baseline session...")
        base_tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx.save(fp32_model, base_tmp.name)
        base_tmp.close()

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 4
        base_sess = ort.InferenceSession(
            base_tmp.name, opts, providers=["CPUExecutionProvider"]
        )

        # Auto-detect model architecture
        self._arch = _detect_model_arch(base_sess)

        # Detect EOS tokens from tokenizer if not provided
        if self.eos_token_ids is not None:
            self._detected_eos = self.eos_token_ids
        else:
            self._detected_eos = self._detect_eos_tokens()
            logger.info("EOS token IDs: %s", self._detected_eos)

        # Collect baseline logits (teacher-forced)
        logger.info("Collecting baseline logits...")
        baseline_all = []
        for i, ids in enumerate(token_sequences):
            t0 = time.time()
            bl = self._collect_logits(base_sess, ids, self.num_tokens)
            elapsed = (time.time() - t0) * 1000
            tokens_text = self._decode_tokens([t for t, _ in bl])
            logger.info(
                '  Prompt %d: %.0fms, %d tokens: "%s"',
                i + 1, elapsed, len(bl), tokens_text[:60],
            )
            baseline_all.append(bl)
        del base_sess

        # Get baseline reference tokens for teacher forcing
        ref_tokens_all = [[t for t, _ in bl] for bl in baseline_all]

        # Find MatMul weight layers
        init_map = {i.name: i for i in fp32_model.graph.initializer}
        fp32_init_idx = {
            i.name: idx for idx, i in enumerate(fp32_model.graph.initializer)
        }
        layers = []
        for node in fp32_model.graph.node:
            if node.op_type != "MatMul":
                continue
            if any(s in node.name for s in self.skip_layers):
                continue
            for inp in node.input:
                if inp in init_map:
                    init = init_map[inp]
                    if len(init.dims) == 2:
                        layers.append((node.name, inp, fp32_init_idx[inp]))
                        break

        logger.info("Found %d MatMul layers to test", len(layers))

        # Per-layer sensitivity testing
        results = SensitivityResults()
        for layer_idx, (node_name, weight_name, init_idx) in enumerate(layers):
            original_init = fp32_model.graph.initializer[init_idx]
            original_data = original_init.raw_data
            fp32_w = numpy_helper.to_array(original_init).astype(np.float32)

            layer_kl = {}
            layer_cos = {}
            layer_top1 = {}

            for bits in bits_options:
                logger.info(
                    "[%d/%d] Testing %s at %d-bit",
                    layer_idx + 1, len(layers), node_name, bits,
                )

                if bits == 16:
                    deq_w = _bf16_roundtrip(fp32_w)
                else:
                    qw = quantize_weight(fp32_w, bits)
                    deq_w = dequantize_weight(qw)

                new_init = numpy_helper.from_array(deq_w, name=weight_name)
                fp32_model.graph.initializer[init_idx].CopyFrom(new_init)

                onnx.save(fp32_model, base_tmp.name)
                mod_sess = ort.InferenceSession(
                    base_tmp.name, opts, providers=["CPUExecutionProvider"]
                )

                kl_divs = []
                cos_sims = []
                top1_matches = []
                for prompt_idx, ids in enumerate(token_sequences):
                    ref_tokens = ref_tokens_all[prompt_idx]
                    mod_results = self._collect_logits(
                        mod_sess, ids, self.num_tokens, ref_tokens=ref_tokens,
                    )

                    bl_results = baseline_all[prompt_idx]
                    for step_idx in range(min(len(bl_results), len(mod_results))):
                        bl_logits = bl_results[step_idx][1]
                        mod_logits = mod_results[step_idx][1]
                        kl_divs.append(kl_divergence(bl_logits, mod_logits))
                        cos_sims.append(cosine_similarity(bl_logits, mod_logits))
                        top1_matches.append(
                            1.0
                            if bl_logits.argmax() == mod_logits.argmax()
                            else 0.0
                        )

                del mod_sess

                mean_kl = float(np.mean(kl_divs)) if kl_divs else float("inf")
                mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0
                mean_top1 = float(np.mean(top1_matches)) if top1_matches else 0.0

                layer_kl[bits] = mean_kl
                layer_cos[bits] = mean_cos
                layer_top1[bits] = mean_top1

                sev = classify_severity(mean_kl)
                logger.info(
                    "  %d-bit: KL=%.10f cos=%.6f top1=%.2f [%s]",
                    bits, mean_kl, mean_cos, mean_top1, sev,
                )

                # Restore original weight
                original_init_restored = onnx.TensorProto()
                original_init_restored.name = weight_name
                original_init_restored.dims[:] = list(fp32_w.shape)
                original_init_restored.data_type = TensorProto.FLOAT
                original_init_restored.raw_data = original_data
                fp32_model.graph.initializer[init_idx].CopyFrom(
                    original_init_restored
                )

            worst_kl = max(layer_kl.values()) if layer_kl else 0.0
            result = SensitivityResult(
                layer_name=node_name,
                kl_divergence=layer_kl,
                cosine_similarity=layer_cos,
                top1_match=layer_top1,
                classification=classify_severity(worst_kl),
            )
            results.layers.append(result)

            parts = [f"{b}bit={layer_kl[b]:.10f}" for b in sorted(layer_kl)]
            logger.info(
                "  → %s: %s [%s]",
                node_name, ", ".join(parts), result.classification,
            )

        # Cleanup
        os.unlink(base_tmp.name)

        # Summary
        classifications = {}
        for r in results.layers:
            classifications[r.classification] = (
                classifications.get(r.classification, 0) + 1
            )
        logger.info("Sensitivity summary: %s", classifications)

        if output_path:
            results.save(output_path)
        if config_output_path:
            config = results.to_config(
                bf16_threshold=bf16_threshold,
                int8_threshold=int8_threshold,
            )
            config.save(config_output_path)
            logger.info("Saved quantization config to %s", config_output_path)

        return results

    # --- internals -----------------------------------------------------------

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if self.tokenizer_path and self.tokenizer_path.exists():
            try:
                from tokenizers import Tokenizer

                self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
                return self._tokenizer
            except ImportError:
                logger.warning("tokenizers not installed")
        return None

    def _detect_eos_tokens(self) -> set[int]:
        """Auto-detect EOS/stop token IDs from the tokenizer."""
        tok = self._load_tokenizer()
        eos_ids = set()
        if tok is None:
            return {1, 2}  # common defaults

        # Check tokenizer special tokens
        model_json = tok.to_str()
        model_data = json.loads(model_json)

        # Look for added_tokens with special=true that look like EOS/end
        added = model_data.get("added_tokens", [])
        for t in added:
            if not t.get("special", False):
                continue
            content = t.get("content", "").lower()
            tid = t.get("id")
            if tid is not None and any(
                k in content
                for k in ("</s>", "<eos>", "end_of_turn", "<|end|>", "<|eot_id|>")
            ):
                eos_ids.add(tid)

        if not eos_ids:
            eos_ids = {1}  # fallback: common EOS

        return eos_ids

    def _tokenize_prompts(self) -> list[list[int]]:
        """Tokenise calibration prompts.

        Supports:
        1. Pre-tokenized file (JSON list of int lists) — skips tokenization
        2. Chat template with {system}/{user} placeholders
        3. Plain text tokenization (no template)
        """
        # Option 1: Pre-tokenized file
        if self.pre_tokenized_file and self.pre_tokenized_file.exists():
            data = json.loads(self.pre_tokenized_file.read_text())
            logger.info("Loaded %d pre-tokenized sequences", len(data))
            return data

        # Need tokenizer for options 2 and 3
        tok = self._load_tokenizer()
        if tok is None:
            raise RuntimeError(
                "Tokenizer is required for sensitivity analysis. "
                "Provide --tokenizer or --pre-tokenized-file"
            )

        all_ids = []
        for prompt in self.prompts:
            # Apply chat template
            text = self.chat_template.format(
                system=self.system_prompt,
                user=prompt,
            )
            ids = tok.encode(text).ids
            all_ids.append(ids)
            logger.debug("Prompt tokenized: %d tokens", len(ids))

        return all_ids

    def _decode_tokens(self, token_ids: list[int]) -> str:
        tok = self._load_tokenizer()
        if tok:
            return tok.decode(token_ids)
        return str(token_ids)

    def _reduced_to_full(self, reduced_idx: int) -> int:
        """Map reduced-vocab index to full-vocab token ID."""
        if self._token_lut is not None:
            return int(self._token_lut[reduced_idx])
        return reduced_idx

    def _full_to_reduced(self, full_id: int) -> int:
        """Map full-vocab token ID to reduced-vocab index."""
        if self._reverse_lut is not None:
            if full_id in self._reverse_lut:
                return self._reverse_lut[full_id]
            # Token not in reduced vocab — fall back to UNK (index 0)
            logger.debug("Token ID %d not in reduced vocab, using index 0", full_id)
            return 0
        return full_id

    def _collect_logits(
        self,
        sess,
        prompt_ids: list[int],
        n_gen: int,
        ref_tokens: list[int] | None = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Run autoregressive inference, returning (full_token_id, logits) per step."""
        arch = self._arch
        emb = self._embeddings
        eos_ids = self._detected_eos or set()

        # Initialize KV cache from model inputs
        kv = {}
        for name, shape in arch["kv_inputs"].items():
            kv[name] = np.zeros(shape, dtype=np.float32)

        out_names = [o.name for o in sess.get_outputs()]

        pos = 0
        logits = None

        # Process prompt tokens
        for tok in prompt_ids:
            # Map full vocab ID to reduced vocab index for embedding lookup
            reduced_tok = self._full_to_reduced(tok)
            logits = self._step(sess, emb, kv, reduced_tok, pos, out_names, arch)
            pos += 1

        # Generate tokens
        results = []
        reduced_idx = int(logits.argmax())
        full_tok = self._reduced_to_full(reduced_idx)
        results.append((full_tok, logits.copy()))

        for i in range(n_gen - 1):
            feed_tok = (
                ref_tokens[i] if ref_tokens and i < len(ref_tokens) else full_tok
            )
            if feed_tok in eos_ids:
                break
            # Map full vocab ID to reduced index for embedding lookup
            reduced_feed = self._full_to_reduced(feed_tok)
            logits = self._step(sess, emb, kv, reduced_feed, pos, out_names, arch)
            pos += 1
            reduced_idx = int(logits.argmax())
            full_tok = self._reduced_to_full(reduced_idx)
            results.append((full_tok, logits.copy()))

        return results

    @staticmethod
    def _step(sess, emb, kv, token_id, pos, out_names, arch):
        """Run a single autoregressive step."""
        e = emb[token_id].astype(np.float32).reshape(1, 1, -1)
        feeds = {}

        # Embedding input
        emb_name = arch["embedding_input"]
        feeds[emb_name] = e

        # Position input (if present)
        if arch["position_input"]:
            pos_name = arch["position_input"]
            # Detect dtype from session inputs
            for inp in sess.get_inputs():
                if inp.name == pos_name:
                    dtype_str = inp.type.replace("tensor(", "").replace(")", "")
                    if dtype_str == "int32":
                        feeds[pos_name] = np.array([[pos]], dtype=np.int32)
                    else:
                        feeds[pos_name] = np.array([[pos]], dtype=np.int64)
                    break

        # KV cache inputs
        feeds.update(kv)

        outs = sess.run(None, feeds)

        # Update KV cache using output→input mapping
        kv_map = arch["kv_output_map"]
        for i, name in enumerate(out_names):
            if name in kv_map:
                kv[kv_map[name]] = outs[i]

        return outs[0][0, -1]  # logits for last position

    @staticmethod
    def _convert_to_fp32(model_proto: onnx.ModelProto) -> onnx.ModelProto:
        """Convert a bf16 model to fp32 for CPU inference with ORT."""
        new_inits = []
        for init in model_proto.graph.initializer:
            if init.data_type == TensorProto.BFLOAT16:
                dims = list(init.dims)
                n = 1
                for d in dims:
                    n *= d
                raw = init.raw_data
                if len(raw) == n * 2:
                    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(dims)
                    fp32 = np.zeros(dims, dtype=np.float32)
                    fp32.view(np.uint32)[...] = u16.astype(np.uint32) << 16
                else:
                    fp32 = np.frombuffer(raw, dtype=np.float32).reshape(dims)
                new_inits.append(numpy_helper.from_array(fp32, name=init.name))
            else:
                new_inits.append(init)

        new_inputs = []
        for inp in model_proto.graph.input:
            t = inp.type.tensor_type
            if t.elem_type in (TensorProto.BFLOAT16, TensorProto.FLOAT16):
                shape = [
                    d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in t.shape.dim
                ]
                new_inputs.append(
                    helper.make_tensor_value_info(
                        inp.name, TensorProto.FLOAT, shape
                    )
                )
            else:
                new_inputs.append(inp)

        new_outputs = []
        for out in model_proto.graph.output:
            t = out.type.tensor_type
            if t.elem_type in (TensorProto.BFLOAT16, TensorProto.FLOAT16):
                shape = [
                    d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in t.shape.dim
                ]
                new_outputs.append(
                    helper.make_tensor_value_info(
                        out.name, TensorProto.FLOAT, shape
                    )
                )
            else:
                new_outputs.append(out)

        new_nodes = []
        for node in model_proto.graph.node:
            if node.op_type == "Cast":
                new_attrs = []
                for attr in node.attribute:
                    if attr.name == "to" and attr.i in (
                        TensorProto.BFLOAT16,
                        TensorProto.FLOAT16,
                    ):
                        new_attr = onnx.AttributeProto()
                        new_attr.name = "to"
                        new_attr.type = onnx.AttributeProto.INT
                        new_attr.i = TensorProto.FLOAT
                        new_attrs.append(new_attr)
                    else:
                        new_attrs.append(attr)
                nn = helper.make_node(
                    node.op_type,
                    list(node.input),
                    list(node.output),
                    name=node.name,
                )
                nn.attribute.extend(new_attrs)
                new_nodes.append(nn)
            else:
                new_nodes.append(node)

        fp32_graph = helper.make_graph(
            new_nodes, model_proto.graph.name, new_inputs, new_outputs, new_inits
        )
        fp32_model = helper.make_model(fp32_graph)
        fp32_model.ir_version = model_proto.ir_version
        del fp32_model.opset_import[:]
        for op in model_proto.opset_import:
            fp32_model.opset_import.append(op)

        return fp32_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_weights_analyze_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input fp32 ONNX model path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output sensitivity results JSON path",
    )
    parser.add_argument(
        "--config-output",
        type=str,
        default=None,
        help="Output quantization config JSON path (derived from sensitivity)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer.json for prompt tokenization",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to token_embeddings.npy for embedding lookup (required)",
    )
    parser.add_argument(
        "--token-lut",
        type=str,
        default=None,
        help="Path to token_id_lut.npy for reduced-vocab models (maps reduced index → full vocab ID)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="Bit-widths to test (default: 4 8 16)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=20,
        help="Number of output tokens to evaluate per prompt (default: %(default)s)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default=None,
        help="Calibration prompts (text strings). If not provided, uses defaults.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="JSON file with list of calibration prompt strings",
    )
    parser.add_argument(
        "--pre-tokenized-file",
        type=str,
        default=None,
        help="JSON file with pre-tokenized token ID lists (skips tokenization)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Chat template format string with {system} and {user} placeholders",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt text for chat template",
    )
    parser.add_argument(
        "--eos-token-ids",
        type=int,
        nargs="*",
        default=None,
        help="Token IDs that stop generation (auto-detected from tokenizer if omitted)",
    )
    parser.add_argument(
        "--bf16-threshold",
        type=float,
        default=0.1,
        help="KL divergence threshold above which layers stay bf16 (default: %(default)s)",
    )
    parser.add_argument(
        "--int8-threshold",
        type=float,
        default=0.01,
        help="KL divergence threshold above which layers use int8 (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-layers",
        type=str,
        nargs="*",
        default=[],
        help="Layer name substrings to skip (e.g. lm_head)",
    )
    add_logging_args(parser)


def weights_analyze_from_args(args: argparse.Namespace) -> None:
    configure_logging(args.logging)

    prompts = args.prompts
    if args.prompts_file:
        prompts = json.loads(open(args.prompts_file).read())

    analyzer = LayerSensitivityAnalyzer(
        model_path=args.input,
        embeddings_path=args.embeddings,
        tokenizer_path=args.tokenizer,
        token_lut_path=args.token_lut,
        calibration_prompts=prompts,
        pre_tokenized_file=args.pre_tokenized_file,
        chat_template=args.chat_template,
        system_prompt=args.system_prompt,
        eos_token_ids=args.eos_token_ids,
        num_tokens=args.num_tokens,
        skip_layers=args.skip_layers,
    )
    analyzer.analyze(
        bits_options=args.bits,
        output_path=args.output,
        config_output_path=args.config_output,
        bf16_threshold=args.bf16_threshold,
        int8_threshold=args.int8_threshold,
    )
