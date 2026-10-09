# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Hugging Face transformers support for the Torq torch model test path.

This module contains the pieces specific to *loading* models from the
Hugging Face Hub (``transformers`` library) for the layer-test pipeline.
The argument tracing that turns a model into layer test cases is generic to
any ``torch.nn.Module`` and lives in :mod:`torq.testing.torch`:

  * ``trace_torch_model_calls`` -- forward-hook trace that records the
    *full* call signature of every named module: tensor arguments (shape +
    dtype), keyword arguments by name, and non-tensor arguments (ints,
    tuples, lists, ``None``, ...) as recorded constants.  Hugging Face
    models call submodules with keyword arguments and with non-tensor
    arguments (e.g. RT-DETR passes ``spatial_shapes_list`` as a Python list
    of tuples), which a positional shape-only trace cannot reconstruct.
  * ``build_traced_torch_cases`` -- turns such a trace into standalone layer
    test cases.
  * ``generate_traced_torch_model_layers`` -- the metadata-cache generator:
    one pytest-xdist worker builds the model and traces it, the others read
    the manifest.  Used by both ``tests/test_hf_models.py`` and
    ``tests/test_torch_zoo_turbine.py``.

What is specific to Hugging Face transformers here:

  1. ``make_hf_model_loader`` -- a lazy model loader that pulls the model
     from the Hub (weights cached by the transformers library in
     ``HF_HOME``) and normalizes it for export: ``eval()`` mode, the target
     dtype (bf16), no in-place ops, and eager attention so the traced graph
     uses the explicit matmul/softmax attention formula that the compiler
     lowers.
  2. ``_demote_sinusoidal_position_embedding_to_f32`` -- RT-DETR's AIFI
     sinusoidal position embedding computes in float64 upstream, which torq
     cannot lower; this swaps in an f32 builder.
"""

import torch

from .torch import (
    _disable_inplace_ops,
    build_traced_torch_cases,
    generate_traced_torch_model_layers,
    trace_torch_model_calls,
)

# Backwards-compatible aliases for callers written against the original
# names, when this machinery was still HF-specific.  The mechanism itself is
# generic; see torq.testing.torch.
build_hf_model_cases = build_traced_torch_cases
generate_hf_model_layers = generate_traced_torch_model_layers
trace_module_calls = trace_torch_model_calls

# transformers Auto classes by task.  Keep this table small; add entries as
# new model families are onboarded.
_HF_AUTO_CLASSES = {
    "image_classification": "AutoModelForImageClassification",
    "object_detection": "AutoModelForObjectDetection",
    "segmentation": "AutoModelForSemanticSegmentation",
    "mask_generation": "AutoModelForMaskGeneration",
    "causal_lm": "AutoModelForCausalLM",
    "sequence_classification": "AutoModelForSequenceClassification",
    "base": "AutoModel",
}


def make_hf_model_loader(
    repo_id,
    task,
    dtype=torch.bfloat16,
    wrapper=None,
    load_kwargs=None,
):
    """Build a lazy loader for a Hugging Face Hub model.

    Args:
        repo_id: Hugging Face repo, e.g. "PekingU/rtdetr_r50vd".
        task: One of ``_HF_AUTO_CLASSES`` (selects the Auto model class).
        dtype: Weight dtype the model is loaded in (bf16 for torq).
        wrapper: Optional ``nn.Module`` subclass taking the loaded model in
            its ``__init__``.  Hugging Face heads return ``ModelOutput``
            (a dict) which the reference executor cannot flatten
            deterministically, so detection/classification heads usually
            need a small wrapper returning plain tensors.
        load_kwargs: Extra kwargs for ``from_pretrained``.

    The loader disables in-place ops, pins the attention implementation to
    ``eager`` (the eager interface traces the explicit matmul/scale/softmax
    attention formula, which both the reference executor and the compiler see
    identically, and avoids SDPA submodules in conv-only backbones rejecting
    the sdpa implementation), and demotes float64 position-embedding math to
    float32 (torq has no f64 support).
    """
    if task not in _HF_AUTO_CLASSES:
        raise ValueError(
            f"Unknown HF task {task!r}; expected one of {sorted(_HF_AUTO_CLASSES)}"
        )
    auto_cls_name = _HF_AUTO_CLASSES[task]

    def _loader():
        import transformers

        auto_cls = getattr(transformers, auto_cls_name)
        model = auto_cls.from_pretrained(repo_id, dtype=dtype, **(load_kwargs or {}))
        model.eval()
        _disable_inplace_ops(model)
        # torq has no f64; some HF models compute position embeddings in f64.
        _demote_sinusoidal_position_embedding_to_f32(model)
        model.config._attn_implementation = "eager"
        if wrapper is not None:
            model = wrapper(model)
        return model

    return _loader


def _demote_sinusoidal_position_embedding_to_f32(model):
    """Force RT-DETR's 2D sinusoidal position embedding to compute in f32.

    ``transformers`` builds the AIFI sinusoidal position embedding with
    ``compute_dtype = torch.float64`` (arange / meshgrid / outer / sin / cos in
    f64, cast to the model dtype at the end).  torq has no f64 support and the
    torch->linalg type conversion crashes on f64 rank-changing ops, so we swap
    the module-level builder for an f32 version.  The result is immediately
    cast to the model dtype (bf16), so f32 intermediate precision is more than
    sufficient and the exported graph is bit-for-bit comparable.
    """
    import importlib

    import torch

    def _f32_build(height, width, embed_dim=256, temperature=10000.0,
                   cls_token=False, device=None, dtype=torch.float32):
        if embed_dim % 4 != 0:
            raise ValueError(f"`embed_dim` must be divisible by 4, got {embed_dim}")
        pos_dim = embed_dim // 4
        compute_dtype = torch.float32
        omega = torch.arange(pos_dim, dtype=compute_dtype, device=device) / pos_dim
        omega = 1.0 / temperature ** omega
        grid_h = torch.arange(height, dtype=compute_dtype, device=device)
        grid_w = torch.arange(width, dtype=compute_dtype, device=device)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
        emb_h = grid_h.flatten().outer(omega)
        emb_w = grid_w.flatten().outer(omega)
        pos_embed = torch.cat(
            [emb_h.sin(), emb_h.cos(), emb_w.sin(), emb_w.cos()], dim=1
        )
        if cls_token:
            pos_embed = torch.cat(
                [torch.zeros(1, embed_dim, dtype=compute_dtype, device=device),
                 pos_embed],
                dim=0,
            )
        return pos_embed.to(dtype)

    cls = type(model)
    module = importlib.import_module(cls.__module__)
    original = getattr(module, "build_2d_sinusoidal_position_embedding", None)
    if original is None:
        return  # not an RT-DETR-style model; nothing to patch
    module.build_2d_sinusoidal_position_embedding = _f32_build
    # Drop any cached (f64) results so the patched builder is actually used.
    sine = getattr(module, "RTDetrSinePositionEmbedding", None)
    cached = getattr(sine, "_cached_build_2d_sinusoidal_position_embedding", None)
    cache_clear = getattr(cached, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
