# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Shared primitives for per-node ONNX quantization sensitivity analysis.

Used by the ``dynamic`` and ``static`` mode analyzers: building calibration
feeds, running ORT sessions, comparing quantized outputs against the fp32
baseline, and writing the JSON report / exclude-list artifacts.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import onnx

from torq.lab.verification.compare import cosine_similarity, kl_divergence

logger = logging.getLogger(__name__)

# Severity labels ordered by increasing sensitivity; used to threshold
# exclude lists (a node joins the list when its classification is at/above
# the requested severity).
SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _random_array(rng: np.random.Generator, shape: list[int], np_dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(np_dtype, np.floating):
        return rng.standard_normal(shape).astype(np_dtype)
    if np.issubdtype(np_dtype, np.integer):
        # 0/1 indices are safe for gathers/embeddings regardless of vocab size.
        return rng.integers(0, 2, size=shape).astype(np_dtype)
    if np_dtype == np.bool_:
        return rng.integers(0, 2, size=shape).astype(np.bool_)
    return np.zeros(shape, dtype=np_dtype)


def build_random_feeds(
    model: onnx.ModelProto, seed: int, dynamic_dim: int = 1
) -> dict[str, np.ndarray]:
    """Build seeded random feeds for every graph input, using ``dynamic_dim`` for unknown dims."""
    rng = np.random.default_rng(seed)
    init_names = {i.name for i in model.graph.initializer}
    feeds: dict[str, np.ndarray] = {}
    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        ttype = inp.type.tensor_type
        shape = [
            d.dim_value if (d.HasField("dim_value") and d.dim_value > 0) else dynamic_dim
            for d in ttype.shape.dim
        ]
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(ttype.elem_type)
        feeds[inp.name] = _random_array(rng, shape, np_dtype)
    return feeds


def run_session(sess, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out_names = [o.name for o in sess.get_outputs()]
    return dict(zip(out_names, sess.run(out_names, dict(feeds))))


def compare_outputs(
    base_out: dict[str, np.ndarray], quant_out: dict[str, np.ndarray]
) -> tuple[float, float, float]:
    """Reduce a per-output comparison to (worst KL, worst cosine, worst max-abs-error)."""
    kls: list[float] = []
    coss: list[float] = []
    errs: list[float] = []
    for name, b in base_out.items():
        q = quant_out.get(name)
        if q is None:
            continue
        b = np.asarray(b, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        coss.append(cosine_similarity(b.ravel(), q.ravel()))
        errs.append(float(np.max(np.abs(b - q))) if b.size else 0.0)
        if b.ndim >= 1 and b.shape[-1] > 1:
            bb = b.reshape(-1, b.shape[-1])
            qq = q.reshape(-1, q.shape[-1])
            row_kls = [kl_divergence(bb[i], qq[i]) for i in range(bb.shape[0])]
            kls.append(float(np.mean(row_kls)) if row_kls else 0.0)
    return (
        max(kls) if kls else 0.0,
        min(coss) if coss else 1.0,
        max(errs) if errs else 0.0,
    )


def write_report(results: list[dict], path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


def write_exclude_list(
    results: list[dict], path: str | os.PathLike, exclude_class: str
) -> int:
    """Write nodes classified at/above *exclude_class* as a JSON node-name list.

    Returns the number of excluded nodes.
    """
    cutoff = SEVERITY_ORDER[exclude_class]
    exclude = [r["node"] for r in results if SEVERITY_ORDER[r["classification"]] >= cutoff]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(exclude, indent=2))
    return len(exclude)
