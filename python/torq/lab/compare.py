# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pure numeric output comparison.

Returns structured results; raises nothing on mismatch.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from torq.lab.io import is_float_type

DEFAULT_COMPARISON_CONFIG = {
    "int_tol": 1,
    "int_thld": 1,
    "fp_avg_tol": 1e-2,
    "fp_max_tol": 1e-2,
    "use_abs_tol_gate": False,
    "fp_abs_tol_frac": 0.0,
    "epsilon": 1e-6,
    "allow_all_zero": False,
    "allowed_wrong": 0,
    "skip_nan_check": False,
}


@dataclass
class TensorComparison:
    """Per-tensor comparison outcome for one observed/expected output pair.

    ``num_diffs`` counts elements outside tolerance and ``max_abs_diff`` /
    ``max_rel_diff`` are the worst deviations (``max_rel_diff`` is ``None`` for
    integer/boolean tensors). ``reason`` is empty when ``passed`` is True.
    """

    index: int
    passed: bool
    size: int
    num_diffs: int
    max_abs_diff: float
    max_rel_diff: Optional[float] = None
    reason: str = ""


@dataclass
class ComparisonResult:
    """Aggregate result of :func:`compare_outputs` over every output tensor.

    ``passed`` is True only if every :class:`TensorComparison` in ``tensors``
    passed; ``reason`` joins the failing tensors' reasons.
    """

    passed: bool
    tensors: List[TensorComparison] = field(default_factory=list)
    reason: str = ""


def _compare_one(idx, observed, expected, cfg) -> TensorComparison:
    if observed.size != expected.size:
        return TensorComparison(idx, False, observed.size, observed.size, float("nan"),
                                reason=f"size differs: {observed.size} vs {expected.size}")

    is_float = is_float_type(expected.dtype)
    is_bool = np.issubdtype(expected.dtype, np.bool_)

    # NaN positions must match (float only).
    if is_float and not cfg["skip_nan_check"]:
        nan_obs = np.isnan(observed.astype(np.float32))
        nan_exp = np.isnan(expected.astype(np.float32))
        if not np.array_equal(nan_obs, nan_exp):
            return TensorComparison(idx, False, observed.size, int((nan_obs != nan_exp).sum()),
                                    float("nan"), reason="NaN positions differ")

    # Guard against an all-zero output when meaningful data is expected.
    if not cfg["allow_all_zero"] and not np.all(expected == 0) and not np.any(observed != 0):
        return TensorComparison(idx, False, observed.size, observed.size, 0.0,
                                reason="output is all zero")

    max_rel = None
    if is_bool:
        diff_mask = expected != observed
        abs_diff = diff_mask.astype(np.int64)
        num_diffs = int(np.sum(diff_mask))
        max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0
        passed = num_diffs == 0
        reason = "" if passed else f"{num_diffs}/{observed.size} boolean differences"
    else:
        abs_diff = np.abs(expected.astype(np.float32) - observed.astype(np.float32))
        max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0
        if np.issubdtype(expected.dtype, np.integer):
            diff_mask = abs_diff > cfg["int_tol"]
            num_diffs = int(np.sum(diff_mask))
            # Mirror origin/main: int_thld gates only the diffs that already
            # exceed int_tol, so int_tol is the effective tolerance and int_thld
            # is inert at its default.  Gating the raw max on int_thld (as this
            # path used to) rejected valid within-int_tol results whenever a
            # caller set int_thld below int_tol (e.g. int_thld=0, int_tol=3310).
            gated = abs_diff * diff_mask
            gated_max = float(np.max(gated)) if gated.size else 0.0
            passed = gated_max <= cfg["int_thld"] and num_diffs == 0
            reason = "" if passed else f"{num_diffs}/{observed.size} exceed int_tol {cfg['int_tol']}"
        else:
            scale = np.abs(expected.astype(np.float32)) + np.abs(observed.astype(np.float32)) + cfg["epsilon"]
            rel_diff = abs_diff / scale
            max_rel = float(np.max(rel_diff)) if rel_diff.size else 0.0
            num_diffs = int(np.sum(rel_diff > cfg["fp_avg_tol"]))
            if cfg["use_abs_tol_gate"]:
                # An element counts as "wrong" only if it exceeds BOTH the relative tolerance
                # and an absolute tolerance scaled to the tensor's dynamic range.  This keeps
                # the check sensitive to real numerical drift while ignoring near-zero
                # cancellation outputs, where small (bf16 ULP-sized) absolute error produces a
                # large relative difference.  Opt-in via use_abs_tol_gate so existing callers
                # keep the pure relative gate unchanged.
                abs_tol = cfg["fp_abs_tol_frac"] * np.max(np.abs(expected.astype(np.float32)))
                wrong = int(np.sum((rel_diff > cfg["fp_max_tol"]) & (abs_diff > abs_tol)))
            else:
                wrong = int(np.sum(rel_diff > cfg["fp_max_tol"]))
            frac = wrong / rel_diff.size if rel_diff.size else 0.0
            passed = frac <= cfg["allowed_wrong"]
            reason = "" if passed else f"{wrong}/{rel_diff.size} exceed fp_max_tol {cfg['fp_max_tol']}"

    return TensorComparison(idx, passed, observed.size, num_diffs, max_abs, max_rel, reason)


def compare_outputs(observed_outputs, expected_outputs, config=None) -> ComparisonResult:
    """Compare observed vs expected output arrays and return a structured result."""
    cfg = dict(DEFAULT_COMPARISON_CONFIG)
    if config:
        cfg.update(config)

    if len(observed_outputs) != len(expected_outputs):
        return ComparisonResult(
            passed=False,
            reason=f"number of outputs differ: {len(observed_outputs)} vs {len(expected_outputs)}",
        )

    tensors = [
        _compare_one(idx, obs, exp, cfg)
        for idx, (obs, exp) in enumerate(zip(observed_outputs, expected_outputs))
    ]
    passed = all(t.passed for t in tensors)
    reason = "" if passed else "; ".join(t.reason for t in tensors if not t.passed)
    return ComparisonResult(passed=passed, tensors=tensors, reason=reason)
