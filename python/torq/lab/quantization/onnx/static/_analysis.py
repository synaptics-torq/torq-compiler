# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Per-node sensitivity analysis for ONNX static quantization.

For each candidate node, statically quantize *only* that node (calibrating on
the analysis feeds), run the model against the same feeds, and measure how far
the outputs drift from the fp32 baseline (KL divergence, cosine similarity,
max absolute error). Nodes are classified by severity so the most damaging
ones can be excluded from quantization via ``quantize static --exclude-nodes``.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader

from torq.lab.logging import add_logging_args, configure_logging
from torq.lab.verification.compare import classify_severity
from torq.lab.utils.onnxruntime import make_cpu_session

from .._analysis import (
    compare_outputs,
    run_session,
    write_exclude_list,
    write_report,
)
from ._quantization import (
    _DATASET_NOT_IMPLEMENTED_MSG,
    RandomCalibrationDataReader,
    add_static_quant_flags,
    get_input_specs,
    is_model_quantized,
    onnx_static_quantize,
)

logger = logging.getLogger(__name__)


class _ListCalibrationDataReader(CalibrationDataReader):
    """One-shot ORT calibration reader over an in-memory list of feed dicts."""

    def __init__(self, samples: list[dict[str, np.ndarray]]):
        self._samples = list(samples)
        self._index = 0

    def get_next(self):
        if self._index >= len(self._samples):
            return None
        sample = self._samples[self._index]
        self._index += 1
        return sample

    def reset(self):
        self._index = 0


def _collect_feeds(
    input_specs,
    *,
    calibration_data: dict[str, np.ndarray] | None,
    num_calib: int,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    """Build the analysis feeds, which double as ORT calibration samples."""
    if calibration_data is not None:
        return [calibration_data]
    reader = RandomCalibrationDataReader(input_specs, num_samples=num_calib, seed=seed)
    feeds = []
    while (sample := reader.get_next()) is not None:
        feeds.append(sample)
    if not feeds:
        raise ValueError("No calibration feeds available")
    return feeds


def analyze_static_quantization(
    model_input_path: str | os.PathLike,
    *,
    op_types: list[str] | tuple[str, ...] = ("MatMul", "Gemm"),
    skip_nodes: list[str] | None = None,
    calibration_data: dict[str, np.ndarray] | None = None,
    dataset: str | os.PathLike | None = None,
    num_calib: int = 1,
    seed: int = 42,
    per_channel: bool = False,
    quant_format: str = "qdq",
    quant_dtype: str = "A8W8",
) -> list[dict]:
    """Rank each candidate node by how much statically quantizing it alone hurts the outputs.

    The analysis feeds (``calibration_data`` or seeded random samples) are
    used both for ORT calibration of the single quantized node and for the
    baseline / quantized output comparison; per-node metrics are averaged
    across the feeds.

    Returns a list of ``{node, op_type, kl, cosine, max_abs_error,
    classification}`` dicts sorted by KL divergence (most sensitive first).
    """
    if dataset is not None:
        raise NotImplementedError(_DATASET_NOT_IMPLEMENTED_MSG)

    skip_nodes = skip_nodes or []
    op_types = set(op_types)

    model = onnx.load(str(model_input_path))
    if is_model_quantized(model):
        raise ValueError(
            f"Model {model_input_path} appears to be already quantized; "
            "run sensitivity analysis on the fp32 source model"
        )
    input_specs = get_input_specs(model)
    if not input_specs:
        raise ValueError("Model has no graph inputs")
    feeds = _collect_feeds(
        input_specs,
        calibration_data=calibration_data,
        num_calib=num_calib,
        seed=seed,
    )

    base_sess = make_cpu_session(str(model_input_path))
    base_outs = [run_session(base_sess, feed) for feed in feeds]
    del base_sess

    candidates = [
        (n.name, n.op_type)
        for n in model.graph.node
        if n.op_type in op_types and n.name and not any(s in n.name for s in skip_nodes)
    ]
    logger.info(
        "Analyzing %d candidate node(s) of type(s) %s with %d feed(s)",
        len(candidates), ", ".join(sorted(op_types)), len(feeds),
    )

    results: list[dict] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        quant_path = Path(tmpdir) / "node.onnx"
        for idx, (node_name, op_type) in enumerate(candidates):
            quantized = onnx_static_quantize(
                model,
                num_calib=len(feeds),
                calibration_data_reader=_ListCalibrationDataReader(feeds),
                per_channel=per_channel,
                quant_format=quant_format,
                quant_dtype=quant_dtype,
                quantize_only_nodes=[node_name],
            )
            onnx.save(quantized, str(quant_path))
            sess = make_cpu_session(str(quant_path))
            kls: list[float] = []
            coss: list[float] = []
            errs: list[float] = []
            for feed, base_out in zip(feeds, base_outs):
                quant_out = run_session(sess, feed)
                kl, cos, err = compare_outputs(base_out, quant_out)
                kls.append(kl)
                coss.append(cos)
                errs.append(err)
            del sess
            kl = float(np.mean(kls))
            cos = float(np.mean(coss))
            err = float(np.mean(errs))
            severity = classify_severity(kl)
            logger.info(
                "[%d/%d] %s: kl=%.6g cos=%.6f max_err=%.6g [%s]",
                idx + 1, len(candidates), node_name, kl, cos, err, severity,
            )
            results.append({
                "node": node_name,
                "op_type": op_type,
                "kl": kl,
                "cosine": cos,
                "max_abs_error": err,
                "classification": severity,
            })

    results.sort(key=lambda r: r["kl"], reverse=True)
    return results


def add_static_analyze_args(parser: argparse.ArgumentParser) -> None:
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
        help="Output per-node sensitivity report JSON path",
    )
    parser.add_argument(
        "--exclude-output",
        type=str,
        default=None,
        help="Write nodes at/above --exclude-class as a JSON list usable with "
        "`quantize static --exclude-nodes`",
    )
    parser.add_argument(
        "--exclude-class",
        type=str,
        default="HIGH",
        choices=["MEDIUM", "HIGH", "CRITICAL"],
        help="Severity at/above which a node joins the exclude list (default: %(default)s)",
    )
    parser.add_argument(
        "--op-types",
        type=str,
        nargs="+",
        default=["MatMul", "Gemm"],
        help="Node op types to test (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-nodes",
        type=str,
        nargs="*",
        default=[],
        help="Node name substrings to skip",
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        default=None,
        help="Path to a .npz of input feeds (keys = model input names). "
        "If omitted, use seeded random inputs.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Calibration dataset (not implemented yet, check back soon); the samples would "
        "drive both ORT calibration and output comparison",
    )
    parser.add_argument(
        "--num-calib",
        type=int,
        default=1,
        help="Number of seeded random calibration/comparison samples when "
        "--calibration-data is omitted (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random calibration inputs (default: %(default)s)",
    )
    add_static_quant_flags(parser, skip=("--full-integer",))
    add_logging_args(parser)


def static_analyze_from_args(args: argparse.Namespace) -> None:
    configure_logging(args.logging)

    calibration_data = None
    if args.calibration_data:
        npz = np.load(args.calibration_data)
        calibration_data = {name: npz[name] for name in npz.files}

    results = analyze_static_quantization(
        args.input,
        op_types=args.op_types,
        skip_nodes=args.skip_nodes,
        calibration_data=calibration_data,
        dataset=args.dataset,
        num_calib=args.num_calib,
        seed=args.seed,
        per_channel=args.per_channel,
        quant_format=args.quant_format,
        quant_dtype=args.quant_dtype,
    )

    out_path = Path(args.output)
    write_report(results, out_path)
    logger.info("Saved sensitivity report (%d nodes) to %s", len(results), out_path)

    summary: dict[str, int] = {}
    for r in results:
        summary[r["classification"]] = summary.get(r["classification"], 0) + 1
    logger.info("Sensitivity summary: %s", summary)

    if args.exclude_output:
        count = write_exclude_list(results, args.exclude_output, args.exclude_class)
        logger.info(
            "Wrote %d node(s) >= %s to exclude list %s",
            count, args.exclude_class, args.exclude_output,
        )
