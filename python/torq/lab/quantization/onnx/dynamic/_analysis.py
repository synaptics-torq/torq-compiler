# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Per-node sensitivity analysis for ONNX dynamic quantization.

For each candidate node, quantize *only* that node, run the model against
calibration inputs, and measure how far the outputs drift from the fp32
baseline (KL divergence, cosine similarity, max absolute error). Nodes are
classified by severity so the most damaging ones can be excluded from
quantization via ``--exclude-nodes``.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization.preprocess import quant_pre_process

from torq.lab.logging import add_logging_args, configure_logging
from torq.lab.verification.compare import classify_severity
from torq.lab.utils.onnxruntime import make_cpu_session

from .._analysis import (
    build_random_feeds,
    compare_outputs,
    run_session,
    write_exclude_list,
    write_report,
)
from ._quantization import onnx_dynamic_quantize_file

logger = logging.getLogger(__name__)


def analyze_dynamic_quantization(
    model_input_path: str | os.PathLike,
    *,
    op_types: list[str] | tuple[str, ...] = ("MatMul", "Gemm"),
    skip_nodes: list[str] | None = None,
    calibration_data: dict[str, np.ndarray] | None = None,
    seed: int = 42,
    skip_preprocess: bool = False,
    uint8_weights: bool = False,
    per_tensor: bool = False,
) -> list[dict]:
    """Rank each candidate node by how much dynamically quantizing it alone hurts the outputs.

    Returns a list of ``{node, op_type, kl, cosine, max_abs_error, classification}``
    dicts sorted by KL divergence (most sensitive first).
    """
    skip_nodes = skip_nodes or []
    op_types = set(op_types)

    with tempfile.TemporaryDirectory() as tmpdir:
        if skip_preprocess:
            base_path = os.fspath(model_input_path)
        else:
            base_path = os.path.join(tmpdir, "base.onnx")
            quant_pre_process(model_input_path, base_path)
        model = onnx.load(base_path)

        feeds = (
            calibration_data
            if calibration_data is not None
            else build_random_feeds(model, seed)
        )
        base_out = run_session(make_cpu_session(base_path), feeds)

        candidates = [
            (n.name, n.op_type)
            for n in model.graph.node
            if n.op_type in op_types and n.name and not any(s in n.name for s in skip_nodes)
        ]
        logger.info(
            "Analyzing %d candidate node(s) of type(s) %s",
            len(candidates), ", ".join(sorted(op_types)),
        )

        results: list[dict] = []
        quant_path = os.path.join(tmpdir, "node.onnx")
        for idx, (node_name, op_type) in enumerate(candidates):
            onnx_dynamic_quantize_file(
                base_path, quant_path,
                quantize_only_nodes=[node_name],
                skip_preprocess=True,
                uint8_weights=uint8_weights,
                per_tensor=per_tensor,
            )
            quant_out = run_session(make_cpu_session(quant_path), feeds)
            kl, cos, max_err = compare_outputs(base_out, quant_out)
            severity = classify_severity(kl)
            logger.info(
                "[%d/%d] %s: kl=%.6g cos=%.6f max_err=%.6g [%s]",
                idx + 1, len(candidates), node_name, kl, cos, max_err, severity,
            )
            results.append({
                "node": node_name,
                "op_type": op_type,
                "kl": kl,
                "cosine": cos,
                "max_abs_error": max_err,
                "classification": severity,
            })

    results.sort(key=lambda r: r["kl"], reverse=True)
    return results


def summarize_dynamic_quantization(
    model_input_path: str | os.PathLike,
    quantized_model_path: str | os.PathLike,
    *,
    calibration_data: dict[str, np.ndarray] | None = None,
    seed: int = 42,
) -> dict:
    """Whole-model, single-pass comparison of a quantized model against its fp32 source."""
    model = onnx.load(model_input_path)
    feeds = (
        calibration_data
        if calibration_data is not None
        else build_random_feeds(model, seed)
    )
    base_out = run_session(make_cpu_session(str(model_input_path)), feeds)
    quant_out = run_session(make_cpu_session(str(quantized_model_path)), feeds)
    kl, cos, max_err = compare_outputs(base_out, quant_out)
    return {
        "kl": kl,
        "cosine": cos,
        "max_abs_error": max_err,
        "classification": classify_severity(kl),
    }


def add_dynamic_analyze_args(parser: argparse.ArgumentParser) -> None:
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
        "`quantize dynamic --exclude-nodes`",
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
        "If omitted, seeded random inputs are used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random calibration inputs (default: %(default)s)",
    )
    parser.add_argument(
        "--uint8-weights",
        action="store_true",
        default=False,
        help="Analyze with unsigned int8 weights",
    )
    parser.add_argument(
        "--per-tensor",
        action="store_true",
        default=False,
        help="Analyze with per-tensor (not per-channel) quantization",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        default=False,
        help="Skip onnxruntime pre-processing before quantization",
    )
    add_logging_args(parser)


def dynamic_analyze_from_args(args: argparse.Namespace) -> None:
    configure_logging(args.logging)

    calibration_data = None
    if args.calibration_data:
        npz = np.load(args.calibration_data)
        calibration_data = {name: npz[name] for name in npz.files}

    results = analyze_dynamic_quantization(
        args.input,
        op_types=args.op_types,
        skip_nodes=args.skip_nodes,
        calibration_data=calibration_data,
        seed=args.seed,
        skip_preprocess=args.skip_preprocess,
        uint8_weights=args.uint8_weights,
        per_tensor=args.per_tensor,
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
