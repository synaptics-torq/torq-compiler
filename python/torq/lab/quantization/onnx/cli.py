# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unified CLI for ONNX quantization and quantization sensitivity analysis.

The quantize tool is installed as the ``torq-quantize-model`` console script
and delegated from ``torq-lab quantize``; it produces quantized models, and
its ``analyze`` subcommand produces the sensitivity reports that feed back
into quantization (``quantize static --exclude-nodes`` /
``quantize dynamic --exclude-nodes`` / ``quantize weights --config``):

    torq-quantize-model static -i model.onnx -o model_static.onnx [--dataset D]
    torq-quantize-model dynamic -i model.onnx -o model_dynamic.onnx
    torq-quantize-model weights -i model.onnx -o model_weights.onnx [--bits 8]
    torq-quantize-model analyze static -i model.onnx -o report.json [--dataset D]
    torq-quantize-model analyze dynamic -i model.onnx -o report.json
    torq-quantize-model analyze weights -i model.onnx -o report.json --embeddings token_embeddings.npy

``torq-lab analyze`` delegates to the same analyze command tree:

    torq-lab analyze dynamic -i model.onnx -o report.json
"""

import argparse
import sys

from torq.lab.quantization.onnx.dynamic import (
    add_dynamic_analyze_args,
    add_dynamic_quantize_args,
    dynamic_analyze_from_args,
    dynamic_quantize_from_args,
)
from torq.lab.quantization.onnx.static import (
    add_static_analyze_args,
    add_static_quantize_args,
    static_analyze_from_args,
    static_quantize_from_args,
)
from torq.lab.quantization.onnx.weights import (
    add_weights_analyze_args,
    add_weights_quantize_args,
    weights_analyze_from_args,
    weights_quantize_from_args,
)

_OK = 0
_ERROR = 1


def _cmd_static(args: argparse.Namespace) -> int:
    output_path = static_quantize_from_args(args)
    if args.full_integer:
        print(f"Full-integer quantized model saved to: {output_path}")
    else:
        print(f"Quantized model saved to: {output_path}")
    return _OK


def _cmd_quantize_dynamic(args: argparse.Namespace) -> int:
    dynamic_quantize_from_args(args)
    print(f"Quantized model saved to: {args.output}")
    return _OK


def _cmd_quantize_weights(args: argparse.Namespace) -> int:
    weights_quantize_from_args(args)
    print(f"Quantized model saved to: {args.output}")
    return _OK


def _cmd_analyze_static(args: argparse.Namespace) -> int:
    static_analyze_from_args(args)
    return _OK


def _cmd_analyze_dynamic(args: argparse.Namespace) -> int:
    dynamic_analyze_from_args(args)
    return _OK


def _cmd_analyze_weights(args: argparse.Namespace) -> int:
    weights_analyze_from_args(args)
    return _OK


def build_parser(prog: str = "torq-quantize-model") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Quantize ONNX models (static / dynamic / weights) and "
            "run quantization sensitivity analysis"
        ),
    )
    command = parser.add_subparsers(dest="command", required=True)

    static_parser = command.add_parser(
        "static",
        help="Static int8 quantization with calibration",
    )
    add_static_quantize_args(static_parser)
    static_parser.set_defaults(func=_cmd_static)

    dynamic_parser = command.add_parser(
        "dynamic",
        help="Dynamic int8 quantization via onnxruntime (no calibration)",
    )
    add_dynamic_quantize_args(dynamic_parser)
    dynamic_parser.set_defaults(func=_cmd_quantize_dynamic)

    weights_parser = command.add_parser(
        "weights",
        help="Weight-only int4/int8/bf16 quantization of MatMul weights",
    )
    add_weights_quantize_args(weights_parser)
    weights_parser.set_defaults(func=_cmd_quantize_weights)

    analyze_parser = command.add_parser(
        "analyze",
        help=(
            "Quantization sensitivity analysis; reports feed back into "
            "quantization (--exclude-nodes / --config)"
        ),
    )
    method = analyze_parser.add_subparsers(dest="method", required=True)

    analyze_static_parser = method.add_parser(
        "static",
        help="Rank nodes by static-quantization sensitivity (per-node)",
    )
    add_static_analyze_args(analyze_static_parser)
    analyze_static_parser.set_defaults(func=_cmd_analyze_static)

    analyze_dynamic_parser = method.add_parser(
        "dynamic",
        help="Rank nodes by dynamic-quantization sensitivity (per-node)",
    )
    add_dynamic_analyze_args(analyze_dynamic_parser)
    analyze_dynamic_parser.set_defaults(func=_cmd_analyze_dynamic)

    analyze_weights_parser = method.add_parser(
        "weights",
        help="Rank MatMul layers by weight-quantization sensitivity (per-layer)",
    )
    add_weights_analyze_args(analyze_weights_parser)
    analyze_weights_parser.set_defaults(func=_cmd_analyze_weights)

    return parser


def build_analyze_parser(prog: str = "torq-lab analyze") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Quantization sensitivity analysis for ONNX models; reports feed back "
            "into quantization (e.g. `torq-lab quantize static --exclude-nodes`, "
            "`torq-lab quantize dynamic --exclude-nodes`, `torq-lab quantize weights --config`)"
        ),
    )
    method = parser.add_subparsers(dest="method", required=True)

    static_parser = method.add_parser(
        "static",
        help="Rank nodes by static-quantization sensitivity (per-node)",
    )
    add_static_analyze_args(static_parser)
    static_parser.set_defaults(func=_cmd_analyze_static)

    dynamic_parser = method.add_parser(
        "dynamic",
        help="Rank nodes by dynamic-quantization sensitivity (per-node)",
    )
    add_dynamic_analyze_args(dynamic_parser)
    dynamic_parser.set_defaults(func=_cmd_analyze_dynamic)

    weights_parser = method.add_parser(
        "weights",
        help="Rank MatMul layers by weight-quantization sensitivity (per-layer)",
    )
    add_weights_analyze_args(weights_parser)
    weights_parser.set_defaults(func=_cmd_analyze_weights)

    return parser


def _run(parser: argparse.ArgumentParser, argv, error_prefix: str) -> int:
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return args.func(args)
    except Exception as exc:
        print(f"{error_prefix} {exc}", file=sys.stderr)
        return _ERROR


def main(argv=None, prog: str = "torq-quantize-model") -> int:
    return _run(build_parser(prog), argv, "Error:")


def analyze_main(argv=None, prog: str = "torq-lab analyze") -> int:
    return _run(build_analyze_parser(prog), argv, "Error analyzing model:")


if __name__ == "__main__":
    sys.exit(main())
