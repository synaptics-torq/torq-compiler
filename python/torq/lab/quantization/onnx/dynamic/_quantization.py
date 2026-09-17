# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import argparse
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.preprocess import quant_pre_process

from torq.lab.utils.cli import (
    parse_remainder_args_to_dict,
)
from torq.lab.logging import (
    add_logging_args,
    configure_logging,
)


def _onnx_dynamic_quantize(
    model_input: str | os.PathLike | onnx.ModelProto,
    model_output: str | os.PathLike,
    *,
    quantize_only_ops: list[str] | None = None,
    quantize_only_nodes: list[str] | None = None,
    exclude_nodes: list[str] | None = None,
    skip_preprocess: bool = False,
    uint8_weights: bool = False,
    per_tensor: bool = False,
    **quantize_kwargs
):
    """Run the onnxruntime dynamic-quantization pass, writing to *model_output*.

    Shared core of :func:`onnx_dynamic_quantize` (ModelProto I/O) and
    :func:`onnx_dynamic_quantize_file` (file I/O); returns the quantized
    model after shape inference (also saved to *model_output*).
    """
    logger.debug("Dynamically quantizing '%s' with the following parameters:", model_input)
    logger.debug("  Weight dtype = %s", "UINT8" if uint8_weights else "INT8")
    logger.debug("  Per-tensor   = %s", str(per_tensor))
    logger.debug("  Op types     = %s", "all" if quantize_only_ops is None else ", ".join(quantize_only_ops))
    logger.debug("  Graph nodes  = %s", "all" if quantize_only_nodes is None else ", ".join(quantize_only_nodes))
    logger.debug("  Exclude nodes = %s", "none" if exclude_nodes is None else ", ".join(exclude_nodes))
    if not skip_preprocess:
        quant_pre_process(model_input, model_output)
        logger.debug("Preprocessed model '%s' before quantization", str(model_output))
    quantize_dynamic(
        model_input if skip_preprocess else model_output,
        model_output,
        op_types_to_quantize=quantize_only_ops,
        nodes_to_quantize=quantize_only_nodes,
        nodes_to_exclude=exclude_nodes,
        weight_type=QuantType.QUInt8 if uint8_weights else QuantType.QInt8,
        per_channel=not per_tensor,
        **quantize_kwargs
    )
    model = onnx.load(model_output)
    model = onnx.shape_inference.infer_shapes(model, True, True, True)
    onnx.save(model, model_output)
    logger.debug("Saved dynamically quantized model to '%s'", model_output)
    return model


def onnx_dynamic_quantize(
    model: onnx.ModelProto,
    *,
    quantize_only_ops: list[str] | None = None,
    quantize_only_nodes: list[str] | None = None,
    exclude_nodes: list[str] | None = None,
    skip_preprocess: bool = False,
    uint8_weights: bool = False,
    per_tensor: bool = False,
    **quantize_kwargs
) -> onnx.ModelProto:
    """Dynamically quantize an in-memory ONNX model to int8.

    Thin wrapper around :func:`_onnx_dynamic_quantize` with ModelProto I/O;
    the onnxruntime pass runs against a scratch file and the quantized model
    is returned.

    Args:
        model: The fp32 ONNX model to quantize.
        quantize_only_ops: Only quantize the given ONNX op types.
        quantize_only_nodes: Only quantize the given node names.
        exclude_nodes: Exclude the given node names from quantization (e.g.
            an analysis exclude list).
        skip_preprocess: Skip the pre-processing pass before quantization.
        uint8_weights: Use unsigned int8 weights.
        per_tensor: Quantize weights per tensor instead of per channel.
        **quantize_kwargs: Forwarded to onnxruntime's ``quantize_dynamic``.

    Returns:
        The quantized ONNX model.
    """
    with tempfile.TemporaryDirectory() as tmp:
        return _onnx_dynamic_quantize(
            model,
            Path(tmp) / "model.onnx",
            quantize_only_ops=quantize_only_ops,
            quantize_only_nodes=quantize_only_nodes,
            exclude_nodes=exclude_nodes,
            skip_preprocess=skip_preprocess,
            uint8_weights=uint8_weights,
            per_tensor=per_tensor,
            **quantize_kwargs
        )


def onnx_dynamic_quantize_file(
    model_input: str | os.PathLike,
    model_output: str | os.PathLike,
    *,
    quantize_only_ops: list[str] | None = None,
    quantize_only_nodes: list[str] | None = None,
    exclude_nodes: list[str] | None = None,
    skip_preprocess: bool = False,
    uint8_weights: bool = False,
    per_tensor: bool = False,
    **quantize_kwargs
) -> Path:
    """Dynamically quantize an ONNX model file to int8.

    Thin wrapper around :func:`_onnx_dynamic_quantize` with file I/O.

    Args:
        model_input: Path to the fp32 ONNX model.
        model_output: Path where the quantized ONNX model will be written.
        quantize_only_ops: Only quantize the given ONNX op types.
        quantize_only_nodes: Only quantize the given node names.
        exclude_nodes: Exclude the given node names from quantization (e.g.
            an analysis exclude list).
        skip_preprocess: Skip the pre-processing pass before quantization.
        uint8_weights: Use unsigned int8 weights.
        per_tensor: Quantize weights per tensor instead of per channel.
        **quantize_kwargs: Forwarded to onnxruntime's ``quantize_dynamic``.

    Returns:
        The path to the written quantized model.
    """
    model_output = Path(model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    _onnx_dynamic_quantize(
        model_input,
        model_output,
        quantize_only_ops=quantize_only_ops,
        quantize_only_nodes=quantize_only_nodes,
        exclude_nodes=exclude_nodes,
        skip_preprocess=skip_preprocess,
        uint8_weights=uint8_weights,
        per_tensor=per_tensor,
        **quantize_kwargs
    )
    return model_output


def add_dynamic_quantize_args(parser: argparse.ArgumentParser) -> None:
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
        help="Output quantized ONNX model path",
    )
    parser.add_argument(
        "--quantize-only-ops",
        type=str,
        nargs="+",
        default=None,
        help="Only quantize specified op types; must be valid ONNX op types",
    )
    parser.add_argument(
        "--quantize-only-nodes",
        type=str,
        nargs="+",
        default=None,
        help="Only quantize specified nodes; must be valid node names from graph",
    )
    parser.add_argument(
        "--exclude-nodes",
        type=str,
        nargs="+",
        default=None,
        help="Exclude specified nodes from quantization (e.g. an analysis exclude list)",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        default=False,
        help="Skip pre-processing steps that may improve quantization quality",
    )
    parser.add_argument(
        "--uint8-weights",
        action="store_true",
        default=False,
        help="Generate unsigned integer weights during quantization",
    )
    parser.add_argument(
        "--per-tensor",
        action="store_true",
        default=False,
        help="Quantize weights per channel",
    )
    parser.add_argument(
        "--extra-quant-args",
        nargs=argparse.REMAINDER,
        default=None,
        metavar="FLAG",
        help=(
            "[Advanced] Extra quantization args for `onnxruntime.quantization.dynamic_quantize`. "
            "Must be specified last; all remaining arguments are forwarded."
        ),
    )
    add_logging_args(parser)


def dynamic_quantize_from_args(args: argparse.Namespace) -> Path:
    configure_logging(args.logging)
    model_path = Path(args.input)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    extra_quant_args = parse_remainder_args_to_dict(
        args.extra_quant_args,
        "--extra_quant_args"
    )
    return onnx_dynamic_quantize_file(
        args.input, args.output,
        quantize_only_ops=args.quantize_only_ops,
        quantize_only_nodes=args.quantize_only_nodes,
        exclude_nodes=args.exclude_nodes,
        skip_preprocess=args.skip_preprocess,
        uint8_weights=args.uint8_weights,
        per_tensor=args.per_tensor,
        **extra_quant_args
    )
