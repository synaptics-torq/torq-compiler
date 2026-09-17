# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TFLite dynamic-to-static shape conversion.

Removes tensor shape signatures (the dynamic-shape markers) from a TFLite
model so it runs with its default shapes. The FlatBuffer schema is borrowed
from TensorFlow's ``tensorflow.lite.python.schema_py_generated``.
"""

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import flatbuffers

from torq.lab import LabError

_tflite_schema = None


def _schema():
    """Lazily import the TFLite flatbuffer schema module from TensorFlow.

    Deferred so that importing this module doesn't pull TensorFlow into the
    process; see ``torq.lab.model_tools.extraction.tflite.layers._schema``.
    """
    global _tflite_schema
    if _tflite_schema is None:
        from tensorflow.lite.python import schema_py_generated as module

        _tflite_schema = module
    return _tflite_schema


def _require_tensorflow() -> None:
    """Fail with an install hint when TensorFlow (the ``[tf]`` extra) is absent."""
    try:
        _schema()
    except ImportError as exc:
        raise LabError(
            "tensorflow is required for 'torq-convert-static tflite' but is not "
            "installed. Install the torq-compiler [tf] extra: "
            "pip install 'torq-compiler[tf]'"
        ) from exc


def convert_model(input_model, output_model):
    """Convert a dynamic TFLite model to static shapes, writing ``output_model``."""
    input_path = Path(input_model)
    output_path = Path(output_model)

    # Validate file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Model file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Path is not a file: {input_path}")

    # Validate it's a .tflite file by extension
    if input_path.suffix.lower() != ".tflite":
        raise ValueError(f"Expected a .tflite file, got: {input_path.suffix}")

    buf = input_path.read_bytes()
    model_t = _schema().ModelT.InitFromPackedBuf(bytearray(buf), 0)

    converted = 0
    for subgraph in model_t.subgraphs:
        for tensor in subgraph.tensors:
            if isinstance(tensor.shapeSignature, Iterable) and -1 in tensor.shapeSignature:
                tensor.shapeSignature = None
                converted += 1

    builder = flatbuffers.Builder(1024)
    builder.Finish(model_t.Pack(builder), b"TFL3")

    output_path.write_bytes(builder.Output())

    print(f"\nWrote {output_path}")
    print(f"Removed dynamic shape signatures from {converted} tensor(s)")


def add_tflite_static_convert_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input TFLite model path"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output TFLite model path"
    )


def tflite_static_convert_from_args(args: argparse.Namespace):
    _require_tensorflow()
    convert_model(
        args.input,
        args.output
    )


def main(argv: list[str] | None = None) -> int:
    """Main entry point for torq-convert-static."""
    parser = argparse.ArgumentParser(
        prog="torq-convert-static",
        description="Convert dynamic models to static, using the default shapes",
    )
    model_type = parser.add_subparsers(dest="model_type", required=True)

    tflite_type = model_type.add_parser("tflite", help="Convert TFLite dynamic models")
    add_tflite_static_convert_args(tflite_type)

    args = parser.parse_args(argv)

    if args.model_type == "tflite":
        try:
            tflite_static_convert_from_args(args)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
