# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""dtype mapping, MLIR IO-spec parsing, and input/output materialization."""

import re
from pathlib import Path
from typing import List, Optional

import ml_dtypes  # support for the bfloat16 dtype
import numpy as np

from torq.lab.types import LabError, MlirIoSpec, TensorType

# MLIR type-name string -> numpy dtype.
_DTYPE_BY_NAME = {
    "i1": np.dtype(bool),
    "i8": np.dtype(np.int8),
    "i16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "ui8": np.dtype(np.uint8),
    "ui16": np.dtype(np.uint16),
    "ui32": np.dtype(np.uint32),
    "ui64": np.dtype(np.uint64),
    "f16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
    "si8": np.dtype(np.int8),
    "si16": np.dtype(np.int16),
    "si64": np.dtype(np.int64),
    "si32": np.dtype(np.int32),
    "bf16": np.dtype(ml_dtypes.bfloat16),
}

# numpy dtype -> MLIR type-name string, for building --input args from raw arrays.
_NAME_BY_DTYPE = {
    np.dtype(bool): "i1",
    np.dtype(np.int8): "si8",
    np.dtype(np.int16): "si16",
    np.dtype(np.int32): "si32",
    np.dtype(np.int64): "si64",
    np.dtype(np.uint8): "ui8",
    np.dtype(np.uint16): "ui16",
    np.dtype(np.uint32): "ui32",
    np.dtype(np.uint64): "ui64",
    np.dtype(np.float16): "f16",
    np.dtype(np.float32): "f32",
    np.dtype(ml_dtypes.bfloat16): "bf16",
}


def get_dtype(name: str) -> np.dtype:
    """Return the numpy dtype for the given MLIR type name."""
    if name in _DTYPE_BY_NAME:
        return _DTYPE_BY_NAME[name]
    raise LabError(f"Unsupported dtype {name}")


def get_type_name(dtype) -> str:
    """Return the MLIR type-name string for a numpy dtype."""
    dtype = np.dtype(dtype)
    if dtype in _NAME_BY_DTYPE:
        return _NAME_BY_DTYPE[dtype]
    raise LabError(f"Unsupported array dtype {dtype}")


def fmt_for_array(array: np.ndarray) -> str:
    """Return the MLIR type-name string for a numpy array's dtype."""
    return get_type_name(array.dtype)


def is_float_type(dtype) -> bool:
    """True for numpy floating types or bfloat16."""
    dtype = np.dtype(dtype)
    return np.issubdtype(dtype, np.floating) or dtype == np.dtype(ml_dtypes.bfloat16)


def parse_mlir_io_spec(mlir_path) -> MlirIoSpec:
    """Parse the input/output tensor types from an MLIR module file."""
    try:
        from iree.compiler.ir import Context, Module
    except ImportError as exc:  # pragma: no cover - exercised only without iree
        raise LabError(
            "iree.compiler.ir is required to parse MLIR IO specs but is not "
            "importable. Install the torq-compiler wheel or set up the compiler "
            "Python bindings."
        ) from exc

    mlir_content = Path(mlir_path).read_text()

    module = Module.parse(mlir_content, Context())
    input_types: List[str] = []
    output_types: List[str] = []
    # Collect the IO types of a single entry function. The lists are reset for
    # every top-level op so they describe the last function op only, rather than
    # accumulating arguments/results across all functions in the module (jax
    # lowerings, for instance, emit more than one top-level function).
    for op in module.body.operations:
        input_types = []
        output_types = []
        for region in op.regions:
            for block in region.blocks:
                for arg in block.arguments:
                    input_types.append(str(arg.type))
                for inner_op in block.operations:
                    if inner_op.name == "func.return":
                        for operand in inner_op.operands:
                            output_types.append(str(operand.type))

    def torch_spec(text: str) -> Optional[TensorType]:
        match = re.match(r"!torch\.vtensor<\[(.*)\],(\w+)>", text)
        if not match:
            return None
        if match.group(1) == "":
            shape: List[int] = []
        else:
            shape = [1 if d == "?" else int(d) for d in match.group(1).split(",")]
        return TensorType(shape, match.group(2))

    def tosa_spec(text: str) -> Optional[TensorType]:
        match = re.match(r"tensor<([^>]+)>", text)
        if match:
            return TensorType.from_string(match.group(1))
        return None

    input_specs = [torch_spec(t) for t in input_types]
    output_specs = [torch_spec(t) for t in output_types]

    if None in input_specs or None in output_specs:
        input_specs = [tosa_spec(t) for t in input_types]
        output_specs = [tosa_spec(t) for t in output_types]

    return MlirIoSpec(inputs=input_specs, outputs=output_specs)


def parse_func_name(mlir_path) -> str:
    """Return the first ``func.func`` symbol name in an MLIR module (or 'main')."""
    func_name = "main"
    for line in Path(mlir_path).read_text().split("\n"):
        if re.match(r"^\s*func.func.*", line):
            m = re.search(r"@(\w+)\s*\(", line) or re.search(r'@"([^"]+)"\s*\(', line)
            if m:
                return m.group(1)
    return func_name


def create_output_paths(output_dir, output_specs) -> List[str]:
    """Return the ``output_<idx>.bin`` paths for each output spec."""
    return [f"{output_dir}/output_{idx}.bin" for idx in range(len(output_specs))]


def create_output_args(output_dir, output_specs) -> List[str]:
    """Return the ``--output=@<path>`` args for ``torq-run-module``."""
    return [f"--output=@{path}" for path in create_output_paths(output_dir, output_specs)]


def load_outputs(output_specs, output_paths) -> List[np.ndarray]:
    """Read the raw output ``.bin`` files back into numpy arrays."""
    outputs = []
    for spec, path in zip(output_specs, output_paths):
        data = np.frombuffer(
            Path(path).read_bytes(), dtype=get_dtype(spec.fmt)
        ).reshape(spec.shape)
        outputs.append(data)
    return outputs


def generate_random_inputs(io_spec: MlirIoSpec, seed: int = 1234) -> List[np.ndarray]:
    """Generate uniform random input arrays for each input tensor spec."""
    rng = np.random.default_rng(seed)
    inputs = []
    for spec in io_spec.inputs:
        dtype = get_dtype(spec.fmt)
        if is_float_type(dtype):
            finfo = ml_dtypes.finfo if dtype == ml_dtypes.bfloat16 else np.finfo
            data = rng.uniform(finfo(dtype).min, finfo(dtype).max, spec.shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            data = rng.integers(
                np.iinfo(dtype).min, np.iinfo(dtype).max, size=spec.shape,
                dtype=dtype, endpoint=True,
            )
        elif dtype == np.dtype(bool):
            data = rng.integers(0, 2, spec.shape, dtype=dtype)
        else:
            raise LabError(f"Cannot generate random data for unsupported dtype '{dtype}'")
        inputs.append(data)
    return inputs


def write_inputs(inputs: List[np.ndarray], inputs_dir) -> List[Path]:
    """Write each input array as ``in_rnd_<i>.bin`` (+ ``.npy``) and return the .bin paths."""
    inputs_dir = Path(inputs_dir)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, data in enumerate(inputs):
        bin_path = inputs_dir / f"in_rnd_{i}.bin"
        bin_path.write_bytes(data.tobytes())
        np.save(str(bin_path) + ".npy", data)
        paths.append(bin_path)
    return paths


def build_input_args(
    input_paths: List[Path],
    inputs: List[np.ndarray],
    io_spec: Optional[MlirIoSpec] = None,
) -> List[str]:
    """Build ``--input=<type>=@<path>`` args from staged inputs.

    Uses the parsed spec's tensor type when available, otherwise derives the
    type string from each array's shape and dtype.
    """
    args = []
    for i, (path, data) in enumerate(zip(input_paths, inputs)):
        if io_spec is not None and i < len(io_spec.inputs):
            type_arg = io_spec.inputs[i].to_arg()
        else:
            type_arg = TensorType(list(data.shape), fmt_for_array(data)).to_arg()
        args.append(f"--input={type_arg}=@{path}")
    return args
