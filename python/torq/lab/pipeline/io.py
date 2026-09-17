# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""MLIR I/O specs, dtype mapping, data materialization, and the runtime I/O dtype policy.

The policy mirrors the compiler's ``--torq-convert-dtypes --torq-convert-io-dtype``
path: when a model is compiled with narrowed I/O dtypes, the reference path must
see the same precision.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ml_dtypes  # support for the bfloat16 dtype
import numpy as np

from torq.lab import LabError

@dataclass
class TensorType:
    """The type of a tensor input or output of an MLIR model.

    ``name`` is optional: the MLIR IO-spec parser leaves it ``None`` (imported
    MLIR carries no argument names); callers building a spec programmatically
    (e.g. from an ONNX model) may set it so that name-keyed ``input_ranges``
    entries can match.

    A dynamic (``?``) dimension is ``None`` in ``shape``; :meth:`to_arg`
    renders it back as ``?``.
    """

    shape: List[Optional[int]]
    fmt: str
    name: Optional[str] = None

    def to_arg(self) -> str:
        return "x".join(["?" if d is None else str(d) for d in self.shape] + [self.fmt])

    @staticmethod
    def from_string(spec: str) -> "TensorType":
        """Parse a tensor type like ``1x16xbf16`` (``?`` marks a dynamic dim).

        Raises :class:`~torq.lab.LabError` on a malformed dimension or an
        unsupported dtype name.
        """
        *shape_str, fmt = spec.split("x")
        if not fmt:
            raise LabError(f"invalid tensor spec {spec!r}: missing dtype after the last 'x'")
        if fmt not in _DTYPE_BY_NAME:
            raise LabError(
                f"invalid tensor spec {spec!r}: unsupported dtype {fmt!r} "
                f"(expected one of: {', '.join(sorted(_DTYPE_BY_NAME))})"
            )
        shape: List[Optional[int]] = []
        for s in shape_str:
            if s == "?":
                shape.append(None)
            else:
                try:
                    shape.append(int(s))
                except ValueError:
                    raise LabError(f"invalid tensor spec {spec!r}: bad dimension {s!r}") from None
        return TensorType(shape, fmt)


@dataclass
class MlirIoSpec:
    """Input and output tensor types parsed from an MLIR module."""

    inputs: List[TensorType]
    outputs: List[TensorType]

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


def _load_module(mlir_path):
    """Parse the MLIR module at ``mlir_path``, with a clean error when the
    compiler bindings are unavailable."""
    try:
        from iree.compiler.ir import Context, Module
    except ImportError as exc:  # pragma: no cover - exercised only without iree
        raise LabError(
            "iree.compiler.ir is required to parse MLIR IO specs but is not "
            "importable. Install the torq-compiler wheel or set up the compiler "
            "Python bindings."
        ) from exc

    return Module.parse(Path(mlir_path).read_text(), Context())


def parse_mlir_io_spec(mlir_path) -> MlirIoSpec:
    """Parse the input/output tensor types from an MLIR module file."""
    module = _load_module(mlir_path)
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


def dynamic_io_types(mlir_path, function: Optional[str] = None) -> List[str]:
    """Return the entry function's I/O types that carry dynamic (``?``) dimensions.

    ``function`` selects the entry point (as ``--function`` does); when omitted
    the first ``func.func`` in the module is inspected (the same default
    :func:`parse_func_name` resolves to). Returns raw type strings, e.g.
    ``["tensor<?x16xbf16>"]`` — empty when the signature is fully static.
    """
    module = _load_module(mlir_path)
    dynamic: List[str] = []
    for op in module.body.operations:
        if op.OPERATION_NAME != "func.func":
            continue
        if function is not None and op.name.value != function:
            continue
        func_type = op.attributes["function_type"].value
        for tensor_type in list(func_type.inputs) + list(func_type.results):
            if "?" in str(tensor_type):
                dynamic.append(str(tensor_type))
        break  # only the first (matching) function op is the entry
    return dynamic


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


def _resolve_input_range(
    ranges: Dict[str, Tuple[float, float]], index: int, spec: TensorType
) -> Optional[Tuple[float, float]]:
    """Return the configured ``(min, max)`` range for one input, if any.

    A key matches either the spec's ``name`` (when it carries one) or the
    input's decimal index (``"0"``, ``"1"``, ...). The range must be
    well-formed (two values, min < max).
    """
    keys = ([spec.name] if spec.name else []) + [str(index)]
    for key in keys:
        if key in ranges:
            value_range = tuple(ranges[key])
            if len(value_range) != 2 or value_range[0] >= value_range[1]:
                raise LabError(
                    f"Invalid input range {list(value_range)} for input '{key}': "
                    "expected (min, max) with min < max"
                )
            return value_range
    return None


def generate_random_inputs(
    io_spec: MlirIoSpec,
    seed: Optional[int] = None,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[np.ndarray]:
    """Generate uniform random input arrays for each input tensor spec.

    ``seed`` selects the RNG stream (``None`` keeps the historical default of
    1234). ``ranges`` maps an input name (when the spec carries one) or its
    decimal index to a ``(min, max)`` range: float inputs draw from
    ``uniform(min, max)`` and integer inputs from ``integers(min, max)`` with
    ``max`` exclusive; bool inputs always draw from {0, 1}. Inputs without a
    matching key use the full dtype range, and ``ranges=None`` reproduces the
    historical output byte-for-byte. Unknown keys raise :class:`LabError`.
    """
    if ranges:
        ranges = {str(k): tuple(v) for k, v in ranges.items()}
        valid = {str(i) for i in range(len(io_spec.inputs))}
        valid |= {spec.name for spec in io_spec.inputs if spec.name}
        unknown = sorted(set(ranges) - valid)
        if unknown:
            raise LabError(
                f"input_ranges keys {unknown} match no model input; "
                f"valid keys are input names or decimal indices {sorted(valid)}"
            )

    rng = np.random.default_rng(1234 if seed is None else seed)
    inputs = []
    for index, spec in enumerate(io_spec.inputs):
        dtype = get_dtype(spec.fmt)
        value_range = _resolve_input_range(ranges, index, spec) if ranges else None
        if is_float_type(dtype):
            finfo = ml_dtypes.finfo if dtype == ml_dtypes.bfloat16 else np.finfo
            lo, hi = value_range if value_range is not None else (
                finfo(dtype).min, finfo(dtype).max
            )
            data = rng.uniform(lo, hi, spec.shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            if value_range is not None:
                # max exclusive, mirroring tweaked_random_input_data
                data = rng.integers(
                    int(value_range[0]), int(value_range[1]), size=spec.shape,
                    dtype=dtype,
                )
            else:
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


def load_npy(path) -> np.ndarray:
    """Load a ``.npy`` input/golden file.

    bfloat16 arrays saved via ``ml_dtypes`` round-trip through ``np.save`` as
    2-byte void (``|V2``); normalize them back to ``ml_dtypes.bfloat16`` so
    dtypes compare, cast, and feed onnxruntime as expected.
    """
    data = np.load(path, allow_pickle=False)
    if data.dtype == np.dtype("V2"):
        data = data.view(np.uint16).view(ml_dtypes.bfloat16)
    return data


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
def cast_round_trip(data: np.ndarray, down_dtype: np.dtype, up_dtype: np.dtype):
    """Cast *data* down to ``down_dtype`` and back to ``up_dtype``.

    For integer narrowing, the up-dtype sentinels (min/max) are remapped to the
    down-dtype sentinels first; values outside the target range raise.
    """
    if np.issubdtype(down_dtype, np.integer):
        up_info, down_info = np.iinfo(up_dtype), np.iinfo(down_dtype)
        data[data == up_info.max] = down_info.max
        data[data == up_info.min] = down_info.min
        if np.any(data < down_info.min) or np.any(data > down_info.max):
            raise ValueError(
                f"Cannot round-trip {data.dtype} through {down_dtype}: "
                "input contains values outside the target range"
            )
    return data.astype(down_dtype).astype(up_dtype)


@dataclass(frozen=True)
class ConvertIODTypesPolicy:

    # currently supported dtype conversions
    CONVERT_IO_DTYPES = {
        np.dtype(np.float16): np.dtype(ml_dtypes.bfloat16),
        np.dtype(np.float32): np.dtype(ml_dtypes.bfloat16),
        np.dtype(np.int64): np.dtype(np.int32),
        np.dtype(np.uint64): np.dtype(np.uint32),
    }

    mode: str | None  # None, "all", "include", "exclude"
    inputs: set[int]
    outputs: set[int]

    def __bool__(self) -> bool:
        return self.mode is not None

    def should_convert_input(self, idx: int) -> bool:
        return self._should_convert(idx, self.inputs)

    def should_convert_output(self, idx: int) -> bool:
        return self._should_convert(idx, self.outputs)

    def convert_io_spec(self, spec: MlirIoSpec) -> MlirIoSpec:
        """Return the VMFB I/O spec after this policy's conversions."""
        inputs = [
            TensorType(list(tensor.shape), self.convert_io_fmt(tensor.fmt))
            if self.should_convert_input(idx) else tensor
            for idx, tensor in enumerate(spec.inputs)
        ]
        outputs = [
            TensorType(list(tensor.shape), self.convert_io_fmt(tensor.fmt))
            if self.should_convert_output(idx) else tensor
            for idx, tensor in enumerate(spec.outputs)
        ]
        return MlirIoSpec(inputs=inputs, outputs=outputs)

    def round_trip_inputs(self, inputs: list) -> list:
        """Round-trip reference *inputs* through their converted dtypes.

        Each input flagged by this policy is cast down to its converted
        dtype and back, so the reference path sees the same precision the
        compiled model will.  Inputs not flagged are returned unchanged.
        """
        if not self:
            return inputs
        result = []
        for i, data in enumerate(inputs):
            if not self.should_convert_input(i):
                result.append(data)
                continue
            down_dtype = self.convert_io_dtype(data.dtype)
            result.append(cast_round_trip(data, down_dtype, data.dtype))
        return result

    def round_trip_outputs(self, outputs: list) -> list:
        """Round-trip reference *outputs* through their converted dtypes.

        Mirror of :meth:`round_trip_inputs` for outputs.  Arrays are copied
        before the cast so the caller's originals (e.g. ONNX Runtime's
        returned tensors) are never mutated.
        """
        if not self:
            return outputs
        result = []
        for i, data in enumerate(outputs):
            if not self.should_convert_output(i):
                result.append(data)
                continue
            down_dtype = self.convert_io_dtype(data.dtype)
            if down_dtype == np.dtype(data.dtype):
                result.append(data)
                continue
            result.append(cast_round_trip(data.copy(), down_dtype, data.dtype))
        return result

    def _should_convert(self, idx: int, indices: set[int]) -> bool:
        if self.mode is None:
            return False
        if self.mode == "all":
            return True
        if self.mode == "include":
            return idx in indices
        if self.mode == "exclude":
            return idx not in indices
        raise ValueError(f"Unknown mode: {self.mode}, expected one of [None, 'all', 'include', 'exclude']")

    @classmethod
    def parse_from_args(cls, args: list[str], torq_convert_dtypes: bool) -> "ConvertIODTypesPolicy":
        if not torq_convert_dtypes:
            return cls(None, set(), set())
        if args == ["all"]:
            return cls("all", set(), set())
        if any(arg == "all" for arg in args):
            raise LabError(
                "--convert-io-dtypes: 'all' cannot be combined with specific indices"
            )
        has_include = any(not arg.startswith("!") for arg in args)
        has_exclude = any(arg.startswith("!") for arg in args)
        if has_include and has_exclude:
            raise LabError("--convert-io-dtypes cannot mix included and excluded indices")
        mode = "exclude" if has_exclude else "include"
        inputs: set[int] = set()
        outputs: set[int] = set()
        for arg in args:
            raw = arg[1:] if arg.startswith("!") else arg
            try:
                kind, idx_str = raw.split(":", 1)
            except ValueError:
                raise LabError(f"--convert-io-dtypes: expected input:IDX or output:IDX, got {arg!r}")
            if kind not in {"input", "output"}:
                raise LabError(f"--convert-io-dtypes: expected 'input' or 'output', got {kind!r}")
            try:
                idx = int(idx_str)
            except ValueError:
                raise LabError(f"--convert-io-dtypes: expected integer index, got {idx_str!r}")
            if idx < 0:
                raise LabError(f"--convert-io-dtypes: index must be non-negative, got {idx}")
            if kind == "input":
                inputs.add(idx)
            else:
                outputs.add(idx)
        return cls(mode, inputs, outputs)

    @staticmethod
    def convert_io_fmt(fmt):
        """
        Maps an MLIR type string to its --torq-convert-io-dtype equivalent.

        Only dtype conversions supported by --torq-convert-io-dtype are mapped.
        """
        dtype = get_dtype(fmt)
        convert_dtype = ConvertIODTypesPolicy.CONVERT_IO_DTYPES.get(dtype, dtype)
        if convert_dtype == dtype:
            return fmt
        return get_type_name(convert_dtype)

    @staticmethod
    def convert_io_dtype(dtype: np.dtype) -> np.dtype:
        """
        Maps a NumPy dtype to its --torq-convert-io-dtype equivalent.

        Only dtype conversions supported by --torq-convert-io-dtype are mapped.
        """
        dtype = np.dtype(dtype)
        return ConvertIODTypesPolicy.CONVERT_IO_DTYPES.get(dtype, dtype)
