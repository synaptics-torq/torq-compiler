from dataclasses import dataclass

import ml_dtypes
import numpy as np
import pytest

from .versioned_fixtures import (
    versioned_hashable_object_fixture,
    versioned_unhashable_object_fixture,
)

MLIR_TO_NP_TYPE_MAPPING = {
    'i1': np.dtype(bool),
    'i8': np.dtype(np.int8),
    'i16': np.dtype(np.int16),
    'i32': np.dtype(np.int32),
    'ui8': np.dtype(np.uint8),
    'ui16': np.dtype(np.uint16),
    'ui32': np.dtype(np.uint32),
    'ui64': np.dtype(np.uint64),
    'f16': np.dtype(np.float16),
    'f32': np.dtype(np.float32),
    'si8': np.dtype(np.int8),
    'si16': np.dtype(np.int16),
    'si64': np.dtype(np.int64),
    'si32': np.dtype(np.int32),
    'bf16': np.dtype(ml_dtypes.bfloat16)
}
NP_TO_MLIR_TYPE_MAPPING = {
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
            raise pytest.UsageError(
                "--convert-io-dtypes: 'all' cannot be combined with specific indices"
            )
        has_include = any(not arg.startswith("!") for arg in args)
        has_exclude = any(arg.startswith("!") for arg in args)
        if has_include and has_exclude:
            raise pytest.UsageError("--convert-io-dtypes cannot mix included and excluded indices")
        mode = "exclude" if has_exclude else "include"
        inputs: set[int] = set()
        outputs: set[int] = set()
        for arg in args:
            raw = arg[1:] if arg.startswith("!") else arg
            try:
                kind, idx_str = raw.split(":", 1)
            except ValueError:
                raise pytest.UsageError(f"--convert-io-dtypes: expected input:IDX or output:IDX, got {arg!r}")
            if kind not in {"input", "output"}:
                raise pytest.UsageError(f"--convert-io-dtypes: expected 'input' or 'output', got {kind!r}")
            try:
                idx = int(idx_str)
            except ValueError:
                raise pytest.UsageError(f"--convert-io-dtypes: expected integer index, got {idx_str!r}")
            if idx < 0:
                raise pytest.UsageError(f"--convert-io-dtypes: index must be non-negative, got {idx}")
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


@versioned_hashable_object_fixture
def convert_io_dtypes_args(request):
    return list(request.config.getoption("--convert-io-dtypes"))


@versioned_unhashable_object_fixture
def convert_io_dtypes_policy(request, torq_compiler_options, convert_io_dtypes_args) -> ConvertIODTypesPolicy:
    torq_convert_dtypes = {"--torq-convert-dtypes", "--torq-convert-io-dtype"}.issubset(torq_compiler_options)
    return ConvertIODTypesPolicy.parse_from_args(convert_io_dtypes_args, torq_convert_dtypes)


def get_dtype(name):
    """
    Returns the numpy dtype corresponding to the given MLIR type name
    """
    
    try:
        return MLIR_TO_NP_TYPE_MAPPING[name]
    except KeyError:
        raise ValueError(f"Unsupported dtype {name}")


def get_type_name(dtype):
    """
    Returns the MLIR type name corresponding to the given NumPy dtype.
    """
    try:
        return NP_TO_MLIR_TYPE_MAPPING[np.dtype(dtype)]
    except KeyError:
        raise ValueError(f"Unsupported dtype {dtype}") from None


def cast_round_trip(data: np.ndarray, down_dtype: np.dtype, up_dtype: np.dtype):
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
