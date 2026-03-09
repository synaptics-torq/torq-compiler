import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import ml_dtypes
try:
    import iree.runtime as iree_rt
    IREE_RT_AVAILABLE = True
except ImportError:
    IREE_RT_AVAILABLE = False

__all__ = [
    "random_inputs_from_info",
    "get_inputs_and_outputs",
    "TensorInfo"
]


_DTYPE_MAP: dict[str, np.dtype] = {
    "f16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
    "bf16": np.dtype(ml_dtypes.bfloat16),
    "i1": np.dtype(np.bool_),
    "i8": np.dtype(np.int8),
    "i16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "i64": np.dtype(np.int64),
    "si8": np.dtype(np.int8),
    "si16": np.dtype(np.int16),
    "si32": np.dtype(np.int32),
    "si64": np.dtype(np.int64),
    "ui8": np.dtype(np.uint8),
    "ui16": np.dtype(np.uint16),
    "ui32": np.dtype(np.uint32),
    "ui64": np.dtype(np.uint64),
}

_TENSOR_RE = re.compile(r"tensor<([^>]+)>")


@dataclass
class TensorInfo:
    """Dtype and shape metadata for a tensor."""

    dtype: npt.DTypeLike
    shape: list[int | str]

    def is_valid(self) -> bool:
        """Return True if every dimension is a an integer."""
        return all(isinstance(d, int) for d in self.shape)


def _parse_tensor_type(tensor_type: str) -> TensorInfo:
    """Parse an MLIR-style tensor type string (e.g. ``1x3x224xf32``) into a TensorInfo.

    Args:
        tensor_type: Dimension-x-dtype string extracted from a tensor<…> descriptor.

    Returns:
        Populated TensorInfo with numpy dtype and integer shape.

    Raises:
        ValueError: If the string has fewer than two parts, contains an
            unsupported dtype, or includes non-integer dimensions.
    """
    parts = tensor_type.strip().split("x")
    if len(parts) < 2:
        raise ValueError(f"Unexpected tensor type payload: {tensor_type!r}")

    dtype_str = parts[-1].strip()
    dtype = _DTYPE_MAP.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_str!r}")
    dims_raw = parts[:-1]

    shape: list[int | str] = []
    for d in dims_raw:
        d = d.strip()
        if d.isdigit():
            shape.append(int(d))
        else:
            shape.append(d)

    t_info = TensorInfo(dtype=dtype, shape=shape)
    if not t_info.is_valid():
        raise ValueError(f"Invalid tensor type: {tensor_type!r}")
    return t_info


def random_inputs_from_info(inputs_info: Iterable[TensorInfo]) -> list[np.ndarray]:
    """Generate random numpy input data from given input tensors metadata.

    Args:
        inputs_info: Iterable of TensorInfo describing each input tensor.

    Returns:
        List of numpy arrays with appropriate shapes and dtypes.

    Raises:
        ValueError: If an unsupported dtype is encountered.
    """
    rng = np.random.default_rng()
    inputs_data = []
    for info in inputs_info:
        if np.issubdtype(info.dtype, np.floating) or info.dtype == np.dtype(ml_dtypes.bfloat16):
            inputs_data.append(rng.standard_normal(info.shape).astype(info.dtype))
        elif np.issubdtype(info.dtype, np.integer):
            iinfo = np.iinfo(info.dtype)
            inputs_data.append(rng.integers(iinfo.min, iinfo.max, size=info.shape, dtype=info.dtype))
        elif np.issubdtype(info.dtype, np.bool_):
            inputs_data.append(rng.integers(0, 2, size=info.shape).astype(info.dtype))
        else:
            raise ValueError(f"Unsupported dtype for random input generation: {info.dtype}")
    return inputs_data


def get_inputs_and_outputs(invoker: iree_rt.FunctionInvoker, function_name: str) -> tuple[list[TensorInfo], list[TensorInfo]] | None:
    """
    Extract inputs and outputs info from a VMFB function.

    Args:
        invoker: IREE FunctionInvoker instance
        function_name: Name of the function to extract info from

    Returns:
        Tuple of (inputs, outputs) TensorInfo lists, or None if reflection data unavailable.

    Raises:
        RuntimeError: If the IREE Runtime python API is not installed.
        ValueError: If the ABI signature cannot be parsed.
    """

    if not IREE_RT_AVAILABLE:
        raise RuntimeError("IREE Runtime python API not available in environment")

    vf = invoker.vm_function
    refl = getattr(vf, "reflection", None) or {}
    sig = refl.get("iree.abi.declaration")
    if not refl or not sig:
        return None

    m_in = re.search(rf"@{function_name}\s*\((.*?)\)\s*->", sig)
    if not m_in:
        raise ValueError("Couldn't find input argument list after '@main'")
    inputs_str = m_in.group(1)

    m_out = re.search(r"->\s*\((.*?)\)\s*$", sig)
    if not m_out:
        raise ValueError("Couldn't find output result list after '->'")
    outputs_str = m_out.group(1)

    inputs_meta = [_parse_tensor_type(t) for t in _TENSOR_RE.findall(inputs_str)]
    outputs_meta = [_parse_tensor_type(t) for t in _TENSOR_RE.findall(outputs_str)]
    return inputs_meta, outputs_meta
