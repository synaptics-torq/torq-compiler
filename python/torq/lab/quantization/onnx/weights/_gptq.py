# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Load GPTQ INT4 weights from a compressed-tensors checkpoint.

The checkpoint is produced by ``llmcompressor`` (see
``torq-tools`` ``torq/models/qwen/int4quant/run_gptq.py``) with asymmetric INT4, in-feature
group quantization (``strategy="group"``, ``group_size=32``). Each quantized
``nn.Linear`` is stored as:

* ``<module>.weight_packed``      int32 ``[N, K/8]``, eight ``q + 8`` nibbles
* ``<module>.weight_scale``       float ``[N, K/group]``
* ``<module>.weight_zero_point``  signed int4 ``[N, K/group]`` (or packed
  8-per-int32 along axis 0 in older checkpoints)
* ``<module>.weight_shape``       ``[N, K]``

They are converted to :class:`QuantizedWeight` in the layout used by
:meth:`WeightQuantizer._quantize_to_dql`: signed int4 ``q`` ``[K, N]``, scales
and zero points ``[K/group, N]``, i.e. ``DequantizeLinear(axis=0,
block_size=group)`` feeding the MatMul ``B`` operand directly.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np

from ._quantization import QuantizedWeight

logger = logging.getLogger(__name__)

_LAYER_MATMUL_RE = re.compile(
    r"^/model/layers\.(?P<layer>\d+)/(?P<block>self_attn|mlp)/(?P<proj>\w+_proj)/MatMul$"
)


def module_key_from_matmul_name(matmul_name: str) -> str | None:
    """Map an optimum-exported ONNX MatMul node name to its HF module name."""
    if matmul_name == "/lm_head/MatMul":
        return "lm_head"
    match = _LAYER_MATMUL_RE.match(matmul_name)
    if match is None:
        return None
    return (
        f"model.layers.{match['layer']}.{match['block']}.{match['proj']}"
    )


def _unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    """Unpack int32 words into eight 4-bit values each (low nibble first)."""
    packed = packed.astype(np.uint32)
    shifts = np.arange(8, dtype=np.uint32) * 4
    nibbles = (packed[..., None] >> shifts) & 0xF
    return nibbles.reshape(*packed.shape[:-1], -1).astype(np.uint8)


def _zero_points_signed(zero_point: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(zero_point.shape) == shape:
        zp = zero_point.astype(np.int8)
    else:
        # Packed 8-per-int32 along axis 0: [ceil(N/8), G] -> [N, G]
        n_rows = shape[0]
        unpacked = _unpack_nibbles(zero_point.T.copy()).T[:n_rows]
        zp = (unpacked.astype(np.int16) - 8).astype(np.int8)
    if zp.min() < -8 or zp.max() > 7:
        raise ValueError("GPTQ zero points are outside signed int4 range [-8, 7]")
    return zp


def _safe_open(path: Path):
    # Optional dependency: only GPTQ checkpoints need safetensors.
    from safetensors import safe_open

    return safe_open(path, framework="np")


class GPTQCheckpoint:
    """Read-only view of a compressed-tensors GPTQ checkpoint directory."""

    def __init__(self, checkpoint_dir: str | os.PathLike):
        self._dir = Path(checkpoint_dir)
        index = self._dir / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            self._files = {k: self._dir / v for k, v in weight_map.items()}
        else:
            single = self._dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"No safetensors found in '{self._dir}'")
            with _safe_open(single) as handle:
                self._files = {k: single for k in handle.keys()}

    def has_module(self, module_key: str) -> bool:
        return f"{module_key}.weight_packed" in self._files

    def _get(self, key: str) -> np.ndarray:
        with _safe_open(self._files[key]) as handle:
            return handle.get_tensor(key)

    def load(
        self,
        module_key: str,
        column_index: np.ndarray | None = None,
    ) -> QuantizedWeight:
        """Return ``module_key`` as a signed-int4 :class:`QuantizedWeight`.

        Args:
            module_key: HF module name, e.g. ``model.layers.0.mlp.up_proj``.
            column_index: Optional output-feature (``N``) indices to keep, used
                for vocab-trimmed LM heads.
        """
        packed = self._get(f"{module_key}.weight_packed")
        scale = self._get(f"{module_key}.weight_scale").astype(np.float32)
        zero_point = self._get(f"{module_key}.weight_zero_point")
        n_out, k_in = (int(v) for v in self._get(f"{module_key}.weight_shape"))

        if scale.shape[0] != n_out:
            raise ValueError(
                f"{module_key}: scale shape {scale.shape} is not in-feature "
                f"grouped for weight [{n_out}, {k_in}]"
            )
        groups = scale.shape[1]
        if k_in % groups:
            raise ValueError(f"{module_key}: K={k_in} not divisible by {groups} groups")
        block_size = k_in // groups

        q = (_unpack_nibbles(packed)[:, :k_in].astype(np.int16) - 8).astype(np.int8)
        zp = _zero_points_signed(zero_point, (n_out, groups))

        if column_index is not None:
            q, scale, zp = q[column_index], scale[column_index], zp[column_index]

        return QuantizedWeight(
            quantized=np.ascontiguousarray(q.T),
            scales=np.ascontiguousarray(scale.T),
            zero_points=np.ascontiguousarray(zp.T),
            block_size=block_size,
            bits=4,
            original_shape=(k_in, q.shape[0]),
        )


def load_gptq_weights(
    checkpoint_dir: str | os.PathLike,
    matmul_info: list,
    lm_head_column_index: np.ndarray | None = None,
) -> dict[str, QuantizedWeight]:
    """Load GPTQ weights for the MatMuls found by ``_find_matmul_weights``.

    Returns a mapping from ONNX MatMul node name to :class:`QuantizedWeight`.
    The shape of every loaded weight is checked against the ONNX initializer it
    replaces. MatMuls without a GPTQ counterpart are left out, so the caller
    falls back to round-to-nearest for them.
    """
    checkpoint = GPTQCheckpoint(checkpoint_dir)
    weights: dict[str, QuantizedWeight] = {}

    for node, _, init in matmul_info:
        module_key = module_key_from_matmul_name(node.name)
        if module_key is None or not checkpoint.has_module(module_key):
            logger.warning("No GPTQ weights for '%s', using RTN", node.name)
            continue

        column_index = lm_head_column_index if module_key == "lm_head" else None
        qw = checkpoint.load(module_key, column_index)

        if tuple(qw.original_shape) != tuple(init.dims):
            raise ValueError(
                f"GPTQ weight for '{node.name}' has shape {qw.original_shape}, "
                f"ONNX initializer has {tuple(init.dims)}"
            )
        weights[node.name] = qw

    logger.info(
        "Loaded GPTQ INT4 weights for %d/%d MatMuls from '%s'",
        len(weights),
        len(matmul_info),
        checkpoint_dir,
    )
    return weights
