# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Weight quantization configuration: per-layer assignments and sensitivity results."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LayerQuantConfig:
    """Quantization settings for a single layer."""

    bits: int = 8  # 4, 8, or 16 (16 = bf16, no quantization)
    block_size: int = 32

    def __post_init__(self):
        if self.bits not in (4, 8, 16):
            raise ValueError(f"bits must be 4, 8, or 16, got {self.bits}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")


@dataclass
class QuantizationConfig:
    """Per-layer quantization configuration.

    ``layers`` maps layer names (e.g. ``/model/layers.0/attn/q_proj/MatMul``)
    to their quantization settings.  Layers not listed use ``default``.
    """

    default: LayerQuantConfig = field(default_factory=lambda: LayerQuantConfig(bits=8))
    layers: dict[str, LayerQuantConfig] = field(default_factory=dict)

    # --- factories -----------------------------------------------------------

    @classmethod
    def uniform(cls, bits: int, block_size: int = 32) -> QuantizationConfig:
        """All layers use the same quantization."""
        return cls(default=LayerQuantConfig(bits=bits, block_size=block_size))

    # --- per-layer access ----------------------------------------------------

    def get(self, layer_name: str) -> LayerQuantConfig:
        """Return the config for *layer_name*, falling back to ``default``."""
        return self.layers.get(layer_name, self.default)

    # --- serialisation -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "default": asdict(self.default),
            "layers": {k: asdict(v) for k, v in self.layers.items()},
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved quantization config to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> QuantizationConfig:
        """Load config from JSON."""
        data = json.loads(Path(path).read_text())
        default = LayerQuantConfig(**data.get("default", {}))
        layers = {k: LayerQuantConfig(**v) for k, v in data.get("layers", {}).items()}
        return cls(default=default, layers=layers)


@dataclass
class SensitivityResult:
    """Per-layer sensitivity metrics from analysis.

    Stores KL divergence for each tested bit-width so that ``to_config``
    can pick the most aggressive quantization whose KL is below threshold.
    """

    layer_name: str
    kl_divergence: dict[int, float] = field(default_factory=dict)  # {bits: kl}
    cosine_similarity: dict[int, float] = field(default_factory=dict)
    top1_match: dict[int, float] = field(default_factory=dict)
    classification: str = "LOW"  # LOW / MEDIUM / HIGH / CRITICAL


@dataclass
class SensitivityResults:
    """Collection of per-layer sensitivity results."""

    layers: list[SensitivityResult] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(r) for r in self.layers], indent=2))
        logger.info("Saved sensitivity results (%d layers) to %s", len(self.layers), path)

    @classmethod
    def load(cls, path: str | Path) -> SensitivityResults:
        data = json.loads(Path(path).read_text())
        layers = []
        for d in data:
            # JSON keys are strings — convert back to int
            for field_name in ("kl_divergence", "cosine_similarity", "top1_match"):
                if field_name in d and isinstance(d[field_name], dict):
                    d[field_name] = {int(k): v for k, v in d[field_name].items()}
            layers.append(SensitivityResult(**d))
        return cls(layers=layers)

    def to_config(
        self,
        bf16_threshold: float = 0.1,
        int8_threshold: float = 0.01,
        block_size: int = 32,
    ) -> QuantizationConfig:
        """Convert sensitivity results to a quantization config.

        For each layer, pick the most aggressive quantization (int4 first,
        then int8) whose KL divergence is below the relevant threshold.
        If none qualifies, assign bf16.
        """
        layers: dict[str, LayerQuantConfig] = {}
        for r in self.layers:
            # Try int4 first (most aggressive)
            kl_4 = r.kl_divergence.get(4, float("inf"))
            kl_8 = r.kl_divergence.get(8, float("inf"))

            if kl_4 <= int8_threshold:
                # int4 KL is low enough → use int4
                layers[r.layer_name] = LayerQuantConfig(bits=4, block_size=block_size)
            elif kl_8 <= bf16_threshold:
                # int8 KL is acceptable → use int8
                layers[r.layer_name] = LayerQuantConfig(bits=8, block_size=block_size)
            else:
                # Both too sensitive → bf16
                layers[r.layer_name] = LayerQuantConfig(bits=16, block_size=block_size)
        return QuantizationConfig(
            default=LayerQuantConfig(bits=4, block_size=block_size),
            layers=layers,
        )
