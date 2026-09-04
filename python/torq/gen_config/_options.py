# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Option container for the standalone ``torq-gen-config`` CLI.

Every option the discovery and full-model run flows read is collected once
into :class:`DiscoveryConfig`, either built from the CLI's
``argparse.Namespace`` (:meth:`DiscoveryConfig.from_argparse`) or constructed
directly in code.

This module must not import ``torq.testing`` (or any test framework) so the
wheel ships it cleanly.
"""

import argparse
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class DiscoveryConfig:
    """All options consumed by the discovery and full-model run flows.

    Defaults match the ``torq-gen-config discover``/``run`` CLI flag defaults.
    """

    # --- Model I/O ---
    model_path: Optional[str] = None
    output_dir: Optional[str] = None
    model_dir: Optional[str] = None
    model_filter: Optional[str] = None
    log_file: Optional[str] = None

    # --- Discovery behavior ---
    skip_mode: bool = False
    skip_executors: Optional[str] = None
    skip_ops: Optional[str] = None
    auto_convert_bf16: bool = False
    save_bf16_model: Optional[str] = None
    auto_convert_int32: bool = False
    subgraph_from: Optional[str] = None
    subgraph_to: Optional[str] = None
    collect_timing: bool = False
    timing_runs: int = 1
    recommend_by_timing: bool = False
    dedup_layers: bool = False
    # Show detailed logs (JSON cache activity, MLIR conversion, comparison
    # metrics, skip reasons); mirrors the CLI's --verbose flag.
    verbose: bool = False

    # --- Quantization (torq.lab.quantize_onnx._ONNX_QUANTIZATION_OPTIONS) ---
    quantize: bool = False
    per_channel: bool = False
    full_integer: bool = False
    quant_format: str = "qdq"
    quant_dtype: str = "A8W8"

    # --- Chip/runtime/compiler ---
    torq_hw: str = "default"
    runtime_hw_type: str = "sim"
    debug_ir: Union[bool, str] = False
    compiler_options: List[str] = field(default_factory=list)
    runtime_options: List[str] = field(default_factory=list)
    trace_buffers: bool = False
    debug_torq_compiler: int = 0
    compiler_timeout: int = 60 * 5
    runtime_timeout: int = 60 * 4
    ignore_binary_mtime: bool = False
    convert_io_dtypes: List[str] = field(default_factory=lambda: ["all"])
    torq_addr: Optional[str] = None
    torq_port: int = 22
    torq_private_key: Optional[str] = None

    # --- Cache ---
    recompute_cache: bool = False
    debug_cache: bool = False
    # Root of the standalone artifact cache; None means the Cache default
    # ($TORQ_GEN_CONFIG_CACHE_DIR or ${XDG_CACHE_HOME:-~/.cache}/torq-gen-config).
    cache_dir: Optional[str] = None

    @property
    def skip_executors_list(self) -> List[str]:
        """Parsed --skip-executors value (e.g. "nss,css" -> ["nss", "css"])."""
        if not self.skip_executors:
            return []
        return [e.strip() for e in self.skip_executors.split(",") if e.strip()]

    @property
    def skip_ops_list(self) -> List[str]:
        """Parsed --skip-ops value (e.g. "MaxPool,Add" -> ["MaxPool", "Add"])."""
        if not self.skip_ops:
            return []
        return [o.strip() for o in self.skip_ops.split(",") if o.strip()]

    @property
    def subgraph_mode(self) -> bool:
        """True when both --subgraph-from and --subgraph-to are set."""
        return bool(self.subgraph_from and self.subgraph_to)

    @property
    def cache(self):
        """Memoized standalone artifact cache.

        Provides the ``mkdir(name)``/``get``/``set`` surface used by the
        case-generation code (``generate_onnx_layers_from_file`` and the
        subgraph MLIR-mapping cache). Creation is lazy so that merely
        constructing a config has no filesystem side effects.
        """
        cache = self.__dict__.get("_cache_obj")
        if cache is None:
            from torq.gen_config._cache import Cache

            cache = Cache(self.cache_dir)
            self.__dict__["_cache_obj"] = cache
        return cache

    @classmethod
    def from_argparse(cls, args: argparse.Namespace) -> "DiscoveryConfig":
        """Build a config from the torq-gen-config CLI's parsed arguments.

        Reads the canonical argparse dest names produced by ``cli.py``'s
        parsers; absent attributes fall back to the dataclass defaults.
        """

        def first(*names: str, default: Any = None) -> Any:
            for name in names:
                value = getattr(args, name, None)
                if value is not None:
                    return value
            return default

        debug_ir = first("debug_ir")
        # The CLI uses None for "not passed"; normalize to False.
        if debug_ir is None:
            debug_ir = False

        return cls(
            model_path=first("model"),
            output_dir=first("output_dir"),
            model_dir=first("model_dir"),
            model_filter=first("model_filter"),
            log_file=first("log_file"),
            skip_mode=bool(first("skip_mode", default=False)),
            skip_executors=first("skip_executors"),
            skip_ops=first("skip_ops"),
            auto_convert_bf16=bool(first("auto_convert_bf16", default=False)),
            save_bf16_model=first("save_bf16_model"),
            auto_convert_int32=bool(first("auto_convert_int32", default=False)),
            subgraph_from=first("subgraph_from"),
            subgraph_to=first("subgraph_to"),
            collect_timing=bool(first("collect_timing", default=False)),
            timing_runs=first("timing_runs") or 1,
            recommend_by_timing=bool(first("recommend_by_timing", default=False)),
            dedup_layers=bool(first("dedup_layers", default=False)),
            verbose=bool(first("verbose", default=False)),
            quantize=bool(first("quantize", default=False)),
            per_channel=bool(first("per_channel", default=False)),
            full_integer=bool(first("full_integer", default=False)),
            quant_format=first("quant_format", default="qdq"),
            quant_dtype=first("quant_dtype", default="A8W8"),
            torq_hw=first("torq_hw", default="default"),
            runtime_hw_type=first("torq_hw_type", "runtime_hw_type", default="sim"),
            debug_ir=debug_ir,
            compiler_options=list(first("compiler_option", default=[]) or []),
            runtime_options=list(first("runtime_option", default=[]) or []),
            trace_buffers=bool(first("trace_buffers", default=False)),
            debug_torq_compiler=int(first("debug_torq_compiler", default=0)),
            compiler_timeout=int(first("compiler_timeout", default=60 * 5)),
            runtime_timeout=int(first("runtime_timeout", default=60 * 4)),
            ignore_binary_mtime=bool(first("ignore_binary_mtime", default=False)),
            convert_io_dtypes=list(first("convert_io_dtypes", default=["all"])),
            torq_addr=first("torq_addr"),
            torq_port=int(first("torq_port", default=22)),
            torq_private_key=first("torq_private_key"),
            recompute_cache=bool(first("recompute_cache", default=False)),
            debug_cache=bool(first("debug_cache", default=False)),
            cache_dir=first("cache_dir"),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain dict of all fields (JSON-friendly values)."""
        return asdict(self)
