# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/licenses/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TORQ executor config generation package."""

__all__ = ["main"]


def __getattr__(name):
    """Expose ``torq.gen_config.main`` lazily (PEP 562 module-level ``__getattr__``).

    ``main`` is the ``torq-gen-config`` CLI entry point. We resolve it on first
    access rather than importing it at module top so that ``import
    torq.gen_config`` and ``from torq.gen_config import <submodule>`` stay cheap
    and do not pull in ``torq.gen_config.cli``'s transitive deps (onnx /
    onnxruntime via ``torq.lab.quantization.onnx.static``). This lets lightweight
    submodules such as ``_options`` and ``_cache`` import in environments
    without the ``[onnx]`` extra installed.
    """
    if name == "main":
        from torq.gen_config.cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
