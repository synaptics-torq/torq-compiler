# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""torq.lab: generic MLIR/VMFB compile/run/profile orchestration.

This package owns release-facing orchestration around ``torq-compile`` and
``torq-run-module``: local and remote execution, artifact layout, input/output
utilities, and result comparison. It ships in the wheel, so it must not import
``torq.testing`` or any test framework.
"""

__all__ = ["main"]


def __getattr__(name):
    """Expose ``torq.lab.main`` lazily (PEP 562 module-level ``__getattr__``).

    ``main`` is the ``torq-lab`` CLI entry point. We resolve it on first access
    rather than importing it at module top so that ``import torq.lab`` and
    ``from torq.lab import <submodule>`` stay cheap and side-effect-free: they do
    not pull in ``torq.lab.cli`` (and its whole argparse graph + transitive
    pipeline imports). ``torq.lab.main`` triggers this hook and returns the CLI
    ``main``; any other attribute raises ``AttributeError`` as usual.
    """
    if name == "main":
        from torq.lab.cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
