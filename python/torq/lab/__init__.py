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

The ``torq-lab`` CLI entry point is ``torq.lab.cli.commands.main``;
``python -m torq.lab`` reaches the same function through ``__main__``.

This module also defines the package-wide shared types that no single
functional owner takes: the base error (:class:`LabError`) and the generic
named-case container (:class:`Case`).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List


class LabError(Exception):
    """Base class for torq.lab errors."""


@dataclass
class Case:
    """Named case container: a ``name`` plus an arbitrary ``data`` payload.

    Used to parametrize discovery layers and runs with a non-exhaustive subset
    of parameter combinations: consumers generate one item per ``Case`` instead
    of the full cross-product.
    """

    name: str
    data: Any


def get_test_cases_from_files(files: Path) -> List[Case]:
    """Generate one :class:`Case` per file, using the file name as case name."""

    cases = []

    for file_path in files:
        cases.append(Case(file_path.name, file_path))

    return cases
