# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Output verification: numeric comparison and reference execution.

``compare`` is the pure numeric core (tolerances, per-tensor results);
``reference`` executes a reference backend (ONNX Runtime / numpy / IREE
llvm-cpu) to produce expected outputs. Reference execution is independent
from output comparison.
"""
