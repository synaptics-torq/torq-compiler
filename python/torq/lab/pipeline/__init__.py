# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pipeline execution: orchestration, artifacts, I/O, remote targets, and tools.

``workflow`` owns ``ModelPipeline`` and the pipeline configuration/result
types; ``artifacts`` owns artifact discovery and manifest persistence;
``io`` owns MLIR I/O specs, dtype mapping, and data materialization;
``remote`` owns remote staging and the SSH/ADB transport; ``tools`` owns
executable discovery and subprocess execution.
"""
