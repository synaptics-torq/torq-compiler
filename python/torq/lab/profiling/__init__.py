# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile/run profiling: annotation, debug info, locations, and Perfetto traces.

``annotate`` is the high-level annotation orchestration used by the pipeline;
``debug_info`` models and parses the ``--torq-debug-info`` output; ``location``
parses MLIR locations; ``perfetto`` generates Perfetto traces and metrics;
``report`` renders the combined self-contained HTML report. The heavy
dependencies (pandas, protobuf, XlsxWriter) ship only with the
``torq-compiler[profile]`` extra and are imported lazily.
"""
