# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-model tools: importing, dtype conversion, shape conversion, and extraction.

The domain boundary for working from source models (``.onnx`` / ``.tflite``)
before compilation: ``importers`` convert source models to MLIR,
``dtype_conversion`` rewrites model dtypes to Torq-supported ones,
``shape_conversion`` makes dynamic TFLite models static, and
``extraction`` splits models into layers and subgraphs.
"""
