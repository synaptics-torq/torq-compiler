# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python wrappers for the Torq compiler CLI tools.

This top-level API provides access to the ``torq-compile`` tool, which compiles
MLIR via the Torq/IREE compiler to a VM FlatBuffer.

Example
~~~~~~~

.. code-block:: python

  import torq.compiler.tools

  binary = torq.compiler.tools.compile_str(
      mlir_asm, target_backends=["torq"])
"""

from .core import *
from .binaries import CompilerToolError, find_tool
