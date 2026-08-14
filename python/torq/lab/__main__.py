# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Entry point for ``python -m torq.lab``."""

import sys

from torq.lab.cli import main

if __name__ == "__main__":
    sys.exit(main())
