# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Process-wide verbosity flag for optional progress/diagnostic logging.

Entry points (for example the ``torq-gen-config`` CLI) call
:func:`set_verbose` from their ``--verbose`` flag; library code gates
detailed logs behind :func:`is_verbose` so normal runs stay quiet.
"""

_VERBOSE = False


def set_verbose(enabled: bool) -> None:
    """Enable or disable verbose logging."""
    global _VERBOSE
    _VERBOSE = enabled


def is_verbose() -> bool:
    """Whether verbose logging is enabled."""
    return _VERBOSE
