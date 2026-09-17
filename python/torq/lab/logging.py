# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Logging utilities for Torq.

Entry points (for example the ``torq-gen-config`` CLI) call
:func:`set_verbose` from their ``--verbose`` flag; library code gates
detailed logs behind :func:`is_verbose` so normal runs stay quiet.
"""

import argparse
import logging
import os
from collections.abc import Iterable

__all__ = [
    "add_logging_args",
    "configure_logging",
    "set_verbose",
    "is_verbose",
]

_ORT_QUANTIZATION_PATH = os.path.join("onnxruntime", "quantization")
_VERBOSE = False


def set_verbose(enabled: bool) -> None:
    """Enable or disable verbose logging."""
    global _VERBOSE
    _VERBOSE = enabled


def is_verbose() -> bool:
    """Whether verbose logging is enabled."""
    return _VERBOSE


def _drop_ort_quantizer_noise(record: logging.LogRecord) -> bool:
    """Silence onnxruntime.quantization's per-tensor/per-node ``logging.info(...)`` chatter.

    That module logs via the bare ``logging`` functions rather than a named logger, so the
    records show up as ``root`` in our own log format; torq's own code never does this, so
    filtering on ``record.name == "root"`` only ever catches vendor noise like this.
    """
    return not (
        record.name == "root"
        and record.levelno <= logging.INFO
        and _ORT_QUANTIZATION_PATH in record.pathname
    )


def add_logging_args(parser: argparse.ArgumentParser):
    """
    Add Torq logging args to an args parser.

    Args:
        parser: An ``argparse.ArgumentParser`` instance.
    """

    parser.add_argument(
        "--logging",
        type=lambda s: s.upper(),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity: %(choices)s (default: %(default)s)"
    )


def configure_logging(
    verbosity: str,
    loggers: Iterable[logging.Logger] | None = None,
    handlers: Iterable[logging.Handler] | None = None
):
    """
    Configure Torq logging.

    **Note**: Formatters and handlers in provided ``loggers`` will be overwritten.

    Args:
        verbosity: Logging level as a string.
        loggers: An optional iterable of ``logging.Logger`` instances to configure. If ``None``, the root is used.
        handlers: An optional iterable of ``logging.Handler`` instances to attach to each logger. If ``None``, a single ``logging.StreamHandler`` is used.

    Raises:
        ValueError: If ``verbosity`` is not a valid logging level name.
    """

    level = getattr(logging, verbosity.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {verbosity}")

    handlers = handlers or [logging.StreamHandler()]
    for handler in handlers:
        formatter = logging.Formatter("Torq-tools [%(levelname)-8s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        handler.addFilter(_drop_ort_quantizer_noise)

    loggers = loggers or [logging.getLogger()]
    for logger in loggers:
        logger.setLevel(level)
        logger.handlers.clear()
        for handler in handlers:
            logger.addHandler(handler)
