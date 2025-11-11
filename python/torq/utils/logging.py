"""Logging utilities for Torq."""

import argparse
import logging
from collections.abc import Iterable

__all__ = [
    "add_logging_args",
    "configure_logging",
]


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
        formatter = logging.Formatter("Torq [%(levelname)-8s] %(name)s: %(message)s")
        handler.setFormatter(formatter)

    loggers = loggers or [logging.getLogger()]
    for logger in loggers:
        logger.setLevel(level)
        logger.handlers.clear()
        for handler in handlers:
            logger.addHandler(handler)
