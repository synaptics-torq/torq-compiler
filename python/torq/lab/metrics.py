# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generic measurement helpers: time a block and persist metrics to JSON.

The ``record_property`` adapter is in ``torq.testing.performance``.
"""

import contextlib
import json
import time
from pathlib import Path

METRICS_FILE_NAME = 'performance_metrics.json'


def clear_measurements(fixture_dir: str):
    """Clear the performance metrics json file in the given directory."""
    f = Path(fixture_dir) / METRICS_FILE_NAME
    if f.exists():
        f.unlink()


@contextlib.contextmanager
def measure_time(fixture_dir: str, metric_name: str):
    """Measure wall time of the wrapped block and append it to the metrics file."""
    start_time = time.perf_counter_ns()
    yield
    end_time = time.perf_counter_ns()
    elapsed_time = end_time - start_time
    append_measurement(fixture_dir, metric_name, elapsed_time)


def append_measurements(fixture_dir: str, measurements: dict):
    """Append multiple measurements to the metrics file in the given directory."""
    f = Path(fixture_dir) / METRICS_FILE_NAME

    data = {}
    if f.exists():
        with open(f, 'r') as fp:
            data = json.load(fp)

    for metric_name, metric_value in measurements.items():
        data[metric_name] = metric_value

    with open(f, 'w') as fp:
        json.dump(data, fp)


def append_measurement(fixture_dir: str, metric_name: str, metric_value):
    """Append a single measurement to the metrics file in the given directory."""
    append_measurements(fixture_dir, {metric_name: metric_value})
