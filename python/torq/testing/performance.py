import contextlib
import time
from pathlib import Path
import json


"""
This module provides helper functions to profile tests
"""


METRICS_FILE_NAME = 'performance_metrics.json'


def clear_measurements(fixture_dir: str):
    """
    Clears the performance metrics json file in the given fixture directory.
    """

    f = Path(fixture_dir) / METRICS_FILE_NAME

    if f.exists():
        f.unlink()


@contextlib.contextmanager
def measure_time(fixture_dir: str, metric_name: str):
    """
    Returns a context manager that measures the time taken
    to execute the code within the context and saves it to 
    a json file in the given fixture directory.
    """

    start_time = time.perf_counter_ns()

    yield

    end_time = time.perf_counter_ns()
    
    elapsed_time = end_time - start_time

    append_measurement(fixture_dir, metric_name, elapsed_time)


def append_measurements(fixture_dir: str, measurements: dict):
    """
    Appends multiple measurements to the performance metrics json file in the given fixture directory.
    """

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
    """
    Appends a single measurement to the performance metrics json file in the given fixture directory.
    """

    append_measurements(fixture_dir, {metric_name: metric_value})


def record_measurements(request, property_name: str, file_path: str):
    """
    Records the measurments from the performance metrics file
    into the pytest request's record_property fixture.
    """

    f = Path(file_path) / METRICS_FILE_NAME

    if not f.exists():
        return

    with open(f, 'r') as fp:
        data = json.load(fp)

    record_property = request.getfixturevalue("record_property")
    
    record_property(property_name, data)
