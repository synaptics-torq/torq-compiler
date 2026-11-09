import json
from pathlib import Path

# The generic measurement helpers live in torq.lab.metrics; re-export them so
# torq.testing.performance keeps its public surface. record_measurements below is
# the pytest adapter that wires those metrics into record_property.
from torq.lab.metrics import (  # noqa: F401
    METRICS_FILE_NAME,
    append_measurement,
    append_measurements,
    clear_measurements,
    measure_time,
)

from .remote_cache import FROM_REMOTE_CACHE_MARKER


"""
This module provides helper functions to profile tests
"""


# wall times measured when a result is computed: not reported for a result fetched from a test
# cache store (see remote_cache), as another machine measured them
MEASURED_TIMES = ("compilation_time", "e2e_inference_time")


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

    if (Path(file_path) / FROM_REMOTE_CACHE_MARKER).exists():
        data = {name: value for name, value in data.items() if name not in MEASURED_TIMES}

    record_property = request.getfixturevalue("record_property")
    
    record_property(property_name, data)
