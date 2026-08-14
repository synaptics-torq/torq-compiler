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


"""
This module provides helper functions to profile tests
"""


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
