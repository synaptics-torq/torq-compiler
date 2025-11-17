import pytest
import torq.performance
from filelock import FileLock

"""
This module provides support for measuring the performance of the compiler and 
the runtime during test execution.
"""


def pytest_addoption(parser):
    parser.addoption("--performance-report", action="store", default=False, help="Path to write test performance report CSV")


def pytest_sessionstart(session):    

    # run this code only when runnning on the controller when using xdist
    # we don't want each worker to truncate the file
    if hasattr(session.config, "workerinput"):
        return

    performance_report = session.config.getoption("--performance-report")

    if performance_report:

        # Truncate the performance report file to ensure we clear out
        # any previous result that may be present. The test will append
        # to this file
        with open(performance_report, 'w') as fp:
            pass


@pytest.fixture
def scenario_log(request):    
    test_file = request.node.fspath.basename if hasattr(request.node, "fspath") else "unknown"
    scenario_log = torq.performance.ScenarioLog([test_file, request.node.name])

    yield scenario_log
    
    performance_report = request.config.getoption("--performance-report")
    
    if performance_report:

        # we need to take a lock because in xdist mode multiple workes
        # may try to write to the file at the same time
        with FileLock(performance_report + ".lock"):
            with open(performance_report, 'a') as fp:
                scenario_log.write(fp)
