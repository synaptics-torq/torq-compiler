import pytest
import contextlib
import contextlib
import time
from typing import Dict, List
from dataclasses import dataclass
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
    scenario_log = ScenarioLog([test_file, request.node.name])

    yield scenario_log
    
    performance_report = request.config.getoption("--performance-report")
    
    if performance_report:

        # we need to take a lock because in xdist mode multiple workes
        # may try to write to the file at the same time
        with FileLock(performance_report + ".lock"):
            with open(performance_report, 'a') as fp:
                scenario_log.write(fp)


class Duration(contextlib.AbstractContextManager):
    def __init__(self, scenario: 'ScenarioLog', test_name: str):
        self.scenario = scenario
        self.test_name = test_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter() * 1000
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.perf_counter() * 1000
        self.scenario.log(self.test_name, self.start_time, end_time)


@dataclass
class Event:
    name: str
    start_time_ms: float
    end_time_ms: float


class ScenarioLog:
    def __init__(self, name):
        if isinstance(name, list):
            self.name = "::".join(name)
        else:
            self.name = name
        self.events = []

    def log(self, event_name: str, start_time_ms: float, end_time_ms: float):
        self.events.append(Event(event_name, start_time_ms, end_time_ms))

    def event(self, name: str) -> Duration:
        return Duration(self, name)
    
    def name_as_tuple(self):
        return tuple(self.name.split("::"))
    
    def write(self, fp):
        for event in self.events:
            duration = event.end_time_ms - event.start_time_ms
            fp.write(f"{self.name},{event.start_time_ms},{event.end_time_ms},{duration},{event.name}\n")


class PerformanceLog:
    def __init__(self):
        self.scenarios: List[ScenarioLog] = []

    def add_scenario(self, name: str) -> ScenarioLog:
        scenario = ScenarioLog(name)
        self.scenarios.append(scenario)
        return scenario

    def write(self, filepath: str):
        with open(filepath, 'w') as f:
            for scenario in self.scenarios:
                scenario.write(f)


def load_performance(filepath: str) -> PerformanceLog:
    performance = PerformanceLog()

    scenarios: Dict[str, ScenarioLog] = {}

    with open(filepath, 'r') as f:
        for line in f:
            scenario_name, start_time_ms_str, end_time_ms_str, _, event_name = line.strip().split(',')
            start_time_ms = float(start_time_ms_str)
            end_time_ms = float(end_time_ms_str)
            
            scenario = scenarios.get(scenario_name)

            if scenario is None:
                scenario = performance.add_scenario(scenario_name)                
                scenarios[scenario_name] = scenario

            scenario.log(event_name, start_time_ms, end_time_ms)

    return performance
