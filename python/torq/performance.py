import contextlib
import time
from typing import Dict, List
from dataclasses import dataclass


class Duration(contextlib.AbstractContextManager):
    def __init__(self, scenario: 'Scenario', test_name: str):
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
