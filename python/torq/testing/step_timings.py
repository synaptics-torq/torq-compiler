"""
Records where the time of each test goes: the setup time of each fixture (compilation,
reference outputs, inputs, cache loads) and named steps inside fixtures (board runs,
profile annotation).

With --torq-step-timings=FILE, the times are sent to the controller with the test reports
(so this works with xdist), written to FILE as JSON and summarized at the end of the session
(the totals, and the slowest tests).
"""

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import pytest

# steps shorter than this are not recorded, to keep the report readable
_MIN_SECONDS = 0.05

# the summary lists only the slowest tests; FILE has them all
_SUMMARY_TESTS = 30

_steps_key = pytest.StashKey[list]()

# per test: phase durations and steps, collected on the controller
_session_results = {}


def pytest_addoption(parser):
    parser.addoption("--torq-step-timings", default=None,
                     help="Write the time spent in each fixture and step of each test to this JSON file, and print a summary")


def _enabled(config):
    return config.getoption("--torq-step-timings", default=None) is not None


def _item(request):
    # fixtures with a wider scope than the test are attributed to the test that set them up first
    return getattr(request, "_pyfuncitem", None) or getattr(request, "node", None)


def record_step(request, name, seconds):
    """Records that the current test spent `seconds` in step `name`."""
    if not _enabled(request.config) or seconds < _MIN_SECONDS:
        return
    item = _item(request)
    if item is not None:
        item.stash.setdefault(_steps_key, []).append((name, seconds))


@contextmanager
def step(request, name):
    """Times the enclosed block as step `name` of the current test."""
    start = time.perf_counter()
    try:
        yield
    finally:
        record_step(request, name, time.perf_counter() - start)


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    # pytest sets up the fixture's dependencies before this hook, so this is the fixture's own time
    start = time.perf_counter()
    yield
    record_step(request, f"fixture {fixturedef.argname}", time.perf_counter() - start)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if _enabled(item.config) and report.when == "teardown":
        steps = item.stash.get(_steps_key, [])
        report.user_properties.append(("torq_step_timings", steps))


def pytest_runtest_logreport(report):
    # runs on the controller (and without xdist), with the reports of every worker
    results = _session_results.setdefault(report.nodeid, {"steps": []})
    results[report.when] = report.duration
    for name, value in report.user_properties:
        if name == "torq_step_timings":
            results["steps"] = [list(s) for s in value]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _enabled(config) or hasattr(config, "workerinput") or not _session_results:
        return

    path = Path(config.getoption("--torq-step-timings"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(_session_results, fp, indent=2)

    tr = terminalreporter
    tr.section("step timings")

    totals = defaultdict(float)
    for results in _session_results.values():
        for name, seconds in results["steps"]:
            totals[name] += seconds
    phase_totals = {when: sum(r.get(when, 0) for r in _session_results.values())
                    for when in ("setup", "call", "teardown")}
    tr.write_line("Total over all tests (s; summed over the xdist workers, steps nested in fixtures are counted twice):")
    for when, seconds in phase_totals.items():
        tr.write_line(f"  {seconds:9.1f}  test {when}")
    for name, seconds in sorted(totals.items(), key=lambda kv: -kv[1]):
        tr.write_line(f"  {seconds:9.1f}  {name}")

    tr.write_line("")
    tr.write_line(f"The {_SUMMARY_TESTS} slowest tests (s; the 5 longest steps; all tests in {path}):")
    slowest = sorted(_session_results.items(),
                     key=lambda kv: -sum(kv[1].get(w, 0) for w in ("setup", "call", "teardown")))
    for nodeid, results in slowest[:_SUMMARY_TESTS]:
        phases = " ".join(f"{w} {results.get(w, 0):.1f}" for w in ("setup", "call", "teardown"))
        tr.write_line(f"  {nodeid}: {phases}")
        for name, seconds in sorted(results["steps"], key=lambda s: -s[1])[:5]:
            tr.write_line(f"      {seconds:9.1f}  {name}")
    tr.write_line(f"Step timings written to {path}")
