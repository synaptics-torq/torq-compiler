"""
Records where the time before the first test goes: the startup of each pytest process (imports
of the plugins, e.g. TensorFlow and torch) and the collection of each test module (its import
and the generation of its test cases, which may build models or extract layers).

With --torq-collection-timings=FILE, each process (each xdist worker) measures its own startup
and collection and sends them to the controller, which writes them to FILE as JSON and prints a
summary: the time of each worker, and the slowest modules. For each module the summary gives the
wall time and the CPU time of the process: a wall time well above the CPU time means the worker
waited (for the CPU shared with the other workers, a lock held by another worker, or a download).
"""

import json
import os
import time
from pathlib import Path

import pytest

# the summary lists only the slowest modules; FILE has them all
_SUMMARY_MODULES = 25

_WORKEROUTPUT_KEY = "torq_collection_timings"

# timings of this process
_timings = {"modules": {}}

# timings of each process, collected on the controller
_session_timings = {}


def pytest_addoption(parser):
    parser.addoption("--torq-collection-timings", default=None,
                     help="Write the startup and collection time of each process (each xdist worker) and test module to this JSON file, and print a summary")


def _enabled(config):
    return config.getoption("--torq-collection-timings", default=None) is not None


def _process_start_time():
    """Wall time when this process started, if the system gives it (Linux)."""
    try:
        with open("/proc/self/stat") as f:
            # the fields after the command name, which is in parentheses and may contain spaces
            fields = f.read().rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])
        with open("/proc/stat") as f:
            boot_time = next(int(line.split()[1]) for line in f if line.startswith("btime "))
        return boot_time + start_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def _worker_name(config):
    workerinput = getattr(config, "workerinput", None)
    return workerinput["workerid"] if workerinput else "main"


def pytest_sessionstart(session):
    if not _enabled(session.config):
        return
    now = time.time()
    start = _process_start_time()
    _timings["startup"] = now - start if start is not None else None
    _timings["session_start"] = now
    _timings["session_start_cpu"] = time.process_time()


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    if not isinstance(collector, pytest.Module) or not _enabled(collector.config):
        yield
        return
    wall, cpu = time.perf_counter(), time.process_time()
    yield
    _timings["modules"][collector.nodeid] = {"wall": time.perf_counter() - wall,
                                             "cpu": time.process_time() - cpu}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session, config, items):
    if _enabled(config):
        # before the deselection by -m and -k
        _timings["collected"] = len(items)


def pytest_collection_finish(session):
    if not _enabled(session.config) or "session_start" not in _timings:
        return
    _timings["collection"] = time.time() - _timings["session_start"]
    _timings["collection_cpu"] = time.process_time() - _timings["session_start_cpu"]
    _timings["selected"] = len(session.items)


def pytest_sessionfinish(session):
    if not _enabled(session.config):
        return
    if hasattr(session.config, "workeroutput"):
        # sent to the controller when the worker shuts down
        session.config.workeroutput[_WORKEROUTPUT_KEY] = json.dumps(_timings)
    elif "collection" in _timings:
        # without xdist (under xdist the controller doesn't collect)
        _session_timings[_worker_name(session.config)] = _timings


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    # xdist hook, on the controller
    data = getattr(node, "workeroutput", {}).get(_WORKEROUTPUT_KEY)
    if data:
        _session_timings[node.workerinput["workerid"]] = json.loads(data)


def _fmt(seconds):
    return f"{seconds:7.1f}" if seconds is not None else "      -"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _enabled(config) or hasattr(config, "workerinput") or not _session_timings:
        return

    path = Path(config.getoption("--torq-collection-timings"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(_session_timings, fp, indent=2)

    tr = terminalreporter
    tr.section("collection timings")
    tr.write_line(f"Per process (s; {os.cpu_count()} CPUs; startup: from the process start to the session start, "
                  "'-' where the system doesn't give the process start):")
    tr.write_line("   startup  collection  (CPU)  collected  selected  process")
    workers = sorted(_session_timings.items(), key=lambda kv: -(kv[1].get("collection") or 0))
    for name, t in workers:
        tr.write_line(f"   {_fmt(t.get('startup'))}  {_fmt(t.get('collection'))}  {_fmt(t.get('collection_cpu'))}"
                      f"  {t.get('collected', 0):9d}  {t.get('selected', 0):8d}  {name}")

    modules = {}
    for t in _session_timings.values():
        for nodeid, m in t["modules"].items():
            modules.setdefault(nodeid, []).append(m)
    tr.write_line("")
    tr.write_line(f"The {_SUMMARY_MODULES} slowest modules to collect (s, over the processes; all modules in {path}):")
    tr.write_line("  max wall  mean wall  mean CPU  module")
    slowest = sorted(modules.items(), key=lambda kv: -max(m["wall"] for m in kv[1]))
    for nodeid, ms in slowest[:_SUMMARY_MODULES]:
        tr.write_line(f"   {_fmt(max(m['wall'] for m in ms))}    {_fmt(sum(m['wall'] for m in ms) / len(ms))}   "
                      f"{_fmt(sum(m['cpu'] for m in ms) / len(ms))}  {nodeid}")
    tr.write_line(f"Collection timings written to {path}")
