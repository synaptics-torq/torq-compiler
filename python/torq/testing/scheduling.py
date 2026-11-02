"""
Start the slow tests first, one per pytest-xdist worker.

A run with many workers ends when its slowest tests end. When they start late, the other
workers wait for them with nothing left to do. With --slow-tests-first, the tests listed in
tests/slow-tests.txt (and extras/tests/slow-tests.txt) are moved to the front of the collection,
spread so that each worker gets its share first.

pytest-xdist's worksteal scheduler first gives each worker a contiguous part of the
collection, so the slow tests are placed at the start of each part, not all at the start
(where they would all go to one worker).
"""

import re
from pathlib import Path

import pytest

_LIST_NAME = "slow-tests.txt"


def pytest_addoption(parser):
    parser.addoption(
        "--slow-tests-first",
        action="store_true",
        default=False,
        help=f"Start the tests listed in tests/{_LIST_NAME} first, one per pytest-xdist worker",
    )


def _load_patterns(rootdir: Path) -> list[re.Pattern]:
    """The patterns of the slow test lists, slowest first: test file name::test id, where * is
    the only wildcard."""
    patterns = []
    for path in (rootdir / "tests" / _LIST_NAME, rootdir / "extras" / "tests" / _LIST_NAME):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        patterns += [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return [re.compile(".*".join(map(re.escape, p.split("*")))) for p in patterns]


def _rank(item, patterns):
    """The index of the first pattern matching the item, or None."""
    module, test_name = item.nodeid.split("::", 1)
    name = f"{Path(module).name}::{test_name}"
    return next((i for i, p in enumerate(patterns) if p.fullmatch(name)), None)


def _worksteal_part_sizes(count: int, workers: int) -> list[int]:
    """The sizes of the parts worksteal first gives the workers (see its check_schedule)."""
    sizes = []
    for k in range(workers):
        sizes.append(count // (workers - k))
        count -= sizes[-1]
    return sizes


def order_slow_first(items, patterns, workers):
    """The items with the slow ones at the start of each worker's part, slowest first."""
    ranks = {id(item): _rank(item, patterns) for item in items}
    slow = sorted((i for i in items if ranks[id(i)] is not None), key=lambda i: ranks[id(i)])
    rest = iter([i for i in items if ranks[id(i)] is None])
    if workers <= 1:
        return slow + list(rest)
    parts = [slow[k::workers] for k in range(workers)]
    ordered = []
    for part, size in zip(parts, _worksteal_part_sizes(len(items), workers)):
        ordered += part
        ordered += [item for _, item in zip(range(size - len(part)), rest)]
    return ordered + list(rest)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    if not config.getoption("--slow-tests-first", default=False):
        return
    patterns = _load_patterns(Path(config.rootpath))
    if not patterns:
        return
    # every worker computes the same order from the same collection
    workers = getattr(config, "workerinput", {}).get("workercount", 1)
    items[:] = order_slow_first(items, patterns, workers)
