"""
This module provides xfail utilities
"""

import os

import pytest


def load_xfails(module: str) -> list[str]:
    """
    Loads a list of xfail test names from a corresponding -xfails.txt file.

    The file is expected to contain one test name per line.
    Empty lines and lines starting with '#' (comments) are ignored.

    Args:
        module: The name of the module (e.g., 'test_host.py') for which to load
                xfails. The function will look for 'test_host-xfails.txt'.

    Returns:
        A list of strings, where each string is an xfail test name.
        Returns an empty list if the file is not found, cannot be read,
        or contains no valid xfail entries.
    """

    xfail_file_name = os.path.splitext(module)[0] + "-xfails.txt"

    xfail_tests: list[str] = []

    try:
        with open(xfail_file_name, "rt", encoding="utf-8") as f:
            # Read lines and strip leading/trailing whitespace
            lines = (line.strip() for line in f)

            # Filter out empty lines and comments
            xfail_tests = [
                test for test in lines
                if test and not test.startswith("#")
            ]

    except FileNotFoundError:
        return []
    except OSError:
        return []

    return xfail_tests

@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[pytest.Item]):
    module_xfails = {}
    for item in items:
        module = item.nodeid.split("::", 1)[0]

        xfails = module_xfails.get(module, None)
        if xfails is None:
            xfails = load_xfails(module)
            module_xfails[module] = xfails

        if item.nodeid in xfails:
            item.add_marker(pytest.mark.xfail(reason="known failure", strict=True))
