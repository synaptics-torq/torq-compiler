"""
This module provides xfail utilities
"""

from pathlib import Path

import pytest
from dataclasses import dataclass


@dataclass
class XFailEntry:
    test_name: str    
    issue: str | None


@dataclass
class Comment:
    text: str


def _parse_xfails_file(xfail_file_path: Path) -> list[XFailEntry|Comment]:    

    xfail_tests = []

    try:
        with open(xfail_file_path, "rt", encoding="utf-8") as f:
            
            for line in f:

                line = line.strip()

                if not line:
                    xfail_tests.append(Comment(text=""))
                elif line.startswith("#"):
                    xfail_tests.append(Comment(text=line[1:].strip()))
                else:
                    fields = line.split(" ")                    

                    if len(fields) > 2:
                        raise Exception(f"Warning: invalid line in xfail file '{xfail_file_path}': '{line}'")

                    test_name = fields[0]

                    if len(fields) == 2:
                        issue = fields[1]
                    else:
                        issue = None               
                    
                    xfail_tests.append(XFailEntry(test_name=test_name, issue=issue))

        return xfail_tests

    except FileNotFoundError:
        return []
    except OSError:
        return []


def _get_xfail_file_paths(module: str) -> tuple[Path, Path]:
    module_path = Path(module)
    xfail_file_name = module_path.stem + "-xfails.txt"
    xfail_file_path = module_path.with_name(xfail_file_name)
    extra_xfail_file_path = module_path.parent.parent / "extras" / "xfail" / xfail_file_name
    return xfail_file_path, extra_xfail_file_path


def _parse_xfails(module: str) -> list[XFailEntry|Comment]:    
    xfail_file_path, extra_xfail_file_path = _get_xfail_file_paths(module)

    xfail_tests: list[XFailEntry] = []
    
    xfail_tests.extend(_parse_xfails_file(xfail_file_path))
    xfail_tests.extend(_parse_xfails_file(extra_xfail_file_path))

    return xfail_tests


def _write_xfails_file(xfail_file_path: Path, xfail_entries: list[XFailEntry|Comment]) -> None:

    print("writing: ", xfail_file_path)
    existing_entries = _parse_xfails_file(xfail_file_path)

    new_entries_by_name = {entry.test_name: entry for entry in xfail_entries if isinstance(entry, XFailEntry)}
    existing_test_names = {entry.test_name for entry in existing_entries if isinstance(entry, XFailEntry)}

    # create a new list of entries that contains all the existing comments and
    # the existing entries that are should be kept
    merged: list[XFailEntry | Comment] = []
    for entry in existing_entries:
        if isinstance(entry, Comment):
            merged.append(entry)
        elif entry.test_name in new_entries_by_name:
            merged.append(new_entries_by_name[entry.test_name])

    # append all the new entries that are not in the existing entries at the end of the file
    for entry in xfail_entries:
        if isinstance(entry, XFailEntry) and entry.test_name not in existing_test_names:
            merged.append(entry)

    xfail_entries = merged

    if not xfail_entries:
        # if there are no entries, remove the file if it exists
        try:
            xfail_file_path.unlink()            
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"Error removing xfail file '{xfail_file_path}': {e}")
        return

    try:

        xfail_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(xfail_file_path, "wt", encoding="utf-8") as f:
            for entry in xfail_entries:
                if isinstance(entry, Comment):
                    if entry.text:
                        f.write(f"# {entry.text}\n")
                    else:
                        f.write("\n")
                elif isinstance(entry, XFailEntry):
                    line = entry.test_name
                    if entry.issue:
                        line += f" {entry.issue}"
                    f.write(line + "\n")
                else:
                    raise RuntimeError(f"Unexpected entry type: {entry}")

    except OSError as e:
        print(f"Error writing xfail file '{xfail_file_path}': {e}")



def write_xfails(module: str, xfail_entries: list[XFailEntry|Comment], extras=False) -> None:
    """
    Updates the xfail entries in the -xfails.txt file of a module.
    
    Args:
        module: The name of the module (e.g., 'test_host.py') for which to write
                xfails. The function will write to 'test_host-xfails.txt'.
        xfail_entries: A list of XFailEntry and Comment objects to be written to the file.
        extras: If True, the entries will be written to the extras xfail file instead of the base xfail file.
    """    
    
    xfail_file_path, extra_xfail_file_path = _get_xfail_file_paths(module)

    if extras:
        xfail_file_path = extra_xfail_file_path

    _write_xfails_file(xfail_file_path, xfail_entries)    


def load_xfails(module: str) -> dict[str, XFailEntry]:
    """
    Loads a list of xfail test names from a corresponding -xfails.txt file.

    The file is expected to contain one test name per line.
    Empty lines and lines starting with '#' (comments) are ignored.

    Args:
        module: The name of the module (e.g., 'test_host.py') for which to load
                xfails. The function will look for 'test_host-xfails.txt'.

    Returns:
        A dictionary mapping test names to XFailEntry objects, where each entry represents an xfail test.
        Returns an empty dictionary if the file is not found, cannot be read,
        or contains no valid xfail entries.
    """

    return {entry.test_name: entry for entry in _parse_xfails(module) if isinstance(entry, XFailEntry)}


def pytest_addoption(parser):
    parser.addoption(
        "--try-xfails",
        action="store_true",
        default=False,
        help="Run xfail tests instead of skipping them",
    )
    parser.addoption(
        "--issue",
        default=None,
        help="Run only tests related to the specified issue (e.g., --issue=1234)"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "issue(number): mark test related to issue number (e.g., issue(1234))"
    )


def pytest_runtest_setup(item):

    # skip all tests not marked as related to a given issue if --issue is specified

    selected_issue = item.config.getoption("--issue")

    if not selected_issue:
        return
    
    test_issues = [mark.args[0] for mark in item.iter_markers(name="issue")]

    if not test_issues or selected_issue not in test_issues:
        pytest.skip("not related to selected issue")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):

    # by default xfail tests are not run to save time, but when --try-xfails
    # is specified or --issue is specified, they are run (this allows to
    # detect XPASS tests)
    run_xfails = config.getoption("--try-xfails", default=False) or \
                    config.getoption("--issue", default=None) is not None

    # cache loaded xfails for each module to avoid redundant file reads
    module_xfails = {}

    for item in items:

        module, test_name = item.nodeid.split("::", 1)

        # load xfails for the module if not already loaded
        xfails = module_xfails.get(module, None)
        if xfails is None:
            xfails = load_xfails(module)
            module_xfails[module] = xfails

        xfail_entry = xfails.get(test_name, None)

        if not xfail_entry:
            continue

        issue = xfail_entry.issue
        reason = f"known failure, see issue {issue}" if issue else "known failure"
        item.add_marker(pytest.mark.xfail(reason=reason, strict=not run_xfails, run=run_xfails))

        if issue:
            item.add_marker(pytest.mark.issue(issue))
            item.user_properties.append(("issue", issue))

