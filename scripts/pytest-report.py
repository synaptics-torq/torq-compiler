"""
A small script to quickly sort out the results of pytest.

Prerequisites:
`pip install pytest-json-report`

Generate a JSON report (the input to the script):
`pytest -m ci -n auto --json-report ...`

Print errors and tests:
- print errors summary: `python scripts/pytest-report.py --errors`
- print list of tests sorted by errors: `python scripts/pytest-report.py --sorted`
"""

import argparse
from collections import defaultdict
import json
import os
import re


def load_tests(report_file):
    try:
        with open(report_file, 'r') as f:
            data = json.load(f)
            tests = data.get("tests", [])
            # Filter tests with outcome "error" and having setup.stderr
            return [t for t in tests if t.get("outcome") == "error"
                    and "setup" in t and "stderr" in t["setup"]]

    except (FileNotFoundError, json.JSONDecodeError):
        return []


def sort_tests(tests):
    # These are the things that look like errors
    # re.MULTILINE makes $ match end of line.
    pattern = re.compile(r'(Assertion .*$|error: .*$)', re.MULTILINE)

    sorted_tests = defaultdict(list)
    unknown_tests = []

    for test in tests:
        nodeid = test["nodeid"]

        error_match = pattern.search(test["setup"]["stderr"])
        if error_match is None:
            unknown_tests.append(test)
            continue

        # Normalize the error text:
        # - remove the suffix tile and fuse adds
        error = error_match.group(0).removesuffix(
            " (encountered while running the pipeline to checking if a tile fits in memory)")
        # - change fixed numbers to N
        error = re.sub(r"[0-9]+", "<N>", error)

        sorted_tests[error].append(nodeid)

    return (sorted_tests, unknown_tests)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
                                     # "Process pytest JSON report (run `pytest --json-report ...` to generate the report).")
    parser.add_argument("--report-file", default=os.environ.get("REPORTFILE", ".report.json"),
                        help="Path to the pytest JSON report file (default: $REPORTFILE or .report.json).")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sorted", action="store_true", help="print the tests sorted by error")
    group.add_argument("--errors", action="store_true", help="print a summary of the errors")
    group.add_argument("--stderr", type=int, nargs='?', const=0, metavar='N',
                       help="print the stderr of the Nth (starting from 0) unsorted test")

    args = parser.parse_args()

    tests = load_tests(args.report_file)

    errors, unknown_tests = sort_tests(tests)
    sorted_errors =  sorted(errors.items(), key=lambda item : len(item[1]), reverse=True)

    # Print the tests sorted by error
    if args.sorted:
        for error, tests in sorted_errors:
            print(f"ERROR ({len(tests)} tests): {error}")
            for test in sorted(tests):
                print(test)
            print()

    # Print a summary of the errors
    if args.errors:
        width = len(str(len(sorted_errors[0][1]))) if sorted_errors else 1
        for error, tests in sorted_errors:
            print(f"{len(tests):{width}} tests: {error}")
        if unknown_tests:
            print(f"*** {len(unknown_tests)} tests with unknown error ***")
        return

    # Print the stderr of an unknown test
    if args.stderr is not None:
        n = args.stderr
        if n < len(unknown_tests):
            print(unknown_tests[n]["setup"]["stderr"])
        return


if __name__ == "__main__":
    main()
