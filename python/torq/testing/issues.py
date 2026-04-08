from collections import defaultdict
from collections.abc import Mapping
import os
import re
import subprocess
from typing import Any, AnyStr, Dict, Optional, Pattern
from urllib.parse import urlparse

import pytest
import requests
import xdist


_failures: Dict[str, Dict[str, Any]] = defaultdict(dict)
_reports: Dict[str, Dict[str, Any]] = defaultdict(dict)

_TASK_NODEID_RE = re.compile(r"^- \[([ xX])\]\s*(?:~~)?`([^`]+)`(?:\s*\(([^)]+)\))?")
_MODULE_TITLE_COUNTS_RE = re.compile(r" \((?:pass \d+ / )fail \d+ / error \d+\)$")
_ERROR_TITLE_COUNTS_RE = re.compile(r" \(\d+ tests\)$")
_ISSUE_TYPE = "Bug"

# Things that look like an error
# re.MULTILINE makes $ match end of line.
_ERROR_PATTERN = re.compile(r'(Assertion .*$|error: .*$)', re.MULTILINE)

def pytest_addoption(parser):

    parser.addoption(
        "--update-github-issue",
        default=None,
        help="Update sub-issues of a GitHub issue at the end of the test session when tests fail",
    )


def _normalize_error(report):
    stderr =  getattr(report, "capstderr", None)

    if stderr is None:
        return None

    error_match = _ERROR_PATTERN.search(stderr)
    if error_match is None:
        return None

    # Normalize the error text:
    # - remove the suffix tile and fuse adds
    error = error_match.group(0).removesuffix(
        " (encountered while running the pipeline to checking if a tile fits in memory)")
    # - change fixed numbers to <N>
    error = re.sub(r"[0-9]+", "<N>", error)

    return error

def pytest_runtest_logreport(report: pytest.TestReport):
    test_report = _reports[report.nodeid]

    test_report[report.when] = report.outcome
    test_report["wasxfail"] = bool(getattr(report, "wasxfail", False))

    # Track unexpected failures from setup/call/teardown phases.
    if report.outcome != "failed":
        return

    if test_report["wasxfail"]:
        return

    _failures[report.nodeid] = {
        "phase": report.when,
        "error": _normalize_error(report)
    }


def _classify_test_result(phases: Dict[str, Any]) -> str | None:
    if phases.get("wasxfail"):
        return None

    setup_outcome = phases.get("setup")
    call_outcome = phases.get("call")
    teardown_outcome = phases.get("teardown")

    if setup_outcome == "failed" or teardown_outcome == "failed":
        return "error"
    if call_outcome == "failed":
        return "fail"
    if call_outcome == "passed":
        return "success"
    return None


def _module_summary(module: str) -> Dict[str, int]:
    counts = {"fail": 0, "error": 0, "success": 0}
    module_prefix = f"{module}::"

    for nodeid, phases in _reports.items():
        if not (nodeid == module or nodeid.startswith(module_prefix)):
            continue

        result = _classify_test_result(phases)
        if result is not None:
            counts[result] += 1

    return counts


def _extract_previous_task_nodeids(body: str) -> list[tuple[str, bool, Optional[str]]]:
    previous: set[str] = set()
    for line in body.splitlines():
        match = _TASK_NODEID_RE.match(line.strip())
        if match:
            previous.add((match.group(2), match.group(1) != " ", match.group(3)))
    return previous


def _build_module_issue_title(base_title: str, summary: Dict[str, int]) -> str:
    return (
        f"{base_title} "
        # f"(pass {summary['success']} / fail {summary['fail']} / error {summary['error']})"
        f"(fail {summary['fail']} / error {summary['error']})"
    )


def _title_matches_module_issue(issue_title: str, base_title: str, title_suffix_re: re.Pattern[AnyStr]) -> bool:
    if issue_title == base_title:
        return True
    if not issue_title.startswith(base_title):
        return False
    return bool(title_suffix_re.fullmatch(issue_title, len(base_title)))


def _build_module_issue(
    module: str,
    base_title: str,
    failures: Dict[str, Dict[str, Any]],
    previous_body: str | None = None,
) -> str:

    summary = _module_summary(module)

    previous_failures = sorted(_extract_previous_task_nodeids(previous_body or ""))
    current_failures = set(failures.keys())

    tests_list = []

    # Preserve history by marking done tests that were previously failing
    # but are not currently failing anymore.
    for nodeid, passed, phase in previous_failures:
        if nodeid in current_failures:
            continue

        if nodeid not in _reports and not passed:
            if phase in ["setup", "teardown"]:
                summary["error"] += 1
            else:
                # phase == "call"
                summary["fail"] += 1

            failures[nodeid] = { "phase": phase }
            continue

        if nodeid not in _reports:
            summary["success"] += 1

        tests_list.append(f"- [x] `{nodeid}`")

    # NB: at this point `failures` include tests that failed in the last run, and
    # tests that failed in the runs before and did not run now.

    for nodeid in sorted(failures.keys()):
        phase = failures[nodeid]["phase"]
        tests_list.append(f"- [ ] `{nodeid}` ({phase})")

    # TODO: dump failures.keys() into the module's xfail.py file.

    lines = [
        "⚠️ Automatically generated, do not edit by hand or it will be overwritten.",
        "",
        "### Summary",
        f"- 🔴 Fail: **{summary['fail']}**",
        f"- 🟠 Error: **{summary['error']}**",
        # f"- 🟢 Success: **{summary['success']}**",
        "",
        "### Failing tests:",
    ] + tests_list

    body = "\n".join(lines)

    title = _build_module_issue_title(base_title, summary)

    return (title, body)


def _build_error_issue(
    error: str,
    base_title,
    failures: list[str],
    previous_body: Optional[str] = None
) -> str:

    previous_failures = sorted(_extract_previous_task_nodeids(previous_body)) if previous_body else []
    current_failures = set(failures)

    tests_list = []
    # Preserve history by marking done tests that were previously failing
    # but are not currently failing anymore.
    for nodeid, passed, phase in previous_failures:
        if nodeid in current_failures:
            continue

        if nodeid not in _reports and not passed:
            failures.append(nodeid)
            continue

        tests_list.append(f"- [x] `{nodeid}`")

    tests_list += [f"- [ ] `{nodeid}`" for nodeid in sorted(failures)]

    lines = [
        "⚠️ Automatically generated, do not edit by hand or it will be overwritten.",
        "",
        "### Summary",
        f"{len(failures)} tests failed with the message:",
        "```",
        error,
        "```",
        "",
        "### Failing tests:",
    ] + tests_list

    body = "\n".join(lines)
    title = f"{base_title} ({len(failures)} tests)"

    return (title, body)


def _find_existing_issue(repo: str, base_title: str, title_suffix_re: Pattern[AnyStr], parent_issue: str | None, headers: Dict[str, str]) -> Dict[str, Any] | None:
    """Find an open issue with the same title among parent sub-issues only."""

    if not parent_issue:
        return None

    url = f"https://api.github.com/repos/{repo}/issues/{parent_issue}/sub_issues"
    params = {"per_page": 100}

    while True:
        response = requests.get(url, headers=headers, params=params, timeout=20)
        if response.status_code != 200:
            raise RuntimeError(
                "Failed to list parent sub-issues for dedupe: "
                f"HTTP {response.status_code} {response.text}"
            )

        issues = response.json()
        for issue in issues:
            if _title_matches_module_issue(issue.get("title", ""), base_title, title_suffix_re):
                return issue

        next_url = None
        links = response.links or {}
        if "next" in links:
            next_url = links["next"].get("url")
        if not next_url:
            break
        url = next_url
        params = None

    return None

def _find_repo():
    """
    Find the owner/name of the repo where this script resides.
    """

    if os.environ.get("GITHUB_REPOSITORY", "") != "":
        print("Using repository from environment variable GITHUB_REPOSITORY")
        return os.environ.get("GITHUB_REPOSITORY")

    remote_url = subprocess.check_output(
        ["git", "config", "--get", "remote.origin.url"], text=True
    ).strip()

    if remote_url.startswith("git@"):
        host, sep, path = remote_url.partition(":")
        if host != "git@github.com" or not sep:
            raise RuntimeError(f"Unsupported git remote (expected GitHub): {remote_url}")
    elif "://" in remote_url:

        parsed = urlparse(remote_url)
        if parsed.hostname != "github.com":
            raise RuntimeError(f"Unsupported git remote host: {parsed.hostname}")
        path = parsed.path.lstrip("/")
    else:
        raise RuntimeError(f"Unrecognized git remote format: {remote_url}")

    path = path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]

    parts = path.split("/")
    if len(parts) != 2 or not all(parts):
        raise RuntimeError(f"Could not parse GitHub owner/repo from remote: {remote_url}")

    return f"{parts[0]}/{parts[1]}"


def _group_failures_by_module(failures: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped = defaultdict(dict)
    for nodeid, info in failures.items():
        module = nodeid.split("::")[0]
        grouped[module][nodeid] = info
    return grouped


def _get_token():
    """
    Get a GitHub token with permissions to read releases from the given repo. This will first check for a token in the environment variable GITHUB_TOKEN, and if not found, will attempt to use the gh CLI to get a token.
    """

    if os.environ.get("GITHUB_TOKEN", "") != "":
        return os.environ.get("GITHUB_TOKEN")

    # not found, try to get a token from the gh CLI
    try:
        token = subprocess.check_output(
            ["gh", "auth", "token", "--hostname", "github.com"], text=True
        ).strip()

    except Exception as e:
        raise RuntimeError(
            f"Failed to get GitHub token from environment or gh CLI: {e}"
        )

    return token

def _check_issue_exists(issue_number: str, repo: str, headers: Mapping[str, str]):
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"

    response = requests.get(url, headers=headers, timeout=20)
    if response.status_code == 404:
        raise RuntimeError(f"Parent issue #{issue_number} not found in repository {repo}")
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to check parent issue existence: HTTP {response.status_code} {response.text}"
        )


def _add_sub_issue(parent_issue: str, sub_issue_id: int, repo: str, headers: Mapping[str, str]):
    url = f"https://api.github.com/repos/{repo}/issues/{parent_issue}/sub_issues"

    response = requests.post(
        url,
        headers=headers,
        json={"sub_issue_id": sub_issue_id},
        timeout=20,
    )
    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"Failed to attach sub-issue #{sub_issue_id} to parent #{parent_issue}: "
            f"HTTP {response.status_code} {response.text}"
        )

def _update_github_issue(repo, headers, existing_issue, title, body):
    url = f"https://api.github.com/repos/{repo}/issues/{existing_issue['number']}"

    update_payload = {
        "title": title,
        "body": body,
        "type": _ISSUE_TYPE,
    }

    response = requests.patch(url, headers=headers, json=update_payload, timeout=20)

    if response.status_code == 200:
        issue_url = response.json().get("html_url", "")
        print(f"[torq.testing.issues] Updated issue: {issue_url}")
        return

    print(
        "[torq.testing.issues] Failed to update issue: "
        f"HTTP {response.status_code} {response.text}"
    )


def _new_github_issue(repo, headers, title, body):
    url = f"https://api.github.com/repos/{repo}/issues"

    payload = {
        "title": title,
        "body": body,
        "labels": ["compiler"],
        "type": _ISSUE_TYPE,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=20)
    if response.status_code not in (200, 201):
        print(
            "[torq.testing.issues] Failed to create issue: "
            f"HTTP {response.status_code} {response.text}"
        )
        return None

    created_issue = response.json()
    issue_url = created_issue.get("html_url", "")
    created_issue_id = created_issue.get("id")
    if created_issue_id is None:
        print("[torq.testing.issues] Created issue but could not read issue id for linking")
        return None

    print(f"[torq.testing.issues] Created issue: {issue_url}")
    return int(created_issue_id)



def _update_issues_by_module(parent_issue, repo, headers):
    for module, module_failures in _group_failures_by_module(_failures).items():

        base_title = f"Test failures on module {module}"

        try:
            existing_issue = _find_existing_issue(repo, base_title, _MODULE_TITLE_COUNTS_RE, parent_issue, headers)

            title, body = _build_module_issue(
                module,
                base_title,
                module_failures,
                previous_body=existing_issue.get("body", "") if existing_issue else None,
            )

            if existing_issue is not None:
                _update_github_issue(repo, headers, existing_issue, title, body)
                continue

            created_issue_id = _new_github_issue(repo, headers, title, body)
            if created_issue_id is None:
                continue

            _add_sub_issue(parent_issue, created_issue_id, repo, headers)

        except requests.RequestException as exc:
            print(f"[torq.testing.issues] Failed to create issue: {exc}")
        except RuntimeError as exc:
            print(f"[torq.testing.issues] Failed to check existing issues: {exc}")


def _group_failures_by_error(failures: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    grouped = defaultdict(list)
    for nodeid, info in failures.items():
        error = info.get("error", None)
        if error is not None:
            grouped[error].append(nodeid)
    return grouped


def _update_issues_by_error(parent_issue, repo, headers):
    for error, failures in _group_failures_by_error(_failures).items():
        base_title = f"Test failures with error '{error}'"

        try:
            existing_issue = _find_existing_issue(repo, base_title, _ERROR_TITLE_COUNTS_RE, parent_issue, headers)

            title, body = _build_error_issue(
                error,
                base_title,
                failures=failures,
                previous_body= existing_issue.get("body", "") if existing_issue else None,
            )
            if existing_issue is not None:
                _update_github_issue(repo, headers, existing_issue, title, body)
                continue

            created_issue_id = _new_github_issue(repo, headers, title, body)
            if created_issue_id is None:
                continue

            _add_sub_issue(parent_issue, created_issue_id, repo, headers)

        except requests.RequestException as exc:
            print(f"[torq.testing.issues] Failed to create issue: {exc}")
        except RuntimeError as exc:
            print(f"[torq.testing.issues] Failed to check existing issues: {exc}")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    if xdist.is_xdist_worker(session):
        return

    config = session.config

    if not config.getoption("--update-github-issue"):
        return

    if not _failures:
        return

    token = _get_token()

    if not token:
        print("[torq.testing.issues] Skipping issue creation: missing GITHUB_TOKEN or GH_TOKEN")
        return

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2026-03-10",
    }

    try:
        repo = _find_repo()
    except:
        print("[torq.testing.issues] Skipping issue creation: could not determine repository")
        return

    parent_issue = config.getoption("--update-github-issue")

    try:
        _check_issue_exists(parent_issue, repo, headers)
    except RuntimeError as exc:
        print(f"[torq.testing.issues] Skipping issue creation: {exc}")
        return

    _update_issues_by_module(parent_issue, repo, headers)
    _update_issues_by_error(parent_issue, repo, headers)
