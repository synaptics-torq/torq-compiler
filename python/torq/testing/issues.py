from collections import defaultdict
from collections.abc import Mapping
import io
import os
from pathlib import Path
import re
import subprocess
from typing import Any, AnyStr, Dict, Optional, Pattern
from urllib.parse import urlparse
import zipfile

import argparse
import pytest
import requests
import xdist
import json

import torq.testing.xfail

_failures: Dict[str, Dict[str, Any]] = defaultdict(dict)
_reports: Dict[str, Dict[str, Any]] = defaultdict(dict)

_TEST_ERROR_BASE_TITLE = "Test failures with error '{error}'"
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

    parser.addoption(
        "--save-test-status-report-to",
        default=None,
        help="Save a report of with all test status at the end of the test session",
    )


def pytest_configure(config: pytest.Config):
    report_path = config.getoption("--save-test-status-report-to", default=None)
    capture_mode = config.getoption("capture", default="fd")

    if report_path and capture_mode == "no":
        raise pytest.UsageError(
            "--save-test-status-report-to requires output capture; do not use -s/--capture=no. "
            "Use --capture=tee-sys if you need live output and captured logs."
        )


def _normalize_error(report: pytest.TestReport) -> str | None:
    
    # try to get the error from the stderr
    while True:

        stderr =  getattr(report, "capstderr", "")

        if stderr is None:
            break

        stderr = stderr.replace("error: Failed call to cuInit", "")

        error_match = _ERROR_PATTERN.search(stderr)
        if error_match is None:
            break

        # Normalize the error text:
        # - remove the suffix tile and fuse adds
        error = error_match.group(0).removesuffix(
            " (encountered while running the pipeline to checking if a tile fits in memory)")
        # - change fixed numbers to <N>
        error = re.sub(r"[0-9]+", "<N>", error)

        return error
        
    # if this is another type of failure try to return the exception message
    longrepr = getattr(report, "longrepr", None)

    if longrepr is None:
        return "generic error"
    
    reprcrash = getattr(longrepr, "reprcrash", None)
    if reprcrash is not None and getattr(reprcrash, "message", None):

        msg = str(reprcrash.message)

        if msg.startswith("subprocess.CalledProcessError:"):
            # parse crashes
            pattern = r"subprocess.CalledProcessError: Command '\['([^']*)'.*\]' died with (.*)"
            match = re.search(pattern, msg)
            if match:
                command = os.path.basename(match.group(1))
                crash = match.group(2)
                return f"Command '{command}' died with {crash}"
            
            # parse exit values
            pattern = r"subprocess.CalledProcessError: Command '\['([^']*)'.*\]' returned (.*)"
            match = re.search(pattern, msg)
            if match:
                command = os.path.basename(match.group(1))
                exit_code = match.group(2)
                return f"Command '{command}' returned exit code {exit_code}"
            
        elif msg.startswith("AssertionError: Number of differences"):
            return "AssertionError: Number of differences too high"

        return str(reprcrash.message)

    if isinstance(longrepr, str):
        return longrepr    

    elif isinstance(longrepr, tuple) and len(longrepr) == 3:
        # longrepr can be a tuple of (file, line, message)
        return longrepr[2]
    
    else:
        return "generic error"



def pytest_runtest_logreport(report: pytest.TestReport):

    # Store the test reports so that we know which tests where run (and where successful)
    test_report = _reports[report.nodeid]

    test_report[report.when] = report.outcome

    # collect the chip_config so that we can know if a given chip is from extra or not
    if "chip_config" not in test_report:
        test_report["chip_config"] = None

        for key, value in report.user_properties:
            if key == "chip_config":
                test_report["chip_config"] = value

    wasxfail = bool(getattr(report, "wasxfail", False))

    # Track logs of test that failed or that were skipped because marked as xfail
    if report.outcome == "passed":
        return
    
    if report.outcome == "skipped" and not wasxfail:
        return
            
    normalized_error = _normalize_error(report)
    
    # we update the test report only here assuming only one phase fails
    test_report["error"] = normalized_error
    test_report["wasxfail"] = wasxfail

    _failures[report.nodeid] = {
        "phase": report.when,
        "error": normalized_error
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


def _extract_test_name(test: str) -> Optional[str]:
    return test.split("::", 1)[-1]


def _write_xfail_file(module: str, failures: Dict[str, Dict[str, Any]]):
    error_tests = []
    fail_tests = []

    for nodeid, info in failures.items():
        test_name = _extract_test_name(nodeid)
        if test_name is None:
            continue
        if info["phase"] in ["setup", "teardown"]:
            error_tests.append(test_name)
        else:
            # info["phase"] == "call"
            fail_tests.append(test_name)


    xfail_file_name = os.path.splitext(module)[0] + "-xfails.txt"
    with open(xfail_file_name, "wt", encoding="utf-8") as f:
        f.write("# Automatically generated, do not edit by hand or it will be overwritten.\n\n")

        if error_tests:
            f.write("# Errors:\n")
            f.write("\n".join(error_tests) + "\n\n")

        if fail_tests:
            f.write("# Fails:\n")
            f.write("\n".join(fail_tests) + "\n")


def _build_module_issue(
    module: str,
    base_title: str,
    failures: Dict[str, Dict[str, Any]],
    previous_body: str | None = None,
) -> Optional[str]:

    summary = _module_summary(module)

    previous_failures = sorted(_extract_previous_task_nodeids(previous_body or ""))
    current_failures = set(failures.keys())

    previous_nodeids = {nodeid for nodeid, _, _ in previous_failures}
    issue_changed = any(nodeid not in previous_nodeids for nodeid in failures)

    tests_list = []

    # Preserve history by marking done tests that were previously failing
    # but are not currently failing anymore.
    for nodeid, passed, phase in previous_failures:
        if nodeid in current_failures:
            # a previous test used to pass and now it fails
            issue_changed = passed or issue_changed
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

        # `not passed` is True only if a previous test used to fail and now it passes
        issue_changed = (not passed) or issue_changed

        tests_list.append(f"- [x] `{nodeid}`")

    # NB: at this point `failures` include tests that failed in the last run, and
    # tests that failed in the runs before and did not run now.

    _write_xfail_file(module, failures)

    if not issue_changed:
        return None

    for nodeid in sorted(failures.keys()):
        phase = failures[nodeid]["phase"]
        tests_list.append(f"- [ ] `{nodeid}` ({phase})")

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


def _extract_module_name(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def _group_by_module(reports: Mapping[str, Any],
                              failures: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped = { _extract_module_name(nodeid): {} for nodeid in reports }

    for nodeid, info in failures.items():
        module = _extract_module_name(nodeid)
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
        raise Exception(f"Failed to create issue: HTTP {response.status_code} {response.text}")

    created_issue = response.json()
    issue_url = created_issue.get("html_url", "")
    created_issue_id = created_issue.get("id")
    if created_issue_id is None:        
        raise Exception("Created issue but could not read issue id for linking")

    print(f"[torq.testing.issues] Created issue: {issue_url}")
    return int(created_issue_id), f"{repo}#{created_issue['number']}"


def _update_issues_by_module(parent_issue, repo, headers):
    for module, module_failures in _group_by_module(_reports, _failures).items():

        base_title = f"Test failures on module {module}"

        try:
            existing_issue = _find_existing_issue(repo, base_title, _MODULE_TITLE_COUNTS_RE, parent_issue, headers)

            issue = _build_module_issue(
                module,
                base_title,
                module_failures,
                previous_body=existing_issue.get("body", "") if existing_issue else None,
            )

            if issue is None:
                continue

            title, body = issue

            if existing_issue is not None:
                _update_github_issue(repo, headers, existing_issue, title, body)
                continue

            created_issue_id, _ = _new_github_issue(repo, headers, title, body)
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
        base_title = _TEST_ERROR_BASE_TITLE.format(error=error)

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

            created_issue_id, _ = _new_github_issue(repo, headers, title, body)
            if created_issue_id is None:
                continue

            _add_sub_issue(parent_issue, created_issue_id, repo, headers)

        except requests.RequestException as exc:
            print(f"[torq.testing.issues] Failed to create issue: {exc}")
        except RuntimeError as exc:
            print(f"[torq.testing.issues] Failed to check existing issues: {exc}")


def _save_report(path: str):
    with open(path, "w") as f:
        json.dump(_reports, f, indent=2)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    if xdist.is_xdist_worker(session):
        return

    config = session.config

    report_path = config.getoption("--save-test-status-report-to", default=False)

    if report_path:
        _save_report(report_path)

    if not _failures:
        return

    if not config.getoption("--update-github-issue"):
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


def _fetch_status_reports(repo: str, headers: dict, run_id: str) -> Dict[str, Dict[str, str]]:

    artifacts = []
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts"
    params = {"per_page": 100}

    while True:
        response = requests.get(url, headers=headers, params=params, timeout=20)
        if response.status_code != 200:
            raise RuntimeError(
                "Failed to list workflow artifacts: "
                f"HTTP {response.status_code} {response.text}"
            )

        payload = response.json()
        artifacts.extend(payload.get("artifacts", []))

        links = response.links or {}
        next_url = links.get("next", {}).get("url")
        if not next_url:
            break

        url = next_url
        params = None

    status_artifacts = [
        artifact
        for artifact in artifacts
        if "test_status" in artifact.get("name", "") and not artifact.get("expired", False)
    ]

    if not status_artifacts:
        print(
            f"No test status artifacts found for workflow run {run_id}"
        )
        return {}

    downloaded_reports: Dict[str, Any] = {}
    for artifact in status_artifacts:
        artifact_name = artifact.get("name", "<unknown>")
        download_url = artifact.get("archive_download_url")
        if not download_url:
            print(
                f"Skipping artifact without download URL: {artifact_name}"
            )
            continue

        response = requests.get(download_url, headers=headers, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download artifact {artifact_name}: "
                f"HTTP {response.status_code} {response.text}"
            )

        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            extracted_any = False
            for member in archive.infolist():
                if member.is_dir():
                    continue

                file_name = os.path.basename(member.filename)
                if not file_name or "test_status" not in file_name or not file_name.endswith(".json"):
                    continue

                with archive.open(member) as source:
                    downloaded_reports.update(json.loads(source.read().decode("utf-8")))

                extracted_any = True
                print(
                    f"Loaded {file_name} from artifact {artifact_name}"
                )

            if not extracted_any:
                print(
                    f"  Artifact {artifact_name} did not contain a matching status report"
                )

    return downloaded_reports


def _is_success(report: Dict[str, Any]) -> bool:
    return report.get("setup") == "passed" and report.get("call") == "passed" and report.get("teardown") == "passed"

def _is_skipped(report: Dict[str, Any]) -> bool:

    # test marked wasxfail with phases marked skipped are actually failures
    if report.get("wasxfail", False):
        return False
    
    return report.get("call") == "skipped" or report.get("setup") == "skipped" or report.get("teardown") == "skipped"


_issue_cache = {}

def _get_issue_for_test(nodeid: str, repo: str, parent_issue: str, headers: str, report: Dict[str, Any], dry_run: bool) -> Optional[str]:

    error = report["error"]

    if error.startswith("AssertionError: Number of differences"):
        error = "AssertionError: Number of differences too high"

    if error in _issue_cache:
        return _issue_cache[error]

    title = _TEST_ERROR_BASE_TITLE.format(error=error)

    existing_issue = _find_existing_issue(repo, title, _ERROR_TITLE_COUNTS_RE, parent_issue, headers)

    if existing_issue is not None:
        issue_name = f"{repo}#{existing_issue['number']}"
        _issue_cache[error] = issue_name
        return issue_name

    if dry_run:
        print(f"  Would create issue for error: {error}")
        _issue_cache[error] = None
        return None

    try:
        created_issue_id, issue_name = _new_github_issue(repo, headers, title, "")
    except Exception as ex:
        print("Error while creating issue for test:", nodeid)
        print("Failed to create issue for error:", report)
        raise ex
    
    _add_sub_issue(parent_issue, created_issue_id, repo, headers)

    _issue_cache[error] = issue_name

    print(f"  Created issue #{issue_name} for error: {error}")

    return issue_name


def _update_from_run(run_id: str, parent_issue: str, dry_run: bool = False):
    
    token = _get_token()

    if not token:
        raise RuntimeError("Missing GitHub token")

    repo = _find_repo()
    
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2026-03-10",
    }

    # load test status report for a given run id
    report = _fetch_status_reports(repo, headers, run_id)

    _update(report, parent_issue, dry_run)


def _update_from_file(report_path: str, parent_issue: str, dry_run: bool = False):
    with open(report_path, "r") as f:
        report = json.load(f)

    _update(report, parent_issue, dry_run)


TOPDIR = Path(__file__).parent.parent.parent.parent

def _get_extra_chip_names():
    chips = { x.stem for x in (TOPDIR / "tests" / "testdata" / "chips").glob("*.json") }
    extra_chips = { x.stem for x in (TOPDIR / "extras" / "chips").glob("*.json") }

    extra_chips = extra_chips - chips

    return extra_chips


def _update(report: Any, parent_issue: str, dry_run: bool = False) -> Dict[str, Any]:

    token = _get_token()

    if not token:
        raise RuntimeError("Missing GitHub token")

    repo = _find_repo()
    
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2026-03-10",
    }

    # group test reports by module so that we can update xfail files module by module
    group_by_module = defaultdict(dict)

    for nodeid, phases in report.items():
        module = _extract_module_name(nodeid)
        group_by_module[module][nodeid] = phases

    total_xfail = 0
    total_new_xfail = 0
    total_removed_xfail = 0
    total_updated_issues = 0

    # for each module with test status, update the corresponding xfail file
    for module, items in group_by_module.items():
        print("Updating:", module)

        # load the existing xfail file
        xfail_file_entries = torq.testing.xfail.load_xfails(module)

        # summarize data about failing and succeeding tests in the status report
        current_success: set[str] = set()  # tests that completed successfully in the current report
        current_skip: set[str] = set() # tests that were really skipped in the current report (as opposed to being marked as xfail or failed)
        current_fail: set[str] = set() # tests that failed in the current report (including xfail and errors)

        test_to_chip = {}

        for nodeid, report in items.items():
            test_name = _extract_test_name(nodeid)
            test_to_chip[test_name] = report.get("chip_config", None)
            if _is_success(report):
                current_success.add(test_name)
            elif _is_skipped(report):
                current_skip.add(test_name)
            else:
                current_fail.add(test_name)

        # update the file entries based on the new information
        new_xfails_entries: list[torq.testing.xfail.XFailEntry] = []

        current_xfail: set[str] = set()

        removed_xfails_count = 0
        updated_issues_count = 0

        # filter out tests that are now succeeding, and keep track of all the tests that are already known to fail
        for entry in xfail_file_entries.values():
            
            # remove tests that now succeed or are permanently skipped
            if entry.test_name in current_success or entry.test_name in current_skip:
                removed_xfails_count += 1
                continue

            # keep track of all the tests that are already known to fail
            current_xfail.add(entry.test_name)

            # try to update the issue ticket if we have information about the failure
            if entry.issue is None and entry.test_name in current_fail:
                node_id = module + "::" + entry.test_name
                entry.issue = _get_issue_for_test(node_id, repo, parent_issue, headers, items[node_id], dry_run)
                updated_issues_count += 1

            new_xfails_entries.append(entry)            
                    
        # append new tests we discovered are failing
        new_xfails = sorted(current_fail - current_xfail)
        for test_name in new_xfails:
            node_id = module + "::" + test_name
            issue = _get_issue_for_test(node_id, repo, parent_issue, headers, items[node_id], dry_run)            
            new_xfails_entries.append(torq.testing.xfail.XFailEntry(test_name=test_name, issue=issue))
    
        total_xfail += len(current_fail)
        total_new_xfail += len(new_xfails)
        total_removed_xfail += removed_xfails_count
        total_updated_issues += updated_issues_count

        if len(new_xfails) > 0 or removed_xfails_count > 0 or updated_issues_count > 0:

            # write out the new xfail files
            if not(dry_run):

                extra_chip_names = _get_extra_chip_names()

                base_xfails_entries = [entry for entry in new_xfails_entries if test_to_chip.get(entry.test_name, None) not in extra_chip_names]
                extra_xfails_entries = [entry for entry in new_xfails_entries if test_to_chip.get(entry.test_name, None) in extra_chip_names]

                torq.testing.xfail.write_xfails(module, base_xfails_entries)
                torq.testing.xfail.write_xfails(module, extra_xfails_entries, extras=True)

            print(f"  added {len(new_xfails)} new xfails, removed {removed_xfails_count} xfails that are now succeeding, updated {updated_issues_count} issues links")        
        else:
            print("  no changes to xfail file")

    print(f"Summary:")
    print(f"  total xfailed tests: {total_xfail}")
    print(f"  new xfails: {total_new_xfail}")
    print(f"  removed xfails: {total_removed_xfail}")
    print(f"  updated issues: {total_updated_issues}")


def main():
    argparser = argparse.ArgumentParser(description="Manage test failure issues on GitHub")

    subparser = argparser.add_subparsers(dest="command", required=True)

    status_parser = subparser.add_parser("update", help="Download the test status report generated with --save-test-status-report-to and update xfail files and issues")

    status_parser.add_argument("--run-id", help="The run id to download the report from")
    status_parser.add_argument("--report-path", help="The path to a local test status report")
    status_parser.add_argument("--parent-issue", required=True, help="The parent issue to attach generated sub-issues to")
    status_parser.add_argument("--dry-run", action="store_true", help="Print the changes that would be made without actually making them")

    args = argparser.parse_args()

    if args.command == "update":        
        if args.run_id is None and args.report_path is None:
            print("Either --run-id or --report-path must be provided")
            return
        if args.run_id is not None and args.report_path is not None:
            print("Only one of --run-id or --report-path can be provided")
            return
        if args.report_path is not None:
            _update_from_file(args.report_path, args.parent_issue, args.dry_run)
        else:
            _update_from_run(args.run_id, args.parent_issue, args.dry_run)



if __name__ == "__main__":
    main()