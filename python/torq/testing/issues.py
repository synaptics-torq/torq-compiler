import os
import re
import subprocess
from typing import Dict, Any
from urllib.parse import urlparse

import pytest
import requests
import xdist


_failures: Dict[str, Dict[str, Any]] = {}
_reports: Dict[str, Dict[str, Any]] = {}
_TASK_NODEID_RE = re.compile(r"^- \[[ xX]\]\s*(?:~~)?`([^`]+)`")
_TITLE_COUNTS_RE = re.compile(r" \(pass \d+ / fail \d+ / error \d+\)$")
_ISSUE_TYPE = "Bug"


def pytest_addoption(parser):    
    
    parser.addoption(
        "--update-github-issue",        
        default=None,
        help="Update sub-issues of a GitHub issue at the end of the test session when tests fail",
    )


def pytest_runtest_logreport(report: pytest.TestReport):
    test_report = _reports.setdefault(report.nodeid, {})
    test_report[report.when] = report.outcome
    if getattr(report, "wasxfail", False):
        test_report["wasxfail"] = True

    # Track unexpected failures from setup/call/teardown phases.
    if report.outcome != "failed":
        return
    if getattr(report, "wasxfail", False):
        return

    _failures.setdefault(
        report.nodeid,
        {            
            "nodeid": report.nodeid,
            "phase": report.when
        },
    )


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


def _extract_previous_task_nodeids(body: str) -> set[str]:
    previous: set[str] = set()
    for line in body.splitlines():
        match = _TASK_NODEID_RE.match(line.strip())
        if match:
            previous.add(match.group(1))
    return previous


def _build_issue_title(base_title: str, summary: Dict[str, int]) -> str:
    return (
        f"{base_title} "
        f"(pass {summary['success']} / fail {summary['fail']} / error {summary['error']})"
    )


def _title_matches_module_issue(issue_title: str, base_title: str) -> bool:
    if issue_title == base_title:
        return True
    if not issue_title.startswith(base_title):
        return False
    suffix = issue_title[len(base_title):]
    return bool(_TITLE_COUNTS_RE.fullmatch(suffix))


def _build_issue_body(
    failures: Dict[str, Dict[str, Any]],
    previous_body: str | None = None,
    summary: Dict[str, int] | None = None,
) -> str:

    lines = []
    summary = summary or {"fail": 0, "error": 0, "success": 0}

    lines.extend([
        "⚠️ Automatically generated, do not edit by hand or it will be overwritten.",
        ""        
        "### Summary",
        f"- 🔴 Fail: **{summary['fail']}**",
        f"- 🟠 Error: **{summary['error']}**",
        f"- 🟢 Success: **{summary['success']}**",
    ])
        
    lines.extend([
        "",
        "### Failing tests:",
    ])

    previous_failures = _extract_previous_task_nodeids(previous_body or "")
    current_failures = set(failures.keys())

    # Preserve history by marking done tests that were previously failing
    # but are not currently failing anymore.
    for nodeid in sorted(previous_failures - current_failures):
        lines.append(f"- [x] `{nodeid}`")

    for nodeid in sorted(failures):
        phase = failures[nodeid]["phase"]
        lines.append(f"- [ ] `{nodeid}` ({phase})")

    return "\n".join(lines)


def _find_existing_issue(repo: str, base_title: str, parent_issue: str | None, headers: Dict[str, str]) -> Dict[str, Any] | None:
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
            if _title_matches_module_issue(issue.get("title", ""), base_title):
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
    grouped = {}
    for nodeid, info in failures.items():
        module = nodeid.split("::")[0]
        grouped.setdefault(module, {})[nodeid] = info
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

def _check_issue_exists(issue_number: str, repo: str, token: str):
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2026-03-10",
    }
    response = requests.get(url, headers=headers, timeout=20)
    if response.status_code == 404:
        raise RuntimeError(f"Parent issue #{issue_number} not found in repository {repo}")
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to check parent issue existence: HTTP {response.status_code} {response.text}"
        )


def _add_sub_issue(parent_issue: str, sub_issue_id: int, repo: str, token: str):
    url = f"https://api.github.com/repos/{repo}/issues/{parent_issue}/sub_issues"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2026-03-10",
    }
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
    
    try:
        repo = _find_repo()
    except:
        print("[torq.testing.issues] Skipping issue creation: could not determine repository")
        return
    
    parent_issue = config.getoption("--update-github-issue")

    try:
        _check_issue_exists(parent_issue, repo, token)
    except RuntimeError as exc:
        print(f"[torq.testing.issues] Skipping issue creation: {exc}")
        return

    failures_by_module = _group_failures_by_module(_failures)

    for module, module_failures in failures_by_module.items():

        base_title = f"Test failures on module {module}"
        summary = _module_summary(module)
        title = _build_issue_title(base_title, summary)
        body = ""
        payload = {}
        
        url = f"https://api.github.com/repos/{repo}/issues"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2026-03-10",
        }

        try:
            existing_issue = _find_existing_issue(repo, base_title, parent_issue, headers)
            if existing_issue is not None:
                body = _build_issue_body(
                    module_failures,
                    previous_body=existing_issue.get("body") or "",
                    summary=summary,
                )
                update_url = f"https://api.github.com/repos/{repo}/issues/{existing_issue['number']}"
                update_payload = {
                    "title": title,
                    "body": body,
                    "type": _ISSUE_TYPE,
                }
                response = requests.patch(update_url, headers=headers, json=update_payload, timeout=20)
                if response.status_code == 200:
                    issue_url = response.json().get("html_url", "")
                    print(f"[torq.testing.issues] Updated issue: {issue_url}")
                    continue
                print(
                    "[torq.testing.issues] Failed to update issue: "
                    f"HTTP {response.status_code} {response.text}"
                )
                continue

            body = _build_issue_body(module_failures, summary=summary)
            payload = {
                "title": title,
                "body": body,
                "labels": ["compiler"],
                "type": _ISSUE_TYPE,
            }
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            if response.status_code in (200, 201):
                created_issue = response.json()
                issue_url = created_issue.get("html_url", "")
                created_issue_id = created_issue.get("id")
                if created_issue_id is None:
                    print("[torq.testing.issues] Created issue but could not read issue id for linking")
                    continue

                _add_sub_issue(parent_issue, int(created_issue_id), repo, token)
                print(f"[torq.testing.issues] Created issue: {issue_url}")
                continue
            print(
                "[torq.testing.issues] Failed to create issue: "
                f"HTTP {response.status_code} {response.text}"
            )
        except requests.RequestException as exc:
            print(f"[torq.testing.issues] Failed to create issue: {exc}")
        except RuntimeError as exc:
            print(f"[torq.testing.issues] Failed to check existing issues: {exc}")
