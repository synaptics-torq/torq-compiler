from urllib.parse import urlencode
import requests
import os
import pytest
import xdist
import uuid
import time
import json
import zipfile
from tempfile import TemporaryDirectory
import shutil
from typing import Dict, Any

import random


reports = {}

def _upload_zip_bundle(zip_path: str) -> Dict[str, Any]:
    """Uploads a ZIP bundle to the bulk endpoint.

    Returns the JSON response (should include the created test session).
    """
    
    server = os.environ.get("TORQ_PERF_SERVER", "")

    if server == "":
        return None
    
    token = os.environ.get("TORQ_PERF_SERVER_TOKEN", "")

    headers = {}

    if token != "":
        headers['Authorization'] = f'Bearer {token}'

    # retry up to 3 times
    for retry_id in range(3):
        if retry_id > 0:
            wait_time = (2 ** retry_id) * random.uniform(0.5, 1.5)
            time.sleep(wait_time)

        with open(zip_path, 'rb') as f:
            files = {'file': (os.path.basename(zip_path), f, 'application/zip')}
            response = requests.post(f"{server}/api/test-sessions/upload_zip/", files=files, headers=headers)

        if response.status_code in [200, 201]:
            return f"/test-sessions/{response.json()['id']}/"

    raise RuntimeError(f"Failed to upload zip bundle: {response.status_code} - {response.text}")


def _build_workflow_url() -> str:
    # inside github actions
    if "GITHUB_RUN_ID" in os.environ:
        github_server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        github_repo = os.environ.get("GITHUB_REPOSITORY")
        github_run_id = os.environ.get("GITHUB_RUN_ID")
        github_run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
        return f"{github_server}/{github_repo}/actions/runs/{github_run_id}/attempts/{github_run_attempt}"
    # local run: use session uuid
    return uuid.uuid4().hex


def _parse_from_nodeid(nodeid: str) -> tuple[str, str, str]:
    module = nodeid
    name = ''
    parameters = ''

    if '::' in nodeid:
        parts = nodeid.split('::', 1)
        module = parts[0]
        name = parts[1]
    else:
        module = nodeid
        name = ''

    if '[' in name and name.endswith(']'):
        base, param = name.split('[', 1)
        name = base
        parameters = param[:-1]

    return module, name, parameters


def pytest_runtest_logreport(report: pytest.TestReport):
    """Called after each test phase (setup, call, teardown)"""    

    reports.setdefault(report.nodeid, {})[report.when] = report


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session):
    
    if xdist.is_xdist_worker(session):
        return

    server_url = os.environ.get("TORQ_PERF_SERVER", "")

    if server_url == "":
        return

    with TemporaryDirectory() as temp_dir:
    
        print("\nCreating results bundle...")

        manifest = {
            'git_commit': os.environ.get("GITHUB_SHA"),
            'git_branch': os.environ.get("GITHUB_REF"),
            'workflow_url': _build_workflow_url(),
            'test_runs': [],
            'batch_name': os.environ.get("TORQ_PERF_BATCH_NAME", "default"),
        }

        profiles_root = os.path.join(temp_dir, 'profiles')
        os.makedirs(profiles_root, exist_ok=True)

        for node_id, report_phases in reports.items():

            # parse the node id of the report            
            module, name, parameters = _parse_from_nodeid(node_id)

            # find the complete outcome of the test across all phases
            outcome = 'failed'            
            for phase in ['setup', 'call', 'teardown']:              
                if phase in report_phases:                    
                    outcome = report_phases[phase].outcome
                    if outcome != 'passed':
                        break

            test_run = {
                'module': module,
                'name': name,
                'parameters': parameters,
                'outcome': outcome,
            }

            if 'call' in report_phases:

                report = report_phases['call']

                profiling_output_file = dict(report.user_properties).get('profiling_output')

                if profiling_output_file:

                    # Copy profiling file into profiles/ and reference it relative to manifest
                    dst_name = os.path.basename(profiling_output_file)
                    dst_path = os.path.join(profiles_root, dst_name)
                    
                    shutil.copy2(profiling_output_file, dst_path)
                    test_run['profiling_file'] = os.path.join('profiles', dst_name)
                
            manifest['test_runs'].append(test_run)

        manifest_path = os.path.join(temp_dir, 'test_session.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f)

        # Create zip
        zip_path = os.path.join(temp_dir, 'bundle.zip')
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, arcname='test_session.json')
            if os.path.isdir(profiles_root):
                for fname in os.listdir(profiles_root):
                    fpath = os.path.join(profiles_root, fname)
                    if os.path.isfile(fpath):
                        zf.write(fpath, arcname=os.path.join('profiles', fname))

        # Upload bundle        
        print("Uploading results bundle...")
        session_path = _upload_zip_bundle(zip_path) + "#batch-" + manifest['batch_name']
        print("Upload complete.")

        space_url = os.environ.get("TORQ_PERF_SPACE_URL", "")

        if space_url == "":
            space_url = server_url + session_path
        else:
            space_url = space_url + "?" + urlencode({"next": session_path})

        print(f"\nTest results available at {space_url}\n")

        if os.getenv("GITHUB_ACTIONS"):
            try:
                github_step_summary = os.getenv("GITHUB_STEP_SUMMARY")
                if github_step_summary:
                    with open(github_step_summary, "a") as f:
                        f.write(f"[View test results]({space_url})\n\n")
            except Exception as e:
                print(f"Failed to write report url to GitHub Actions summary: {e}")
