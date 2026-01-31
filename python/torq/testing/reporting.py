from urllib.parse import urlencode
import requests
import os
import pytest
import xdist
import uuid
import time
import json
import zipfile
import csv
import re
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
from typing import Dict, Any, List, Optional

import random

logger = logging.getLogger("torq.testing.reporting")


reports = {}
profiling_results: List[Dict[str, Any]] = []
_pytest_config = None  # Store config globally for access in hooks


def pytest_configure(config):
    """Store config globally for access in other hooks."""
    global _pytest_config
    _pytest_config = config


def pytest_addoption(parser):
    """Add command-line options for profiling log configuration."""
    group = parser.getgroup("profiling", "Profiling result aggregation and plotting")
    
    group.addoption(
        "--template-profiling-enabled",
        action="store_true",
        default=False,
        help="Enable profiling result aggregation and plotting"
    )
    
    group.addoption(
        "--template-profiling-csv-filename",
        action="store",
        default="template_mlir_profiling_summary.csv",
        help="Filename for aggregated profiling CSV"
    )
    
    group.addoption(
        "--profiling-param-pattern",
        action="store",
        default="shape_info-variant-template",
        help="Parameter extraction pattern from node ID, e.g. 'shape_info-variant-template' or 'template-shape_info-variant'"
    )
    
    group.addoption(
        "--profiling-x-axis",
        action="store",
        default="shape",
        help="Parameter to plot on X-axis (default: shape)"
    )
    
    group.addoption(
        "--profiling-y-axis",
        action="store",
        default="variant",
        help="Parameter to group by (bars/colors) on plot (default: variant)"
    )


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


def _extract_params_from_nodeid(nodeid: str, pattern: str = "shape_info-variant-template") -> Dict[str, str]:
    """Extract parametrized values from pytest node ID based on pattern.
    
    Examples:
        pattern="shape_info-variant-template"
        test_run_templates_on_soc[r4_bf16_1x1x292x292-nss-add-1x24x56x56-bf16.mlir-astra_machina-default]
        -> {"rank": "4", "dtype": "bf16", "shape": "1x1x292x292",
            "variant": "nss", "template": "add-1x24x56x56-bf16.mlir"}
    
    Args:
        nodeid: Pytest test node identifier
        pattern: Dash-separated pattern defining parameter names
        
    Returns:
        Dictionary of extracted parameters
    """
    params = {}
    
    # Extract bracketed parameters: [param1-param2-param3]
    match = re.search(r'\[(.*?)\]', nodeid)
    if not match:
        return params
    
    param_str = match.group(1)
    parts = param_str.split('-')
    
    # Parse pattern to understand structure
    pattern_parts = pattern.split('-')
    
    if len(parts) < len(pattern_parts):
        return params
    
    # Map pattern parts to actual parameter values
    for i, pattern_name in enumerate(pattern_parts):
        if i < len(parts):
            # Special handling for shape_info pattern (backward compatibility)
            if pattern_name == "shape_info":
                # Parse shape_info: r{rank}_{dtype}_{shape}
                shape_info = '-'.join(parts[i:len(parts)-len(pattern_parts)+i+1])
                shape_match = re.match(r'r(\d+)_([a-z0-9]+)_([\dx]+)', shape_info)
                if shape_match:
                    params["rank"] = shape_match.group(1)
                    params["dtype"] = shape_match.group(2)
                    params["shape"] = shape_match.group(3)
            else:
                # Direct mapping
                params[pattern_name] = parts[i] if i < len(parts) else ""
    
    return params


def _extract_latency_from_profile_csv(profile_csv: Path) -> Optional[float]:
    """Extract elapsed_time(us) from host profile CSV.
    
    Args:
        profile_csv: Path to host_profile.csv
        
    Returns:
        Last elapsed_time(us) value, or None if not found
    """
    if not profile_csv.exists():
        return None
    
    elapsed_us = None
    try:
        with open(profile_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = row.get("elapsed_time(us)") or row.get(" elapsed_time(us)")
                if value not in (None, ""):
                    elapsed_us = float(value)
    except (IOError, ValueError, KeyError):
        return None
    
    return elapsed_us


def pytest_runtest_logreport(report: pytest.TestReport):
    """Called after each test phase (setup, call, teardown)"""

    reports.setdefault(report.nodeid, {})[report.when] = report
    
    # Collect profiling data if enabled and test passed
    if report.when == "teardown" and report.outcome == "passed":
        if _pytest_config and _pytest_config.getoption("--template-profiling-enabled", default=False):
            _collect_profiling_data(report, _pytest_config)


def _collect_profiling_data(report: pytest.TestReport, config):
    """Extract profiling data from test report."""
    try:
        # Get host_profile_csv path from user properties
        host_profile_csv = dict(report.user_properties).get('host_profile_csv')
        if not host_profile_csv:
            logger.debug(f"No host_profile_csv in user_properties for {report.nodeid}")
            return
        
        # Convert to Path if it's a string
        if not isinstance(host_profile_csv, Path):
            host_profile_csv = Path(host_profile_csv)
        
        logger.debug(f"Looking for profile at: {host_profile_csv}")
        
        if not host_profile_csv.exists():
            logger.debug(f"Profile CSV not found: {host_profile_csv}")
            return
        
        # Extract latency
        latency_us = _extract_latency_from_profile_csv(host_profile_csv)
        if latency_us is None:
            logger.debug(f"Could not extract latency from {host_profile_csv}")
            return
        
        # Get pattern from config
        pattern = config.getoption("--profiling-param-pattern", default="shape_info-variant-template")
        
        # Extract parameters from test node ID using pattern
        params = _extract_params_from_nodeid(report.nodeid, pattern)
        
        # Get input specs from user properties if available
        input_specs = dict(report.user_properties).get('input_specs', 'unknown')
        
        # Get template name from user properties (set by test)
        template_name = dict(report.user_properties).get('template_name')
        testcase_name = template_name if template_name else report.nodeid.split('::')[-1].split('[')[0]
        
        # Store result
        result_entry = {
            "testcase": testcase_name,
            "nodeid": report.nodeid,
            "latency_us": latency_us,
            "profiling_file": str(host_profile_csv),
            "input_specs": input_specs,
            **params
        }
        
        profiling_results.append(result_entry)
        logger.debug(f"Added result entry. Total entries: {len(profiling_results)}")
        
    except Exception as e:
        # Don't fail tests due to profiling issues
        logger.warning(f"Failed to collect profiling data for {report.nodeid}: {e}")
        import traceback
        traceback.print_exc()


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session):
    
    if xdist.is_xdist_worker(session):
        return

    # Generate template profiling report (CSV + plots) if enabled (independent of performance server)
    if session.config.getoption("--template-profiling-enabled", default=False):
        _generate_template_profiling_report(session)

    # Upload to performance server if configured
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


def _generate_template_profiling_report(session):
    """Generate template MLIR profiling report: CSV summary and latency plots."""
    logger.debug(f"_generate_template_profiling_report called. profiling_results count: {len(profiling_results)}")
    
    if not profiling_results:
        logger.debug("No profiling results to generate")
        return
    
    # Determine output directory
    output_dir_opt = session.config.getoption("--torq-runtime-profiling-output-dir", default=None)
    if output_dir_opt:
        output_dir = Path(output_dir_opt)
    else:
        output_dir = Path(session.config.rootpath)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get CSV filename
    csv_filename = session.config.getoption("--template-profiling-csv-filename")
    csv_path = output_dir / csv_filename
    
    # Write CSV with all captured results
    with open(csv_path, "w", newline="") as f:
        # Determine all unique parameter keys
        all_keys = set()
        for result in profiling_results:
            all_keys.update(result.keys())
        
        # Fixed column order
        base_columns = ["id", "testcase", "variant", "shape", "inputs", "latency_us", "profiling_file"]
        extra_columns = sorted(all_keys - set(base_columns))
        columns = base_columns + extra_columns
        
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        
        for idx, result in enumerate(profiling_results):
            row = {"id": idx, "inputs": result.get("input_specs", "unknown"), **result}
            writer.writerow(row)
    
    print(f"\n✓ Profiling results written to: {csv_path}")
    print(f"  Total tests logged: {len(profiling_results)}")
    
    # Generate plots
    try:
        from torq.utils.plotting import generate_latency_plots
        
        x_axis = session.config.getoption("--profiling-x-axis", default="shape")
        y_axis = session.config.getoption("--profiling-y-axis", default="variant")
        
        generate_latency_plots(
            csv_path=csv_path, 
            output_dir=output_dir,
            x_param=x_axis,
            y_param=y_axis
        )
        print(f"✓ Profiling plots generated in: {output_dir} (X={x_axis}, Y={y_axis})")
    except Exception as e:
        logger.warning(f"Failed to generate profiling plots: {e}")
