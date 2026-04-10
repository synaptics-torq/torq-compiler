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

from torq.model_profiler.generate_perfetto_combined_report import extract_perfetto_summary, extract_model_name

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
    last_error = None
    for retry_id in range(3):
        if retry_id > 0:
            wait_time = (2 ** retry_id) * random.uniform(0.5, 1.5)
            time.sleep(wait_time)

        try:
            with open(zip_path, 'rb') as f:
                files = {'file': (os.path.basename(zip_path), f, 'application/zip')}
                response = requests.post(f"{server}/api/test-sessions/upload_zip/", files=files, headers=headers, timeout=60)
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"Upload attempt {retry_id + 1}/3 failed: {e}")
            continue

        if response.status_code in [200, 201]:
            return f"/test-sessions/{response.json()['id']}/"
        last_error = f"{response.status_code} - {response.text}"
        print(f"Upload attempt {retry_id + 1}/3 failed: {last_error}")

    raise RuntimeError(f"Failed to upload zip bundle after 3 attempts: {last_error}")


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
        failure_logs_root = os.path.join(temp_dir, 'failure_logs')
        os.makedirs(failure_logs_root, exist_ok=True)

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

            # Distinguish xfail from regular skip: both have outcome="skipped"
            # but xfail reports carry a wasxfail attribute.
            # Also detect nxpass (not-expected pass, i.e. xfail test that unexpectedly passes):
            # - strict=False: outcome="passed" with wasxfail set
            # - strict=True: outcome="failed" with wasxfail set
            if outcome == 'skipped':
                for phase_report in report_phases.values():
                    if getattr(phase_report, 'wasxfail', None):
                        outcome = 'xfail'
                        break
            elif outcome == 'passed':
                for phase_report in report_phases.values():
                    if getattr(phase_report, 'wasxfail', None):
                        outcome = 'nxpass'
                        break

            test_run = {
                'module': module,
                'name': name,
                'parameters': parameters,
                'outcome': outcome,
            }

            # Capture failure log and failed phase for failed tests
            # Classification is done server-side in processing.py
            if outcome == 'failed':
                failure_parts = []
                failed_phase = 'call'
                for phase in ['setup', 'call', 'teardown']:
                    if phase in report_phases and report_phases[phase].outcome == 'failed':
                        failed_phase = phase
                        report = report_phases[phase]
                        if hasattr(report, 'longreprtext') and report.longreprtext:
                            failure_parts.append(report.longreprtext)
                        # Include captured stdout/stderr for full context
                        for section_name, section_content in report.sections:
                            if section_content.strip():
                                failure_parts.append(f"--- {section_name} ---\n{section_content}")
                        break
                test_run['failed_phase'] = failed_phase
                if failure_parts:
                    log_text = '\n'.join(failure_parts)
                    # Use the same node_id-based naming as the manifest to stay consistent
                    # with how test runs are identified across the pipeline.
                    safe_node_id = node_id.replace('/', '_').replace('::', '_').replace('[', '_').replace(']', '_').replace(' ', '_')
                    log_filename = f"{safe_node_id}.log"
                    log_path = os.path.join(failure_logs_root, log_filename)
                    with open(log_path, 'w', encoding='utf-8') as lf:
                        lf.write(log_text)
                    test_run['failure_log_file'] = os.path.join('failure_logs', log_filename)

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
            if os.path.isdir(failure_logs_root):
                for fname in os.listdir(failure_logs_root):
                    fpath = os.path.join(failure_logs_root, fname)
                    if os.path.isfile(fpath):
                        zf.write(fpath, arcname=os.path.join('failure_logs', fname))

        # Upload bundle        
        print("Uploading results bundle...")
        try:
            session_path = _upload_zip_bundle(zip_path) + "#batch-" + manifest['batch_name']
        except Exception as e:
            print(f"ERROR: Failed to upload results bundle: {e}")
            return
        print("Upload complete.")

        log_url = server_url + session_path

        print(f"\nTest results available at {log_url}\n")

        if os.getenv("GITHUB_ACTIONS"):
            try:
                github_step_summary = os.getenv("GITHUB_STEP_SUMMARY")
                if github_step_summary:
                    with open(github_step_summary, "a") as f:
                        f.write(f"[View test results]({log_url})\n\n")
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


def _format_profiling_summary(summary: dict, model_name: str, wall_time: str | None = None) -> str:
    """Format an extracted profiling summary into a human-readable table.

    Dynamically prints all metrics present in summary (e.g. dma_time, compute_time, etc.) 
    along with their percentages, and includes wall time if provided.
    """
    if not summary.get('available'):
        return ""

    lines = []
    lines.append(f"  Model: {model_name}")
    lines.append(f"  {'─' * 52}")

    # Wall time first (if available).
    if wall_time is not None:
        try:
            secs = float(wall_time)
            wall_time_str = f"{secs:.3f}s" if secs >= 1.0 else f"{secs * 1000:.3f}ms"
        except ValueError:
            wall_time_str = wall_time
        lines.append(f"  {'WALL_TIME':<26s} {wall_time_str:>14s}")

    # OVERALL duration (special key — no percent).
    total_duration = summary.get('total_duration')
    if total_duration is not None:
        lines.append(f"  {'OVERALL':<26s} {total_duration:>14s}")

    # Pretty labels: strip _time suffix, replace underscores with spaces,
    # title-case.  E.g. 'dma_total_time' -> 'Dma Total'.
    def _label(key: str) -> str:
        return key.removesuffix('_time').replace('_', ' ').title()

    # Print every *_time key and its matching *_percent (if present).
    skip = {'total_duration', 'available'}
    for key, value in summary.items():
        if key in skip or value is None or not key.endswith('_time'):
            continue
        label = _label(key)
        pct_key = key.removesuffix('_time') + '_percent'
        pct = summary.get(pct_key)
        if pct is not None:
            lines.append(f"  {label:<26s} {value:>14s}  ({pct}%)")
        else:
            lines.append(f"  {label:<26s} {value:>14s}")

    return "\n".join(lines)


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print Host Profile Overview summary at the end of the pytest session.

    Only active when --update-astra-runtime is enabled.
    """
    if not config.getoption("--update-astra-runtime", default=False):
        return

    profiling_output_dir = config.getoption("--torq-runtime-profiling-output-dir", default=None)
    if not profiling_output_dir:
        return

    profiling_output_dir = Path(profiling_output_dir)
    if not profiling_output_dir.exists():
        return

    # Collect .pb files and wall times recorded by tests in this session
    pb_files_from_session = []
    wall_times: dict[Path, str] = {}  # pb_file -> wall_time string
    for node_id, report_phases in reports.items():
        for phase in ['call', 'setup']:
            if phase not in report_phases:
                continue
            report = report_phases[phase]
            props = dict(report.user_properties)
            pb_path = props.get('profiling_output')
            if pb_path and Path(pb_path).exists():
                pb = Path(pb_path)
                pb_files_from_session.append(pb)
                wt = props.get('wall_time')
                if wt:
                    wall_times[pb] = wt

    if not pb_files_from_session:
        return

    terminalreporter.section("Host Profile Overview")

    for pb_file in sorted(set(pb_files_from_session)):
        model_name = extract_model_name(pb_file.name)
        summary = extract_perfetto_summary(str(pb_file))
        formatted = _format_profiling_summary(summary, model_name, wall_times.get(pb_file))
        if formatted:
            terminalreporter.write_line(formatted)
            terminalreporter.write_line("")
