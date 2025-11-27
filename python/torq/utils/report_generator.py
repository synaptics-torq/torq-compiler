#!/usr/bin/env python3
"""
Abstract report generator module that can generate HTML and text reports from JSON test data.

This module provides a generic interface for generating test reports from structured JSON data,
making it reusable across different test suites.

Usage:
    # As a module
    from helpers.report_generator import TestReportGenerator
    generator = TestReportGenerator(json_data)
    generator.generate_html_report()
    generator.generate_text_report()
    
    # As a standalone script
    python3 report_generator.py test_data.json
    python3 report_generator.py test_data.json --html-only
    python3 report_generator.py test_data.json --html-output custom_report.html
"""

import os
import json
import datetime
from pathlib import Path


class TestReportGenerator:
    """
    Generic test report generator that accepts structured JSON data.
    
    JSON Schema:
    {
        "test_suite": {
            "name": "Test Suite Name",
            "timestamp": "2025-11-20 12:00:00",
            "base_dir": "/path/to/base/dir",
            "timeout": 180,
            "dry_run": false,
            "args": {...}  # Additional arguments
        },
        "results": [
            {
                "model_path": "/path/to/model.mlir",
                "model_dir": "model001",
                "model_name": "conv_3X3_synai.mlir",
                "success": true/false,
                "failure_stage": "compile"/"run"/null,
                "max_diff": "0.123456"/null,
                "log_file": "/path/to/log.log"
            }
        ]
    }
    """
    
    def __init__(self, json_data):
        """
        Initialize the report generator with JSON test data.
        
        Args:
            json_data: Dictionary containing test results in the expected schema
        """
        if isinstance(json_data, str):
            # If string provided, assume it's a file path
            with open(json_data, 'r') as f:
                self.data = json.load(f)
        elif isinstance(json_data, dict):
            self.data = json_data
        else:
            raise ValueError("json_data must be a dictionary or file path string")
        
        self.test_suite = self.data.get('test_suite', {})
        self.results = self.data.get('results', [])
        
        # Group results by model directory
        self.model_groups = {}
        for result in self.results:
            model_dir = result.get('model_dir', 'unknown')
            if model_dir not in self.model_groups:
                self.model_groups[model_dir] = []
            self.model_groups[model_dir].append(result)
        
        # Group failures by stage
        self.failures_by_stage = {}
        for result in self.results:
            if not result.get('success') and result.get('failure_stage'):
                stage = result['failure_stage']
                if stage not in self.failures_by_stage:
                    self.failures_by_stage[stage] = []
                self.failures_by_stage[stage].append(result)
    
    def get_display_stage(self, stage, max_diff=None):
        """Convert internal stage name to display-friendly name."""
        stage_display_names = {
            "import_tflite": "TFLite Import Error",
            "tflite_to_tosa": "TFLite Import Error",
            "optimize": "MLIR Optimization Error",
            "tosa_to_mlir": "MLIR Optimization Error",
            "compile": "Compilation Error",
            "mlir_to_vmfb": "Compilation Error",
            "run": "Execution Error",
            "execute_module": "Execution Error",
            "output_missing": "Missing Output",
            "missing_output": "Missing Output",
            "large_difference": "Output Difference",
            "exception": "Exception",
            "timeout": "Timeout",
            "assertion_error": "Assertion Failed",
            "crash": "Crash/Abort",
            "segfault": "Segmentation Fault",
            "not_implemented": "Not Implemented",
            "out_of_memory": "Out of Memory",
            "mlir_error": "MLIR Error",
            "compile_error": "Compile Error",
            "import_error": "Import Error",
            "shape_mismatch": "Shape Mismatch",
            "output_comparison": "Output Comparison Failed",
            "uncategorized": "Uncategorized Error",
            "keyboard_interrupt": "Interrupted"
        }
        
        if stage in stage_display_names:
            display_name = stage_display_names[stage]
            if stage == "large_difference" and max_diff is not None:
                try:
                    max_diff_formatted = f"{float(max_diff):.6f}"
                except (ValueError, TypeError):
                    max_diff_formatted = str(max_diff)
                return f"{display_name} > 1 (max: {max_diff_formatted})"
            return display_name
        
        return stage if stage else "Unknown"
    
    def generate_text_report(self, output_path=None):
        """Generate a text summary report."""
        if output_path is None:
            reports_dir = Path(self.test_suite.get('base_dir', '.')).parent / 'reports'
            reports_dir.mkdir(parents=True, exist_ok=True)
            output_path = reports_dir / f"{self.test_suite.get('name', 'test')}_summary.txt"
        
        success_count = sum(1 for r in self.results if r.get('success'))
        failed_count = len(self.results) - success_count
        success_rate = (success_count / len(self.results) * 100) if self.results else 0
        
        with open(output_path, 'w') as f:
            f.write(f"Test Suite: {self.test_suite.get('name', 'Unknown')}\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Run Time: {self.test_suite.get('timestamp', 'Unknown')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"SUMMARY:\n")
            f.write(f"  Total Tests: {len(self.results)}\n")
            f.write(f"  Passed: {success_count} ({success_rate:.1f}%)\n")
            f.write(f"  Failed: {failed_count}\n\n")
            
            if self.test_suite.get('timeout'):
                f.write(f"  Timeout: {self.test_suite['timeout']} seconds\n")
            if self.test_suite.get('dry_run'):
                f.write(f"  Mode: Dry Run\n")
            f.write("\n")
            
            # Breakdown by model group
            f.write("BREAKDOWN BY MODEL GROUP:\n")
            for model_dir in sorted(self.model_groups.keys()):
                group_results = self.model_groups[model_dir]
                group_success = sum(1 for r in group_results if r.get('success'))
                if group_success == len(group_results):
                    status = "Pass"
                elif group_success == 0:
                    status = "Failed"
                else:
                    status = f"{group_success}/{len(group_results)} passed"
                f.write(f"  {model_dir}: {status}\n")
            f.write("\n")
            
            # Failures by stage
            if failed_count > 0:
                f.write("FAILED MODELS BY STAGE:\n")
                for stage, failures in sorted(self.failures_by_stage.items()):
                    display_name = self.get_display_stage(stage)
                    f.write(f"\n  {display_name} ({len(failures)} failures):\n")
                    
                    for result in failures:
                        model_name = result.get('model_name', 'unknown')
                        model_dir = result.get('model_dir', 'unknown')
                        max_diff = result.get('max_diff')
                        
                        if stage == "large_difference" and max_diff:
                            f.write(f"    - {model_dir}/{model_name} (Max Diff: {max_diff})\n")
                        else:
                            f.write(f"    - {model_dir}/{model_name}\n")
        
        return str(output_path)
    
    def generate_html_report(self, output_path=None, json_file_path=None):
        """Generate an interactive HTML report."""
        if output_path is None:
            # If json_file_path is provided, generate HTML in same directory
            if json_file_path:
                json_path = Path(json_file_path)
                reports_dir = json_path.parent
                # Use same base name as JSON file
                base_name = json_path.stem.replace('_data_', '_report_')
                output_path = reports_dir / f"{base_name}.html"
            else:
                reports_dir = Path(self.test_suite.get('base_dir', '.')).parent / 'reports'
                reports_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_path = reports_dir / f"{self.test_suite.get('name', 'test')}_report_{timestamp}.html"
        
        success_count = sum(1 for r in self.results if r.get('success'))
        failed_count = len(self.results) - success_count
        success_rate = (success_count / len(self.results) * 100) if self.results else 0
        
        # Collect actual error types
        actual_error_types = set()
        for result in self.results:
            if not result.get('success'):
                stage = result.get('failure_stage')
                if stage:
                    actual_error_types.add(stage)
        
        # Filter button definitions
        all_filter_types = {
            'timeout': 'Timeout',
            'assertion_error': 'Assertion Failed',
            'crash': 'Crash/Abort',
            'segfault': 'Segmentation Fault',
            'not_implemented': 'Not Implemented',
            'out_of_memory': 'Out of Memory',
            'mlir_error': 'MLIR Error',
            'compile_error': 'Compile Error',
            'import_error': 'Import Error',
            'mlir_to_vmfb': 'Compilation Error',
            'execute_module': 'Execution Error',
            'large_difference': 'Max Difference',
            'shape_mismatch': 'Shape Mismatch',
            'missing_output': 'Missing Output',
            'uncategorized': 'Uncategorized'
        }
        
        with open(output_path, 'w') as f:
            self._write_html_header(f)
            self._write_html_styles(f)
            self._write_html_body_start(f, success_count, failed_count, success_rate)
            self._write_filter_buttons(f, all_filter_types, actual_error_types)
            self._write_model_groups(f)
            self._write_html_scripts(f)
            f.write("</body>\n</html>")
        
        return str(output_path)
    
    def _write_html_header(self, f):
        """Write HTML header."""
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Report - {}</title>
""".format(self.test_suite.get('name', 'Test Suite')))
    
    def _write_html_styles(self, f):
        """Write CSS styles."""
        f.write("""    <style>
        :root { --primary-color:#2563eb; --success-color:#10b981; --warning-color:#f59e0b; --error-color:#ef4444; --bg-color:#f9fafb; --card-bg:#ffffff; --text-color:#111827; --border-color:#e5e7eb; }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; font-size: 16px; line-height: 1.6; background: var(--bg-color); color: var(--text-color); padding: 20px; max-width: 1600px; margin: 0 auto; }
        
        header { margin-bottom: 30px; }
        header h1 { font-size: 36px; font-weight: 700; color: var(--text-color); margin-bottom: 8px; }
        header p { font-size: 16px; color: #6b7280; }
        
        .summary-card { background: var(--card-bg); border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid var(--border-color); }
        .summary-card h2 { font-size: 24px; font-weight: 600; margin-bottom: 16px; }
        
        .stats { display: flex; gap: 16px; margin-bottom: 20px; }
        .stat-box { flex: 1; min-width: 200px; padding: 24px; border-radius: 8px; color: white; text-align: center; }
        .stat-box.success { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
        .stat-box.error { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
        .stat-box.warning { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
        .stat-box h3 { font-size: 16px; font-weight: 500; margin-bottom: 8px; opacity: 0.95; }
        .stat-box p { font-size: 36px; font-weight: 700; }
        
        .filters { margin-top: 20px; }
        .filters h3 { font-size: 16px; font-weight: 600; margin-bottom: 12px; color: #374151; }
        .filter-buttons { display: flex; flex-wrap: wrap; gap: 8px; }
        .filter-btn { padding: 10px 18px; font-size: 15px; font-weight: 500; border: 1px solid var(--border-color); border-radius: 6px; background: white; cursor: pointer; transition: all 0.2s; color: #374151; }
        .filter-btn:hover { background: #f3f4f6; border-color: var(--primary-color); }
        .filter-btn.active { background: var(--primary-color); color: white; border-color: var(--primary-color); }
        
        .model-groups { margin-top: 8px; display: block; width: 100%; }
        
        .group-card { background: transparent; padding: 0; margin: 0; border: none; display: block; width: 100%; }
        
        .group-header { display: none; }
        .group-header h3 { font-size: 11px; font-weight: 700; color: var(--text-color); line-height: 1; margin: 0; }
        
        .badge { font-size: 9px; font-weight: 600; padding: 1px 4px; border-radius: 4px; color: white; line-height: 1; }
        .badge.success { background: var(--success-color); }
        .badge.error { background: var(--error-color); }
        .badge.warning { background: var(--warning-color); }
        
        .models-list { list-style: none; padding: 0; margin: 0; display: block; width: 100%; }
        
        .model-item { display: flex; align-items: center; gap: 10px; padding: 4px 8px; border-bottom: 1px solid #f3f4f6; transition: background 0.1s; line-height: 1.4; height: 28px; flex-wrap: nowrap; width: 100%; }
        .model-item::before { content: attr(data-group); font-size: 15px; font-weight: 700; color: var(--text-color); width: 100px; flex-shrink: 0; }
        .model-item:last-child { border-bottom: 1px solid #e5e7eb; }
        .model-item:hover { background: #f9fafb; }
        
        .model-status { font-size: 16px; width: 24px; text-align: center; flex-shrink: 0; line-height: 1; }
        .model-name { flex: 1 1 auto; min-width: 250px; max-width: 600px; font-family: 'Consolas', 'Monaco', monospace; font-size: 15px; font-weight: 500; color: var(--text-color); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; line-height: 1.4; }
        .model-stage { min-width: 160px; max-width: 300px; font-size: 14px; color: #6b7280; flex-shrink: 0; line-height: 1.4; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-right: 8px; }
        .model-stage.failed { color: var(--error-color); font-weight: 500; }
        
        .log-button { padding: 5px 12px; font-size: 13px; font-weight: 500; background: var(--primary-color); color: white; border: none; border-radius: 5px; cursor: pointer; transition: background 0.15s; flex-shrink: 0; line-height: 1; height: 24px; white-space: nowrap; margin-left: auto; }
        .log-button:hover { background: #1d4ed8; }
        
        .log-container { margin: 6px 0 6px 30px; padding: 12px; background: #1e293b; border-radius: 6px; border: 1px solid #334155; }
        .log-container pre { max-height: 400px; font-size: 13px; line-height: 1.5; white-space: pre-wrap; overflow: auto; color: #e2e8f0; font-family: 'Consolas', 'Monaco', monospace; margin: 0; }
        
        footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border-color); text-align: center; font-size: 15px; color: #9ca3af; }
        
        .hidden { display: none !important; }
    </style>
</head>
""")
    
    def _write_html_body_start(self, f, success_count, failed_count, success_rate):
        """Write HTML body start with summary."""
        f.write("""<body>
    <div class="container">
        <header>
            <h1>{} Test Results</h1>
            <p>Generated on: {}</p>
        </header>

        <div class="summary-card">
            <h2>Summary</h2>
            <div class="stats">
                <div class="stat-box success">
                    <h3>Passed</h3>
                    <p>{} / {}</p>
                </div>
                <div class="stat-box error">
                    <h3>Failed</h3>
                    <p>{} / {}</p>
                </div>
                <div class="stat-box {}">
                    <h3>Success Rate</h3>
                    <p>{:.1f}%</p>
                </div>
            </div>
        </div>

        <div class="summary-card">
            <h2>Model Groups</h2>
            
            <div class="filters">
                <h3>Filter By Status & Error Type:</h3>
                <div class="filter-buttons">
""".format(
            self.test_suite.get('name', 'Test Suite'),
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            success_count,
            len(self.results),
            failed_count,
            len(self.results),
            "success" if success_rate > 80 else "warning" if success_rate > 50 else "error",
            success_rate
        ))
    
    def _write_filter_buttons(self, f, all_filter_types, actual_error_types):
        """Write filter buttons."""
        f.write("""                    <button class="filter-btn active" data-filter="all" onclick="filterModels('all')">All</button>
                    <button class="filter-btn" data-filter="success" onclick="filterModels('success')">Passed</button>
                    <button class="filter-btn" data-filter="failure" onclick="filterModels('failure')">Failed</button>
""")
        
        for error_type, display_name in sorted(all_filter_types.items()):
            if error_type in actual_error_types:
                f.write(f'                    <button class="filter-btn" data-filter="{error_type}" onclick="filterModels(\'{error_type}\')">{display_name}</button>\n')
        
        f.write("""                </div>
            </div>
            
            <div class="model-groups">
""")
    
    def _write_model_groups(self, f):
        """Write model groups section."""
        f.write("""                <ul class="models-list">
""")
        
        for model_dir in sorted(self.model_groups.keys()):
            group_results = self.model_groups[model_dir]
            
            for result in group_results:
                success = result.get('success')
                model_name = result.get('model_name', 'unknown')
                failure_stage = result.get('failure_stage')
                max_diff = result.get('max_diff')
                log_file = result.get('log_file', '')
                
                status_icon = "✅" if success else "❌"
                status_class = "success" if success else "failure"
                
                error_class = f"error-{failure_stage}" if failure_stage and not success else ""
                
                f.write(f'                    <li class="model-item {status_class} {error_class}" data-group="{model_dir}" data-status="{"success" if success else "failure"}" data-error="{failure_stage or ""}">\n')
                f.write(f'                        <span class="model-status">{status_icon}</span>\n')
                f.write(f'                        <span class="model-name">{model_name}</span>\n')
                
                if success:
                    f.write('                        <span class="model-stage">Passed</span>\n')
                    f.write('                    </li>\n')
                else:
                    display_stage = self.get_display_stage(failure_stage, max_diff)
                    f.write(f'                        <span class="model-stage failed">{display_stage}</span>\n')
                    
                    # Generate a unique ID for the log container
                    model_id = f"{model_dir}-{model_name}".replace('.', '-').replace('/', '-')
                    f.write(f'                        <button class="log-button" onclick="toggleLog(\'{model_id}\')">View Log</button>\n')
                    f.write('                    </li>\n')
                    
                    # Add log container as sibling div after the </li>
                    f.write(f'                    <div id="log_{model_id}" class="log-container" style="display:none;">\n')
                    f.write('                        <pre>')
                    
                    if log_file and os.path.exists(log_file):
                        try:
                            with open(log_file, 'r') as log_f:
                                log_content = log_f.read()
                                # Escape HTML special chars
                                log_content = log_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                                f.write(log_content)
                        except Exception as e:
                            f.write(f"Error reading log file: {e}")
                    else:
                        f.write("No log file available")
                    
                    f.write('</pre>\n')
                    f.write('                    </div>\n')
        
        f.write("""                </ul>
            </div>
        </div>
""")
    
    def _write_html_scripts(self, f):
        """Write JavaScript for interactivity."""
        f.write("""    <script>
        function toggleLog(id) {
            const logElem = document.getElementById('log_' + id);
            if (logElem) {
                if (logElem.style.display === "none" || logElem.style.display === "") {
                    logElem.style.display = "block";
                } else {
                    logElem.style.display = "none";
                }
            } else {
                console.error('Log element not found:', 'log_' + id);
            }
        }
        
        function filterModels(filterType) {
            // Update active button
            document.querySelectorAll('.filter-btn').forEach(btn => {
                if(btn.getAttribute('data-filter') === filterType) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
            
            // Filter model items
            const items = document.querySelectorAll('.model-item');
            items.forEach(item => {
                const status = item.getAttribute('data-status');
                const error = item.getAttribute('data-error');
                
                if (filterType === 'all') {
                    item.style.display = '';
                } else if (filterType === 'success' && status === 'success') {
                    item.style.display = '';
                } else if (filterType === 'failure' && status === 'failure') {
                    item.style.display = '';
                } else if (error === filterType) {
                    item.style.display = '';
                } else {
                    item.style.display = 'none';
                }
            });
            
            // Also hide/show log containers
            const logs = document.querySelectorAll('.log-container');
            logs.forEach(log => {
                // Extract model ID from log_ prefix
                const modelId = log.id.replace('log_', '');
                const modelItem = document.querySelector(`.model-item button[onclick*="${modelId}"]`)?.closest('.model-item');
                if (modelItem && modelItem.style.display === 'none') {
                    log.style.display = 'none';
                }
            });
        }
    </script>
    
    <footer>
        Generated by Torq Compiler Test Suite
    </footer>
""")


# Legacy function wrappers for backward compatibility
def generate_html_report(results, model_groups, failures_by_stage, args):
    """Legacy wrapper for backward compatibility."""
    # Convert old format to new JSON format
    json_data = {
        "test_suite": {
            "name": getattr(args, 'test_name', 'Model Tests'),
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "base_dir": getattr(args, 'base_dir', '.'),
            "timeout": getattr(args, 'timeout', 180),
            "dry_run": getattr(args, 'dry_run', False),
            "args": vars(args) if hasattr(args, '__dict__') else {}
        },
        "results": []
    }
    
    # Convert results to new format
    for model_path, success, failure_stage, max_diff in results:
        json_data["results"].append({
            "model_path": model_path,
            "model_dir": os.path.basename(os.path.dirname(model_path)),
            "model_name": os.path.basename(model_path),
            "success": success,
            "failure_stage": failure_stage,
            "max_diff": max_diff,
            "log_file": ""
        })
    
    generator = TestReportGenerator(json_data)
    return generator.generate_html_report()


def generate_text_report(results, model_groups, failures_by_stage, report_path, args):
    """Legacy wrapper for backward compatibility."""
    json_data = {
        "test_suite": {
            "name": getattr(args, 'test_name', 'Model Tests'),
            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "base_dir": getattr(args, 'base_dir', '.'),
            "timeout": getattr(args, 'timeout', 180),
            "dry_run": getattr(args, 'dry_run', False)
        },
        "results": []
    }
    
    for model_path, success, failure_stage, max_diff in results:
        json_data["results"].append({
            "model_path": model_path,
            "model_dir": os.path.basename(os.path.dirname(model_path)),
            "model_name": os.path.basename(model_path),
            "success": success,
            "failure_stage": failure_stage,
            "max_diff": max_diff,
            "log_file": ""
        })
    
    generator = TestReportGenerator(json_data)
    return generator.generate_text_report(report_path)


# Keep legacy helper functions
def get_display_stage(stage, max_diff=None):
    """Legacy helper function."""
    generator = TestReportGenerator({"test_suite": {}, "results": []})
    return generator.get_display_stage(stage, max_diff)


def debug_log_paths(model_path, model_name):
    """Return a list of possible log file paths."""
    model_dir = os.path.dirname(model_path)
    model_basename = os.path.splitext(os.path.basename(model_name))[0]
    
    return [
        os.path.join(os.getcwd(), "logs", f"{model_basename}_process.log"),
        os.path.join(model_dir, "logs", f"{model_basename}_process.log"),
        os.path.join(os.path.dirname(os.path.dirname(model_path)), 
                    os.path.basename(model_dir), "logs", 
                    f"{model_basename}_process.log")
    ]


def ensure_log_in_common_location(src_log_path, content=None):
    """Ensure log file exists in the common location."""
    common_logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(common_logs_dir, exist_ok=True)
    
    dst_log_path = os.path.join(common_logs_dir, os.path.basename(src_log_path))
    
    try:
        if content is not None:
            with open(dst_log_path, 'w') as dst_file:
                if isinstance(content, list):
                    dst_file.writelines(content)
                else:
                    dst_file.write(content)
        elif os.path.exists(src_log_path):
            with open(src_log_path, 'r') as src_file:
                log_content = src_file.read()
            with open(dst_log_path, 'w') as dst_file:
                dst_file.write(log_content)
    except Exception as e:
        print(f"Error copying log file: {e}")
        
    return dst_log_path, src_log_path, dst_log_path


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Generate HTML and text reports from JSON test data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate both HTML and text reports from JSON file
  python report_generator.py test_data.json
  
  # Generate only HTML report
  python report_generator.py test_data.json --html-only
  
  # Generate only text report
  python report_generator.py test_data.json --text-only
  
  # Specify custom output paths
  python report_generator.py test_data.json --html-output report.html --text-output summary.txt
        """
    )
    
    parser.add_argument(
        'json_file',
        help='Path to JSON file containing test data'
    )
    parser.add_argument(
        '--html-output',
        help='Path for HTML report output (default: auto-generated based on JSON filename)'
    )
    parser.add_argument(
        '--text-output',
        help='Path for text report output (default: auto-generated based on JSON filename)'
    )
    parser.add_argument(
        '--html-only',
        action='store_true',
        help='Generate only HTML report'
    )
    parser.add_argument(
        '--text-only',
        action='store_true',
        help='Generate only text report'
    )
    
    args = parser.parse_args()
    
    # Validate JSON file exists
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file '{args.json_file}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Load JSON data
    try:
        with open(args.json_file, 'r') as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading JSON file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create report generator
    generator = TestReportGenerator(json_data)
    
    # Generate reports
    try:
        if not args.text_only:
            html_path = generator.generate_html_report(
                output_path=args.html_output,
                json_file_path=args.json_file
            )
            print(f"HTML report generated: {html_path}")
        
        if not args.html_only:
            text_path = generator.generate_text_report(
                output_path=args.text_output
            )
            print(f"Text report generated: {text_path}")
    
    except Exception as e:
        print(f"Error generating report: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
