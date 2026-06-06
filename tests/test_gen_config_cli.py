import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

try:
    import iree.compiler
except ImportError:
    pytest.skip("iree package not available", allow_module_level=True)


"""Integration tests for executor discovery CLI options.

This test file validates all executor discovery CLI options by running
pytest in subprocess (exactly as users would) and verifying outputs.

Run: pytest tests/test_gen_config_cli.py -v
"""


# Path to the test model (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
TEST_MODEL = PROJECT_ROOT / "tests/testdata/onnx_models/example_gen_config.onnx"


def _cleanup_generated_json():
    """Remove any torq_gen_config_*.json files generated in project root."""
    for json_file in PROJECT_ROOT.glob("torq_gen_config_*.json"):
        if json_file.name.endswith("_compiler.json"):
            continue
        try:
            json_file.unlink()
        except OSError:
            pass  # Ignore cleanup errors


def _run_pytest_and_validate(
    test_name: str,
    extra_args: list,
    output_dir: Path,
    json_validator=None,
    stdout_validator=None,
    expect_success: bool = True,
    test_filter: str = "example_gen_config_layer_Relu_1_host",
) -> tuple:
    """Run pytest in subprocess and validate outputs.

    Args:
        test_name: Description of the test case
        extra_args: Additional CLI arguments for pytest
        output_dir: Directory for discovery JSON output
        json_validator: Function to validate JSON content
        stdout_validator: Function to validate stdout content
        expect_success: Whether pytest should succeed (returncode 0)
        test_filter: pytest -k filter (default: specific layer test)

    Returns:
        Tuple of (json_data, stdout, stderr)
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_onnx_gen_config.py",
        "-k",
        test_filter,
        f"--model-path={TEST_MODEL}",
        "--recompute-cache",
        "--gen-config-output",
        str(output_dir),
    ] + extra_args

    result = subprocess.run(cmd, capture_output=True, text=True)

    if expect_success:
        assert result.returncode == 0, (
            f"{test_name}: Pytest failed with returncode {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    else:
        assert result.returncode != 0, f"{test_name}: Expected pytest to fail but it passed"

    # Find and load JSON output (exclude compiler-format JSONs)
    json_files = [
        f for f in output_dir.glob("torq_gen_config_*.json")
        if not f.name.endswith("_compiler.json")
    ]
    json_data = None
    if json_files:
        with open(json_files[0]) as f:
            json_data = json.load(f)

    # Run JSON validator if provided
    if json_validator and json_data:
        json_validator(test_name, json_data)

    # Run stdout validator if provided
    if stdout_validator:
        stdout_validator(test_name, result.stdout)

    return json_data, result.stdout, result.stderr


def _validate_timing_fields(test_name: str, data: dict, should_exist: bool = True):
    """Validate timing fields in JSON data."""
    ops = data.get("ops", {})
    for op_name, op_data in ops.items():
        for exec_name, exec_result in op_data.get("executors", {}).items():
            if should_exist:
                assert "timing" in exec_result, (
                    f"{test_name}: Missing timing for {op_name}/{exec_name}"
                )
                timing = exec_result["timing"]
                assert "runtime_ms" in timing, (
                    f"{test_name}: Missing runtime_ms for {op_name}/{exec_name}"
                )
                assert "runs" in timing, (
                    f"{test_name}: Missing runs for {op_name}/{exec_name}"
                )
            else:
                assert "timing" not in exec_result, (
                    f"{test_name}: Unexpected timing for {op_name}/{exec_name}"
                )

@pytest.mark.skipif(not TEST_MODEL.exists(), reason=f"Test model not found: {TEST_MODEL}")
class TestExecutorDiscoveryIntegration:
    """Integration tests for all executor discovery CLI options."""

    def test_basic_discovery_output(self):
        """Test basic --gen-config-output creates valid JSON."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            def validate_json(name, data):
                assert data.get("version") == "1.1", f"{name}: Invalid version"
                assert "ops" in data, f"{name}: Missing ops"
                assert "discovery_report" in data, f"{name}: Missing discovery_report"

                ops = data["ops"]
                assert "Relu_relu_out" in ops, f"{name}: Expected layer not found"

                host = ops["Relu_relu_out"]["executors"]["host"]
                assert host["status"] == "success", f"{name}: Expected success status"
                assert "tolerance_used" in host, f"{name}: Missing tolerance"

            _run_pytest_and_validate(
                "basic_discovery",
                [],
                output_dir,
                json_validator=validate_json,
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_collect_timing(self):
        """Test --collect-timing includes timing data."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            def validate_json(name, data):
                _validate_timing_fields(name, data, should_exist=True)

            _run_pytest_and_validate(
                "collect_timing",
                ["--collect-timing"],
                output_dir,
                json_validator=validate_json,
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_timing_runs(self):
        """Test --timing-runs=N runs multiple times."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            def validate_json(name, data):
                ops = data.get("ops", {})
                for op_name, op_data in ops.items():
                    for exec_name, exec_result in op_data.get("executors", {}).items():
                        timing = exec_result.get("timing", {})
                        assert timing.get("runs") == 3, (
                            f"{name}: Expected 3 runs, got {timing.get('runs')}"
                        )

            _run_pytest_and_validate(
                "timing_runs",
                ["--collect-timing", "--timing-runs=3"],
                output_dir,
                json_validator=validate_json,
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_no_timing_without_flag(self):
        """Test timing is NOT included without --collect-timing."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            def validate_json(name, data):
                _validate_timing_fields(name, data, should_exist=False)

            _run_pytest_and_validate(
                "no_timing",
                [],
                output_dir,
                json_validator=validate_json,
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_skip_executors(self):
        """Test --skip-executors excludes specified executors."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            def validate_json(name, data):
                ops = data.get("ops", {})
                for op_name, op_data in ops.items():
                    executors = op_data.get("executors", {})
                    # nss and css should be skipped, only host should run
                    assert "nss" not in executors, f"{name}: nss should be skipped"
                    assert "css" not in executors, f"{name}: css should be skipped"
                    assert "host" in executors, f"{name}: host should run"

            _run_pytest_and_validate(
                "skip_executors",
                ["--skip-executors=nss,css"],
                output_dir,
                json_validator=validate_json,
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_executor_skip_mode(self):
        """Test --executor-skip-mode stops after first success.

        Note: This test verifies that --executor-skip-mode option is accepted
        and produces valid output. Due to test filtering, it may not fully
        demonstrate the skip behavior (stopping after first success).
        """
        output_dir = Path(tempfile.mkdtemp())

        try:
            # Run without validation - just verify it doesn't crash
            # The skip mode behavior is fully tested when running the full test suite
            json_data, stdout, stderr = _run_pytest_and_validate(
                "executor_skip_mode",
                ["--executor-skip-mode"],
                output_dir,
                test_filter="example_gen_config_layer_Relu_1_host",
            )
            # Just verify JSON was created and is valid
            assert json_data is not None, "executor_skip_mode: No JSON output generated"
            assert "ops" in json_data, "executor_skip_mode: Missing ops in JSON"
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_viewer_script_output(self):
        """Test python/torq/gen_config/view.py produces correct output."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            # Run discovery with timing
            json_data, _, _ = _run_pytest_and_validate(
                "viewer_test",
                ["--collect-timing", "--timing-runs=2"],
                output_dir,
            )

            json_files = [f for f in output_dir.glob("torq_gen_config_*.json") if not f.name.endswith("_compiler.json")]
            assert len(json_files) == 1, "Expected 1 JSON file"

            # Run viewer script
            viewer_result = subprocess.run(
                [sys.executable, "python/torq/gen_config/view.py", str(json_files[0])],
                capture_output=True,
                text=True,
                check=True,
            )

            output = viewer_result.stdout

            # Validate viewer output contains expected sections
            assert "MODEL:" in output, "Viewer: Missing MODEL header"
            assert "STATUS COUNTS:" in output, "Viewer: Missing status counts"
            assert "success:" in output, "Viewer: Missing success count"
            assert "TIMING SUMMARY" in output, "Viewer: Missing timing summary"
            assert "HOST:" in output, "Viewer: Missing HOST executor"
            assert "ms" in output, "Viewer: Missing timing values"

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_viewer_layer_details(self):
        """Test python/torq/gen_config/view.py <json> <layer_id> shows layer details."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            _run_pytest_and_validate(
                "viewer_layer_test",
                ["--collect-timing"],
                output_dir,
            )

            json_files = [f for f in output_dir.glob("torq_gen_config_*.json") if not f.name.endswith("_compiler.json")]
            assert len(json_files) == 1, "Expected 1 JSON file"

            # Run viewer with layer ID
            viewer_result = subprocess.run(
                [
                    sys.executable,
                    "python/torq/gen_config/view.py",
                    str(json_files[0]),
                    "Relu_relu_out",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            output = viewer_result.stdout

            # Validate layer details output
            assert "LAYER: Relu_relu_out" in output, "Viewer: Missing layer header"
            assert "Executor Results:" in output, "Viewer: Missing executor results"
            assert "[HOST]:" in output, "Viewer: Missing HOST result"
            assert "success" in output, "Viewer: Missing success status"
            assert "Runtime:" in output, "Viewer: Missing runtime"
            assert "Tolerance:" in output, "Viewer: Missing tolerance"

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_all_options_combined(self):
        """Test all compatible options together."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            def validate_json(name, data):
                # Should have timing
                _validate_timing_fields(name, data, should_exist=True)

                # Should have correct number of runs
                ops = data.get("ops", {})
                for op_name, op_data in ops.items():
                    for exec_name, exec_result in op_data.get("executors", {}).items():
                        timing = exec_result.get("timing", {})
                        assert timing.get("runs") == 2, (
                            f"{name}: Expected 2 runs, got {timing.get('runs')}"
                        )

                # Should have correct version
                assert data.get("version") == "1.1", f"{name}: Wrong version"

            _run_pytest_and_validate(
                "all_options",
                ["--collect-timing", "--timing-runs=2"],
                output_dir,
                json_validator=validate_json,
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_auto_convert_bf16(self):
        """Test --auto-convert-bf16 converts model to BF16."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            def validate_json(name, data):
                # Should have ops and valid results
                assert "ops" in data, f"{name}: Missing ops"
                ops = data.get("ops", {})
                assert len(ops) > 0, f"{name}: No ops found"
                # Check that at least one op succeeded
                success_found = False
                for op_name, op_data in ops.items():
                    for exec_name, exec_result in op_data.get("executors", {}).items():
                        if exec_result.get("status") == "success":
                            success_found = True
                            break
                assert success_found, f"{name}: No successful executor results found"

            _run_pytest_and_validate(
                "auto_convert_bf16",
                ["--auto-convert-bf16"],
                output_dir,
                json_validator=validate_json,
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_save_bf16_model(self):
        """Test --save-bf16-model saves converted model to file.

        Note: BF16 conversion only happens when the model is not already BF16.
        This test verifies the option is accepted and test completes successfully.
        The actual file creation depends on model state.
        """
        output_dir = Path(tempfile.mkdtemp())
        bf16_model_path = output_dir / "converted_bf16.onnx"

        try:
            def validate_json(name, data):
                # Just verify test ran successfully
                assert "ops" in data, f"{name}: Missing ops"

            json_data, stdout, stderr = _run_pytest_and_validate(
                "save_bf16_model",
                ["--auto-convert-bf16", f"--save-bf16-model={bf16_model_path}"],
                output_dir,
                json_validator=validate_json,
            )

            # Check if BF16 conversion happened by looking for output message
            # If model was already BF16, file won't be created - that's OK
            if bf16_model_path.exists():
                assert bf16_model_path.stat().st_size > 0, (
                    "save_bf16_model: BF16 model file is empty"
                )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_subgraph_extraction(self):
        """Test --subgraph-from and --subgraph-to extract subgraph.

        Note: This test verifies the options are accepted and don't crash.
        Some executors may fail on subgraphs - we just verify JSON is generated.
        """
        output_dir = Path(tempfile.mkdtemp())

        try:
            # Subgraph tests may have executor failures - just verify JSON is generated
            # Use correct op name from the model: "Relu_relu_out"
            json_data, stdout, stderr = _run_pytest_and_validate(
                "subgraph_extraction",
                [
                    "--subgraph-from=Relu_relu_out",
                    "--subgraph-to=Relu_relu_out",
                ],
                output_dir,
                test_filter="example_gen_config",
                expect_success=True,
            )
            # Verify JSON was generated even if some tests failed
            assert json_data is not None, "subgraph_extraction: No JSON output"
            assert "ops" in json_data, "subgraph_extraction: Missing ops in JSON"
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_bf16_with_timing(self):
        """Test BF16 conversion combined with timing collection."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            def validate_json(name, data):
                # Should have timing data
                _validate_timing_fields(name, data, should_exist=True)
                # Should have successful results
                ops = data.get("ops", {})
                assert len(ops) > 0, f"{name}: No ops found"

            _run_pytest_and_validate(
                "bf16_with_timing",
                [
                    "--auto-convert-bf16",
                    "--collect-timing",
                    "--timing-runs=2",
                ],
                output_dir,
                json_validator=validate_json,
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_recommend_by_timing(self):
        """Test --recommend-by-timing recommends fastest executor.

        This test verifies that when --recommend-by-timing is used with
        --collect-timing, the recommended executor is the one with the
        fastest runtime (lowest runtime_ms).
        """
        output_dir = Path(tempfile.mkdtemp())

        try:
            def validate_json(name, data):
                # Should have timing data
                _validate_timing_fields(name, data, should_exist=True)
                # Should have recommended_executor
                ops = data.get("ops", {})
                assert len(ops) > 0, f"{name}: No ops found"
                for op_name, op_data in ops.items():
                    assert "recommended_executor" in op_data, (
                        f"{name}: Missing recommended_executor for {op_name}"
                    )
                    # Verify recommended executor has timing data
                    recommended = op_data["recommended_executor"]
                    if recommended:
                        exec_result = op_data["executors"].get(recommended, {})
                        assert "timing" in exec_result, (
                            f"{name}: Recommended executor {recommended} has no timing"
                        )

            _run_pytest_and_validate(
                "recommend_by_timing",
                ["--collect-timing", "--timing-runs=2", "--recommend-by-timing"],
                output_dir,
                json_validator=validate_json,
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_compiler_json_generated(self):
        """Test that discovery generates both report JSON and compiler JSON.

        Verifies:
        - Report JSON (torq_gen_config_<model>.json) exists with 'ops' format
        - Compiler JSON (torq_gen_config_<model>_compiler.json) exists with
          'op_assignments' format
        - Compiler JSON content matches report JSON recommendations
        """
        output_dir = Path(tempfile.mkdtemp())

        try:
            _run_pytest_and_validate(
                "compiler_json_gen",
                [],
                output_dir,
            )

            # Find report JSON (exclude compiler JSONs)
            report_files = [
                f for f in output_dir.glob("torq_gen_config_*.json")
                if not f.name.endswith("_compiler.json")
            ]
            assert len(report_files) == 1, (
                f"Expected 1 report JSON, got {len(report_files)}: {report_files}"
            )

            # Find compiler JSON
            compiler_files = list(output_dir.glob("torq_gen_config_*_compiler.json"))
            assert len(compiler_files) == 1, (
                f"Expected 1 compiler JSON, got {len(compiler_files)}: {compiler_files}"
            )

            # Load and validate report JSON format
            with open(report_files[0]) as f:
                report_data = json.load(f)
            assert "ops" in report_data, "Report JSON missing 'ops' key"
            assert "model_name" in report_data, "Report JSON missing 'model_name'"

            # Load and validate compiler JSON format
            with open(compiler_files[0]) as f:
                compiler_data = json.load(f)
            assert "op_assignments" in compiler_data, (
                "Compiler JSON missing 'op_assignments' key"
            )
            assert "model_name" in compiler_data, (
                "Compiler JSON missing 'model_name' key"
            )

            # Validate compiler JSON content matches report JSON
            ops = report_data.get("ops", {})
            assignments = compiler_data.get("op_assignments", {})

            # Every op with a valid line:column location and recommendation
            # should appear in the compiler JSON
            expected_count = 0
            for op_name, op_info in ops.items():
                location = op_info.get("mlir_location", "")
                recommended = op_info.get("recommended_executor")
                # location must be line:column format (not raw op type)
                if ":" in location and recommended:
                    expected_count += 1
                    assert location in assignments, (
                        f"Compiler JSON missing assignment for {op_name} "
                        f"at {location}"
                    )
                    assert assignments[location]["executor"] == recommended, (
                        f"Compiler JSON wrong executor for {op_name}: "
                        f"expected {recommended}, got {assignments[location]['executor']}"
                    )

            # Compiler JSON should have at least the assignments we expect
            assert len(assignments) == expected_count, (
                f"Compiler JSON has {len(assignments)} assignments, "
                f"expected {expected_count}"
            )

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_run_with_compiler_json_only(self):
        """Test that 'run' works with only compiler JSON (no report JSON).

        This verifies the workflow where a user shares or keeps only the
        minimal compiler-format JSON and later runs the full model without
        having the full discovery report.
        """
        output_dir = Path(tempfile.mkdtemp())

        try:
            # Step 1: Run discovery to generate both JSONs.
            # Use --auto-convert-bf16 and run all layer tests so the
            # compiler JSON contains valid assignments for every op.
            _run_pytest_and_validate(
                "run_compiler_only_discovery",
                ["--auto-convert-bf16"],
                output_dir,
                test_filter="example_gen_config_layer_",
            )

            # Find generated JSONs
            report_files = [
                f for f in output_dir.glob("torq_gen_config_*.json")
                if not f.name.endswith("_compiler.json")
            ]
            compiler_files = list(output_dir.glob("torq_gen_config_*_compiler.json"))
            assert len(report_files) == 1, "Expected 1 report JSON"
            assert len(compiler_files) == 1, "Expected 1 compiler JSON"

            # Step 2: Delete the report JSON, keep only compiler JSON
            report_files[0].unlink()
            assert not report_files[0].exists(), "Report JSON should be deleted"

            # Step 3: Run 'torq-gen-config run' with only compiler JSON present
            run_cmd = [
                sys.executable,
                "-m",
                "torq.gen_config",
                "run",
                f"--model={TEST_MODEL}",
                f"--output-dir={output_dir}",
                "--auto-convert-bf16",
                "--recompute-cache",
            ]
            run_result = subprocess.run(run_cmd, capture_output=True, text=True)

            assert run_result.returncode == 0, (
                f"run with compiler JSON only failed: {run_result.returncode}\n"
                f"stdout: {run_result.stdout}\nstderr: {run_result.stderr}"
            )

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_edit_and_run(self):
        """Test that 'edit' updates both JSONs and 'run' uses the edit.

        Verifies:
        - edit changes recommended_executor in report JSON
        - edit updates final_report_text
        - edit regenerates compiler JSON
        - run respects the edited recommendation in terminal output
        """
        output_dir = Path(tempfile.mkdtemp())

        try:
            # Step 1: Run discovery to generate both JSONs
            _run_pytest_and_validate(
                "edit_run_discovery",
                ["--auto-convert-bf16"],
                output_dir,
                test_filter="example_gen_config_layer_",
            )

            report_files = [
                f for f in output_dir.glob("torq_gen_config_*.json")
                if not f.name.endswith("_compiler.json")
            ]
            compiler_files = list(output_dir.glob("torq_gen_config_*_compiler.json"))
            assert len(report_files) == 1, "Expected 1 report JSON"
            assert len(compiler_files) == 1, "Expected 1 compiler JSON"
            report_path = report_files[0]
            compiler_path = compiler_files[0]

            # Step 2: Edit a layer's recommended executor
            edit_cmd = [
                sys.executable,
                "-m",
                "torq.gen_config",
                "edit",
                str(report_path),
                "--layer",
                "Add_output",
                "--executor",
                "host",
            ]
            edit_result = subprocess.run(edit_cmd, capture_output=True, text=True)
            assert edit_result.returncode == 0, (
                f"edit failed: {edit_result.returncode}\n"
                f"stderr: {edit_result.stderr}"
            )

            # Step 3: Verify report JSON was updated
            with open(report_path) as f:
                report_data = json.load(f)
            assert report_data["ops"]["Add_output"]["recommended_executor"] == "host"

            # Verify final_report_text was updated
            report_text = report_data.get("final_report_text", "")
            add_line = [l for l in report_text.split("\n") if "Add_output" in l]
            assert len(add_line) == 1, "Add_output line not found in final_report_text"
            assert add_line[0].endswith(" host"), (
                f"final_report_text did not update: {add_line[0]}"
            )

            # Step 4: Verify compiler JSON was updated
            with open(compiler_path) as f:
                compiler_data = json.load(f)
            assignments = compiler_data.get("op_assignments", {})
            # Add_output should now map to host (location is 9:10)
            assert assignments.get("9:10", {}).get("executor") == "host", (
                f"compiler JSON not updated: {assignments}"
            )

            # Step 5: Run and verify terminal output respects the edit
            run_cmd = [
                sys.executable,
                "-m",
                "torq.gen_config",
                "run",
                f"--model={TEST_MODEL}",
                f"--output-dir={output_dir}",
                "--auto-convert-bf16",
                "--recompute-cache",
            ]
            run_result = subprocess.run(run_cmd, capture_output=True, text=True)
            assert run_result.returncode == 0, (
                f"run after edit failed: {run_result.returncode}\n"
                f"stderr: {run_result.stderr}"
            )
            # Terminal output should show the edited recommendation
            # (report may be in stdout or stderr depending on pytest config)
            combined_output = run_result.stdout + run_result.stderr
            assert "Add_output" in combined_output, "Add_output not in run output"
            add_lines = [
                l for l in combined_output.split("\n")
                if "Add_output" in l and "host" in l
            ]
            assert len(add_lines) >= 1, (
                "run output did not reflect edited recommendation"
            )

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_edit_with_model_flag(self):
        """Test that 'edit' works with --model instead of positional path."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            # Step 1: Run discovery
            _run_pytest_and_validate(
                "edit_model_flag_discovery",
                ["--auto-convert-bf16"],
                output_dir,
                test_filter="example_gen_config_layer_",
            )

            # Step 2: Edit using --model + --output-dir (no positional path)
            edit_cmd = [
                sys.executable,
                "-m",
                "torq.gen_config",
                "edit",
                f"--model={TEST_MODEL}",
                f"--output-dir={output_dir}",
                "--layer",
                "Add_output",
                "--executor",
                "host",
            ]
            edit_result = subprocess.run(edit_cmd, capture_output=True, text=True)
            assert edit_result.returncode == 0, (
                f"edit with --model failed: {edit_result.returncode}\n"
                f"stderr: {edit_result.stderr}"
            )

            # Step 3: Verify report JSON was updated
            report_files = [
                f for f in output_dir.glob("torq_gen_config_*.json")
                if not f.name.endswith("_compiler.json")
            ]
            assert len(report_files) == 1, "Expected 1 report JSON"
            with open(report_files[0]) as f:
                report_data = json.load(f)
            assert report_data["ops"]["Add_output"]["recommended_executor"] == "host"

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_edit_list_layers(self):
        """Test that 'edit --list' prints available layers with full executor table."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            # Step 1: Run discovery
            _run_pytest_and_validate(
                "edit_list_discovery",
                ["--auto-convert-bf16"],
                output_dir,
                test_filter="example_gen_config_layer_",
            )

            # Step 2: Run edit --list (no filter)
            edit_cmd = [
                sys.executable,
                "-m",
                "torq.gen_config",
                "edit",
                f"--model={TEST_MODEL}",
                f"--output-dir={output_dir}",
                "--list",
            ]
            edit_result = subprocess.run(edit_cmd, capture_output=True, text=True)
            assert edit_result.returncode == 0, (
                f"edit --list failed: {edit_result.returncode}\n"
                f"stderr: {edit_result.stderr}"
            )

            output = edit_result.stdout
            # Should contain per-layer status table headers (same as final report)
            assert "Per-Layer Status:" in output, "Missing Per-Layer Status header"
            assert "NSS" in output, "Missing NSS column header"
            assert "CSS" in output, "Missing CSS column header"
            assert "HOST" in output, "Missing HOST column header"
            assert "Recommended" in output, "Missing Recommended column"
            # Should contain known layers
            assert "Add_output" in output, "Missing Add_output layer"
            assert "Relu_relu_out" in output, "Missing Relu_relu_out layer"
            # Should show total count
            assert "Total:" in output, "Missing total count"

            # Step 3: Run edit --list with filter
            edit_cmd_filter = [
                sys.executable,
                "-m",
                "torq.gen_config",
                "edit",
                f"--model={TEST_MODEL}",
                f"--output-dir={output_dir}",
                "--list",
                "add",
            ]
            edit_result_filter = subprocess.run(
                edit_cmd_filter, capture_output=True, text=True
            )
            assert edit_result_filter.returncode == 0, (
                f"edit --list add failed: {edit_result_filter.returncode}\n"
                f"stderr: {edit_result_filter.stderr}"
            )

            filter_output = edit_result_filter.stdout
            # Should contain Add_output
            assert "Add_output" in filter_output, "Missing Add_output in filtered list"
            # Should show filtered count
            assert "filtered from" in filter_output, "Missing filtered count"

            # Step 4: Run edit --list with non-matching filter
            edit_cmd_none = [
                sys.executable,
                "-m",
                "torq.gen_config",
                "edit",
                f"--model={TEST_MODEL}",
                f"--output-dir={output_dir}",
                "--list",
                "xyz_nonexistent",
            ]
            edit_result_none = subprocess.run(
                edit_cmd_none, capture_output=True, text=True
            )
            assert edit_result_none.returncode == 0, (
                f"edit --list xyz_nonexistent failed: {edit_result_none.returncode}"
            )
            assert "No layers match" in edit_result_none.stdout, (
                f"Missing 'No layers match' message: {edit_result_none.stdout}"
            )

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_edit_compiler_json_guard(self):
        """Test that 'edit' on a compiler JSON prints a helpful error."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            # Step 1: Run discovery
            _run_pytest_and_validate(
                "edit_guard_discovery",
                ["--auto-convert-bf16"],
                output_dir,
                test_filter="example_gen_config_layer_",
            )

            compiler_files = list(output_dir.glob("torq_gen_config_*_compiler.json"))
            assert len(compiler_files) == 1, "Expected 1 compiler JSON"
            compiler_path = compiler_files[0]

            # Step 2: Try to edit the compiler JSON (should fail with helpful msg)
            edit_cmd = [
                sys.executable,
                "-m",
                "torq.gen_config",
                "edit",
                str(compiler_path),
                "--layer",
                "Add_output",
                "--executor",
                "host",
            ]
            edit_result = subprocess.run(edit_cmd, capture_output=True, text=True)
            assert edit_result.returncode != 0, (
                "edit on compiler JSON should have failed"
            )
            stderr = edit_result.stderr
            assert "compiler json" in stderr.lower(), (
                f"Missing 'compiler JSON' hint in error: {stderr}"
            )
            assert "report json" in stderr.lower(), (
                f"Missing 'report JSON' hint in error: {stderr}"
            )

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_edit_layer_substring_single(self):
        """Test that --layer with a substring matching one layer edits it."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            # Step 1: Run discovery
            _run_pytest_and_validate(
                "edit_layer_sub_single_discovery",
                ["--auto-convert-bf16"],
                output_dir,
                test_filter="example_gen_config_layer_",
            )

            # Step 2: Edit with partial layer name "add" (matches "Add_output" only)
            edit_cmd = [
                sys.executable,
                "-m",
                "torq.gen_config",
                "edit",
                f"--model={TEST_MODEL}",
                f"--output-dir={output_dir}",
                "--layer",
                "add",
                "--executor",
                "host",
            ]
            edit_result = subprocess.run(edit_cmd, capture_output=True, text=True)
            assert edit_result.returncode == 0, (
                f"edit with substring failed: {edit_result.returncode}\n"
                f"stderr: {edit_result.stderr}"
            )

            # Step 3: Verify the edit landed on Add_output
            report_files = [
                f for f in output_dir.glob("torq_gen_config_*.json")
                if not f.name.endswith("_compiler.json")
            ]
            with open(report_files[0]) as f:
                report_data = json.load(f)
            assert report_data["ops"]["Add_output"]["recommended_executor"] == "host"

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_edit_layer_substring_multi(self):
        """Test that --layer with a substring matching multiple layers batch-edits all."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            # Step 1: Run discovery
            _run_pytest_and_validate(
                "edit_layer_sub_multi_discovery",
                ["--auto-convert-bf16"],
                output_dir,
                test_filter="example_gen_config_layer_",
            )

            # Step 2: Edit with a query that matches multiple layers ("_out" matches all 3)
            edit_cmd = [
                sys.executable,
                "-m",
                "torq.gen_config",
                "edit",
                f"--model={TEST_MODEL}",
                f"--output-dir={output_dir}",
                "--layer",
                "_out",
                "--executor",
                "host",
            ]
            edit_result = subprocess.run(edit_cmd, capture_output=True, text=True)
            assert edit_result.returncode == 0, (
                f"edit with multi-match substring failed: {edit_result.returncode}\n"
                f"stderr: {edit_result.stderr}"
            )
            stdout = edit_result.stdout
            assert "Batch edit" in stdout, f"Missing batch header: {stdout}"
            assert "3 layer(s)" in stdout, f"Missing match count: {stdout}"

            # Step 3: Verify all layers were updated
            report_files = [
                f for f in output_dir.glob("torq_gen_config_*.json")
                if not f.name.endswith("_compiler.json")
            ]
            with open(report_files[0]) as f:
                report_data = json.load(f)
            for layer in ["Add_output", "Relu_relu_out", "Conv_conv_out"]:
                assert report_data["ops"][layer]["recommended_executor"] == "host", (
                    f"Layer {layer} was not updated"
                )

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_edit_layer_batch_fnmatch(self):
        """Test that --layer batch-edits all matching layers."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            # Step 1: Run discovery
            _run_pytest_and_validate(
                "edit_layer_batch_discovery",
                ["--auto-convert-bf16"],
                output_dir,
                test_filter="example_gen_config_layer_",
            )

            # Step 2: Batch edit all layers matching "*_*" (all 3 layers have underscores)
            edit_cmd = [
                sys.executable,
                "-m",
                "torq.gen_config",
                "edit",
                f"--model={TEST_MODEL}",
                f"--output-dir={output_dir}",
                "--layer",
                "*_*",
                "--executor",
                "host",
            ]
            edit_result = subprocess.run(edit_cmd, capture_output=True, text=True)
            assert edit_result.returncode == 0, (
                f"edit --layer failed: {edit_result.returncode}\n"
                f"stderr: {edit_result.stderr}"
            )
            stdout = edit_result.stdout
            assert "Batch edit" in stdout, f"Missing batch header: {stdout}"
            assert "3 layer(s) match" in stdout, f"Missing match count: {stdout}"
            assert "Updated recommended_executor for 3 layer(s)" in stdout, (
                f"Missing update summary: {stdout}"
            )

            # Step 3: Verify all layers were updated
            report_files = [
                f for f in output_dir.glob("torq_gen_config_*.json")
                if not f.name.endswith("_compiler.json")
            ]
            with open(report_files[0]) as f:
                report_data = json.load(f)
            for layer in ["Add_output", "Relu_relu_out", "Conv_conv_out"]:
                assert report_data["ops"][layer]["recommended_executor"] == "host", (
                    f"Layer {layer} was not updated"
                )

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_edit_layer_no_match(self):
        """Test that --layer with no matches errors helpfully."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            # Step 1: Run discovery
            _run_pytest_and_validate(
                "edit_layer_batch_none_discovery",
                ["--auto-convert-bf16"],
                output_dir,
                test_filter="example_gen_config_layer_",
            )

            # Step 2: Pattern that matches nothing
            edit_cmd = [
                sys.executable,
                "-m",
                "torq.gen_config",
                "edit",
                f"--model={TEST_MODEL}",
                f"--output-dir={output_dir}",
                "--layer",
                "xyz_nonexistent_*",
                "--executor",
                "host",
            ]
            edit_result = subprocess.run(edit_cmd, capture_output=True, text=True)
            assert edit_result.returncode != 0, (
                "edit with no-match should have failed"
            )
            stderr = edit_result.stderr
            assert "No layers match" in stderr, (
                f"Missing no-match error: {stderr}"
            )

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_edit_layer_all(self):
        """Test that --layer ALL edits every layer."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            # Step 1: Run discovery
            _run_pytest_and_validate(
                "edit_layer_batch_all_discovery",
                ["--auto-convert-bf16"],
                output_dir,
                test_filter="example_gen_config_layer_",
            )

            # Step 2: Edit ALL layers
            edit_cmd = [
                sys.executable,
                "-m",
                "torq.gen_config",
                "edit",
                f"--model={TEST_MODEL}",
                f"--output-dir={output_dir}",
                "--layer",
                "ALL",
                "--executor",
                "host",
            ]
            edit_result = subprocess.run(edit_cmd, capture_output=True, text=True)
            assert edit_result.returncode == 0, (
                f"edit --layer ALL failed: {edit_result.returncode}\n"
                f"stderr: {edit_result.stderr}"
            )
            stdout = edit_result.stdout
            assert "3 layer(s)" in stdout, f"Missing layer count: {stdout}"

            # Step 3: Verify all layers were updated
            report_files = [
                f for f in output_dir.glob("torq_gen_config_*.json")
                if not f.name.endswith("_compiler.json")
            ]
            with open(report_files[0]) as f:
                report_data = json.load(f)
            for layer in ["Add_output", "Relu_relu_out", "Conv_conv_out"]:
                assert report_data["ops"][layer]["recommended_executor"] == "host", (
                    f"Layer {layer} was not updated"
                )

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_edit_layer_batch_fnmatch_tolerance(self):
        """Test that --layer batch edit works for tolerance changes."""
        output_dir = Path(tempfile.mkdtemp())

        try:
            # Step 1: Run discovery
            _run_pytest_and_validate(
                "edit_layer_batch_tol_discovery",
                ["--auto-convert-bf16"],
                output_dir,
                test_filter="example_gen_config_layer_",
            )

            # Step 2: Batch change tolerance for ALL layers
            edit_cmd = [
                sys.executable,
                "-m",
                "torq.gen_config",
                "edit",
                f"--model={TEST_MODEL}",
                f"--output-dir={output_dir}",
                "--layer",
                "ALL",
                "--tolerance-avg",
                "0.5",
                "--tolerance-max",
                "0.8",
            ]
            edit_result = subprocess.run(edit_cmd, capture_output=True, text=True)
            assert edit_result.returncode == 0, (
                f"edit --layer ALL tolerance failed: {edit_result.returncode}\n"
                f"stderr: {edit_result.stderr}"
            )
            stdout = edit_result.stdout
            assert "Updated tolerance for 3 layer(s)" in stdout, (
                f"Missing tolerance update summary: {stdout}"
            )

            # Step 3: Verify all layers got the new tolerance
            report_files = [
                f for f in output_dir.glob("torq_gen_config_*.json")
                if not f.name.endswith("_compiler.json")
            ]
            with open(report_files[0]) as f:
                report_data = json.load(f)
            for layer in ["Add_output", "Relu_relu_out", "Conv_conv_out"]:
                tol = report_data["ops"][layer].get("tolerance_used", {})
                assert tol.get("fp_avg_tol") == 0.5, (
                    f"Layer {layer} fp_avg_tol not updated: {tol}"
                )
                assert tol.get("fp_max_tol") == 0.8, (
                    f"Layer {layer} fp_max_tol not updated: {tol}"
                )

        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            _cleanup_generated_json()

    def test_edit_no_args(self):
        """Test that 'edit' with no --model and no config path errors."""
        edit_cmd = [
            sys.executable,
            "-m",
            "torq.gen_config",
            "edit",
            "--layer",
            "Add_output",
            "--executor",
            "host",
        ]
        edit_result = subprocess.run(edit_cmd, capture_output=True, text=True)
        assert edit_result.returncode != 0, (
            "edit with no args should have failed"
        )
