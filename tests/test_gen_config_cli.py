"""CLI tests for the standalone torq-gen-config path.

The module-level quantize tests are fast library unit tests that run by
default. The ``TestStandaloneCliWorkflows`` class runs a shared host-only
discovery through the standalone engine once per module and exercises the
``edit``/``view``/``run`` CLI subcommands against its JSON output; those are
opt-in (marker ``gen_config_standalone``, excluded by default — see
pytest.ini).

Run: pytest tests/test_gen_config_cli.py -v
Run including the CLI workflows: pytest tests/test_gen_config_cli.py -m "" -v
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType

from torq.testing.quantize_onnx import (
    _parse_quant_dtype,
    convert_qdq_to_full_integer,
    quantize_onnx_model,
)

# Path to the test model (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
TEST_MODEL = PROJECT_ROOT / "tests/testdata/onnx_models/example_gen_config.onnx"


def _tools_available():
    try:
        from torq.lab.tools import find_compile_tool, find_iree_run_tool, find_run_tool

        find_compile_tool()
        find_run_tool()
        find_iree_run_tool()
        return True
    except Exception:
        return False


TOOLS_AVAILABLE = _tools_available()
needs_tools = pytest.mark.skipif(
    not TOOLS_AVAILABLE, reason="torq-compile/torq-run-module/iree-run-module not found"
)
# The CLI workflow tests run a real discovery; opt-in (heavy) via the
# gen_config_standalone marker.
standalone = pytest.mark.gen_config_standalone


def _run_cli(*args):
    """Invoke the torq-gen-config CLI exactly as users would."""
    return subprocess.run(
        [sys.executable, "-m", "torq.gen_config", *args],
        capture_output=True,
        text=True,
    )


def _report_and_compiler_paths(output_dir: Path):
    """Locate the discovery report and compiler JSONs in an output dir."""
    report_files = [
        f
        for f in output_dir.glob("torq_gen_config_*.json")
        if not f.name.endswith(("_compiler.json", "_mac_debug.json"))
    ]
    compiler_files = list(output_dir.glob("torq_gen_config_*_compiler.json"))
    assert len(report_files) == 1, "Expected 1 report JSON"
    assert len(compiler_files) == 1, "Expected 1 compiler JSON"
    return report_files[0], compiler_files[0]


def test_edit_no_args():
    """'edit' with no --model and no config path errors."""
    result = _run_cli("edit", "--layer", "Add_output", "--executor", "host")
    assert result.returncode != 0, "edit with no args should have failed"


@needs_tools
@standalone
class TestStandaloneCliWorkflows:
    """edit/view/run CLI workflows against a shared standalone discovery.

    Discovery runs once per module (host executor only, simulator runtime)
    and each test copies the resulting JSONs into its own tmp dir before
    invoking the CLI in a subprocess.
    """

    @pytest.fixture(scope="class")
    def shared_discovery(self, tmp_path_factory):
        from torq.gen_config._options import DiscoveryConfig
        from torq.gen_config._runner import run_discovery

        work_dir = tmp_path_factory.mktemp("gen_config_cli")
        cfg = DiscoveryConfig(
            model_path=str(TEST_MODEL),
            output_dir=str(work_dir / "out"),
            skip_executors="nss,css",
            skip_mode=True,
            # BF16 like the legacy tests: an FP32 full-model compile trips a
            # known torq-compile assertion (Kernel.cpp iWidth requires bf16)
            # on the compiler-JSON-only run path.
            auto_convert_bf16=True,
            collect_timing=True,
            timing_runs=2,
            recompute_cache=True,
            cache_dir=str(work_dir / "cache"),
        )
        assert run_discovery(cfg) == 0
        return work_dir / "out"

    @pytest.fixture
    def output_dir(self, shared_discovery, tmp_path):
        """Per-test output dir pre-populated with the shared discovery JSONs."""
        for json_file in shared_discovery.glob("torq_gen_config_*.json"):
            shutil.copy(json_file, tmp_path / json_file.name)
        return tmp_path

    def test_viewer_script_output(self, output_dir):
        """python/torq/gen_config/view.py produces correct summary output."""
        report_path, _ = _report_and_compiler_paths(output_dir)

        viewer_result = subprocess.run(
            [sys.executable, "python/torq/gen_config/view.py", str(report_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        output = viewer_result.stdout

        assert "MODEL:" in output, "Viewer: Missing MODEL header"
        assert "STATUS COUNTS:" in output, "Viewer: Missing status counts"
        assert "success:" in output, "Viewer: Missing success count"
        assert "TIMING SUMMARY" in output, "Viewer: Missing timing summary"
        assert "HOST:" in output, "Viewer: Missing HOST executor"
        assert "ms" in output, "Viewer: Missing timing values"

    def test_viewer_layer_details(self, output_dir):
        """view.py <json> <layer_id> shows layer details."""
        report_path, _ = _report_and_compiler_paths(output_dir)

        viewer_result = subprocess.run(
            [
                sys.executable,
                "python/torq/gen_config/view.py",
                str(report_path),
                "Relu_relu_out",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        output = viewer_result.stdout

        assert "LAYER: Relu_relu_out" in output, "Viewer: Missing layer header"
        assert "Executor Results:" in output, "Viewer: Missing executor results"
        assert "[HOST]:" in output, "Viewer: Missing HOST result"
        assert "success" in output, "Viewer: Missing success status"
        assert "Runtime:" in output, "Viewer: Missing runtime"
        assert "Tolerance:" in output, "Viewer: Missing tolerance"

    def test_run_with_compiler_json_only(self, output_dir):
        """'run' works with only compiler JSON (no report JSON)."""
        report_path, compiler_path = _report_and_compiler_paths(output_dir)

        report_path.unlink()
        assert not report_path.exists(), "Report JSON should be deleted"

        run_result = _run_cli(
            "run",
            f"--model={TEST_MODEL}",
            f"--output-dir={output_dir}",
            "--auto-convert-bf16",
        )
        assert run_result.returncode == 0, (
            f"run with compiler JSON only failed: {run_result.returncode}\n"
            f"stdout: {run_result.stdout}\nstderr: {run_result.stderr}"
        )

        # Regression: the run must not clobber the compiler JSON it consumed
        # (the mac_debug sidecar used to be misidentified as the report JSON,
        # and regenerating from it wrote empty op_assignments).
        compiler_data = json.loads(compiler_path.read_text())
        assert len(compiler_data.get("op_assignments", {})) == 3, (
            f"compiler JSON was clobbered: {compiler_data}"
        )

    def test_edit_and_run(self, output_dir):
        """'edit' updates both JSONs and 'run' uses the edited recommendation."""
        report_path, compiler_path = _report_and_compiler_paths(output_dir)

        edit_result = _run_cli(
            "edit", str(report_path), "--layer", "Add_output", "--executor", "host"
        )
        assert edit_result.returncode == 0, (
            f"edit failed: {edit_result.returncode}\nstderr: {edit_result.stderr}"
        )

        report_data = json.loads(report_path.read_text())
        assert report_data["ops"]["Add_output"]["recommended_executor"] == "host"

        # final_report_text was regenerated
        report_text = report_data.get("final_report_text", "")
        add_line = [l for l in report_text.split("\n") if "Add_output" in l]
        assert len(add_line) == 1, "Add_output line not found in final_report_text"
        assert add_line[0].endswith(" host"), (
            f"final_report_text did not update: {add_line[0]}"
        )

        # compiler JSON was regenerated at the layer's MLIR location
        location = report_data["ops"]["Add_output"]["mlir_location"]
        compiler_data = json.loads(compiler_path.read_text())
        assignments = compiler_data.get("op_assignments", {})
        assert assignments.get(location, {}).get("executor") == "host", (
            f"compiler JSON not updated: {assignments}"
        )

        # run output reflects the edited recommendation
        run_result = _run_cli(
            "run",
            f"--model={TEST_MODEL}",
            f"--output-dir={output_dir}",
            "--auto-convert-bf16",
        )
        assert run_result.returncode == 0, (
            f"run after edit failed: {run_result.returncode}\n"
            f"stderr: {run_result.stderr}"
        )
        combined_output = run_result.stdout + run_result.stderr
        assert "Add_output" in combined_output, "Add_output not in run output"
        add_lines = [
            l
            for l in combined_output.split("\n")
            if "Add_output" in l and "host" in l
        ]
        assert len(add_lines) >= 1, "run output did not reflect edited recommendation"

    def test_edit_with_model_flag(self, output_dir):
        """'edit' works with --model + --output-dir instead of positional path."""
        edit_result = _run_cli(
            "edit",
            f"--model={TEST_MODEL}",
            f"--output-dir={output_dir}",
            "--layer",
            "Add_output",
            "--executor",
            "host",
        )
        assert edit_result.returncode == 0, (
            f"edit with --model failed: {edit_result.returncode}\n"
            f"stderr: {edit_result.stderr}"
        )

        report_path, _ = _report_and_compiler_paths(output_dir)
        report_data = json.loads(report_path.read_text())
        assert report_data["ops"]["Add_output"]["recommended_executor"] == "host"

    def test_edit_list_layers(self, output_dir):
        """'edit --list' prints the per-layer status table, with optional filter."""
        list_result = _run_cli(
            "edit", f"--model={TEST_MODEL}", f"--output-dir={output_dir}", "--list"
        )
        assert list_result.returncode == 0, (
            f"edit --list failed: {list_result.returncode}\n"
            f"stderr: {list_result.stderr}"
        )

        output = list_result.stdout
        assert "Per-Layer Status:" in output, "Missing Per-Layer Status header"
        assert "HOST" in output, "Missing HOST column header"
        assert "Recommended" in output, "Missing Recommended column"
        assert "Add_output" in output, "Missing Add_output layer"
        assert "Relu_relu_out" in output, "Missing Relu_relu_out layer"
        assert "Total:" in output, "Missing total count"

        # With a substring filter
        filter_result = _run_cli(
            "edit",
            f"--model={TEST_MODEL}",
            f"--output-dir={output_dir}",
            "--list",
            "add",
        )
        assert filter_result.returncode == 0, (
            f"edit --list add failed: {filter_result.returncode}\n"
            f"stderr: {filter_result.stderr}"
        )
        assert "Add_output" in filter_result.stdout, "Missing Add_output in filtered list"
        assert "filtered from" in filter_result.stdout, "Missing filtered count"

        # With a non-matching filter
        none_result = _run_cli(
            "edit",
            f"--model={TEST_MODEL}",
            f"--output-dir={output_dir}",
            "--list",
            "xyz_nonexistent",
        )
        assert none_result.returncode == 0, (
            f"edit --list xyz_nonexistent failed: {none_result.returncode}"
        )
        assert "No layers match" in none_result.stdout, (
            f"Missing 'No layers match' message: {none_result.stdout}"
        )

    def test_edit_compiler_json_guard(self, output_dir):
        """'edit' on a compiler JSON prints a helpful error."""
        _, compiler_path = _report_and_compiler_paths(output_dir)

        edit_result = _run_cli(
            "edit", str(compiler_path), "--layer", "Add_output", "--executor", "host"
        )
        assert edit_result.returncode != 0, "edit on compiler JSON should have failed"
        stderr = edit_result.stderr
        assert "compiler json" in stderr.lower(), (
            f"Missing 'compiler JSON' hint in error: {stderr}"
        )
        assert "report json" in stderr.lower(), (
            f"Missing 'report JSON' hint in error: {stderr}"
        )

    def test_edit_layer_substring_single(self, output_dir):
        """--layer with a substring matching one layer edits only that layer."""
        edit_result = _run_cli(
            "edit",
            f"--model={TEST_MODEL}",
            f"--output-dir={output_dir}",
            "--layer",
            "add",
            "--executor",
            "host",
        )
        assert edit_result.returncode == 0, (
            f"edit with substring failed: {edit_result.returncode}\n"
            f"stderr: {edit_result.stderr}"
        )

        report_path, _ = _report_and_compiler_paths(output_dir)
        report_data = json.loads(report_path.read_text())
        assert report_data["ops"]["Add_output"]["recommended_executor"] == "host"

    def _batch_edit(self, output_dir, *edit_args):
        edit_result = _run_cli(
            "edit",
            f"--model={TEST_MODEL}",
            f"--output-dir={output_dir}",
            *edit_args,
        )
        assert edit_result.returncode == 0, (
            f"edit failed: {edit_result.returncode}\nstderr: {edit_result.stderr}"
        )
        return edit_result.stdout

    def _assert_all_recommended_host(self, output_dir):
        report_path, _ = _report_and_compiler_paths(output_dir)
        report_data = json.loads(report_path.read_text())
        for layer in ["Add_output", "Relu_relu_out", "Conv_conv_out"]:
            assert report_data["ops"][layer]["recommended_executor"] == "host", (
                f"Layer {layer} was not updated"
            )

    def test_edit_layer_substring_multi(self, output_dir):
        """--layer with a substring matching multiple layers batch-edits all."""
        stdout = self._batch_edit(output_dir, "--layer", "_out", "--executor", "host")
        assert "Batch edit" in stdout, f"Missing batch header: {stdout}"
        assert "3 layer(s)" in stdout, f"Missing match count: {stdout}"
        self._assert_all_recommended_host(output_dir)

    def test_edit_layer_batch_fnmatch(self, output_dir):
        """--layer with an fnmatch pattern batch-edits all matching layers."""
        stdout = self._batch_edit(output_dir, "--layer", "*_*", "--executor", "host")
        assert "Batch edit" in stdout, f"Missing batch header: {stdout}"
        assert "3 layer(s) match" in stdout, f"Missing match count: {stdout}"
        assert "Updated recommended_executor for 3 layer(s)" in stdout, (
            f"Missing update summary: {stdout}"
        )
        self._assert_all_recommended_host(output_dir)

    def test_edit_layer_no_match(self, output_dir):
        """--layer with no matches errors helpfully."""
        edit_result = _run_cli(
            "edit",
            f"--model={TEST_MODEL}",
            f"--output-dir={output_dir}",
            "--layer",
            "xyz_nonexistent_*",
            "--executor",
            "host",
        )
        assert edit_result.returncode != 0, "edit with no-match should have failed"
        assert "No layers match" in edit_result.stderr, (
            f"Missing no-match error: {edit_result.stderr}"
        )

    def test_edit_layer_all(self, output_dir):
        """--layer ALL edits every layer."""
        stdout = self._batch_edit(output_dir, "--layer", "ALL", "--executor", "host")
        assert "3 layer(s)" in stdout, f"Missing layer count: {stdout}"
        self._assert_all_recommended_host(output_dir)

    def test_edit_layer_batch_tolerance(self, output_dir):
        """--layer batch edit also works for tolerance changes."""
        stdout = self._batch_edit(
            output_dir, "--layer", "ALL", "--tolerance-avg", "0.5", "--tolerance-max", "0.8"
        )
        assert "Updated tolerance for 3 layer(s)" in stdout, (
            f"Missing tolerance update summary: {stdout}"
        )

        report_path, _ = _report_and_compiler_paths(output_dir)
        report_data = json.loads(report_path.read_text())
        for layer in ["Add_output", "Relu_relu_out", "Conv_conv_out"]:
            tol = report_data["ops"][layer].get("tolerance_used", {})
            assert tol.get("fp_avg_tol") == 0.5, (
                f"Layer {layer} fp_avg_tol not updated: {tol}"
            )
            assert tol.get("fp_max_tol") == 0.8, (
                f"Layer {layer} fp_max_tol not updated: {tol}"
            )


def _load_example_model() -> onnx.ModelProto:
    """Load the example FP32 ONNX model used by the CLI tests."""
    return onnx.load(str(TEST_MODEL))


def test_quantize_onnx_model_qdq():
    """quantize_onnx_model produces a QDQ model with float I/O."""
    quantized = quantize_onnx_model(_load_example_model())

    # QDQ format keeps float I/O.
    assert quantized.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert quantized.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT

    # QuantizeLinear / DequantizeLinear nodes should be present.
    op_types = [n.op_type for n in quantized.graph.node]
    assert "QuantizeLinear" in op_types
    assert "DequantizeLinear" in op_types


def test_quantize_onnx_model_full_integer():
    """quantize_onnx_model with full_integer=True produces int8 I/O."""
    quantized = quantize_onnx_model(_load_example_model(), full_integer=True)

    # Full-integer rewrite should make I/O int8.
    assert quantized.graph.input[0].type.tensor_type.elem_type == TensorProto.INT8
    assert quantized.graph.output[0].type.tensor_type.elem_type == TensorProto.INT8


def test_convert_qdq_to_full_integer():
    """Full-integer rewrite removes I/O Q/DQ nodes and changes graph I/O to int8."""
    qdq_model = quantize_onnx_model(_load_example_model())
    converted = convert_qdq_to_full_integer(qdq_model)

    # I/O should now be int8.
    assert converted.graph.input[0].type.tensor_type.elem_type == TensorProto.INT8
    assert converted.graph.output[0].type.tensor_type.elem_type == TensorProto.INT8

    # Inner Q/DQ pairs between ops should remain.
    remaining_ops = [n.op_type for n in converted.graph.node]
    assert "QuantizeLinear" in remaining_ops
    assert "DequantizeLinear" in remaining_ops


def test_quantize_onnx_model_qoperator():
    """quantize_onnx_model with quant_format='qoperator' produces QLinearConv."""
    quantized = quantize_onnx_model(_load_example_model(), quant_format="qoperator")

    # QOperator format keeps float I/O unless full_integer is used.
    assert quantized.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert quantized.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT

    # The graph should contain a QLinearConv node.
    op_types = [n.op_type for n in quantized.graph.node]
    assert "QLinearConv" in op_types


def test_quantize_onnx_model_qoperator_full_integer():
    """qoperator + full-integer produces int8 I/O and QLinearConv."""
    quantized = quantize_onnx_model(
        _load_example_model(), quant_format="qoperator", full_integer=True
    )

    # Full-integer rewrite should make I/O int8.
    assert quantized.graph.input[0].type.tensor_type.elem_type == TensorProto.INT8
    assert quantized.graph.output[0].type.tensor_type.elem_type == TensorProto.INT8

    # The graph should contain a QLinearConv node.
    op_types = [n.op_type for n in quantized.graph.node]
    assert "QLinearConv" in op_types



def test_quantize_onnx_model_hybrid_prefers_qoperator_for_conv():
    """hybrid format picks qoperator for Conv layers (QLinearConv)."""
    quantized = quantize_onnx_model(
        _load_example_model(), quant_format="hybrid", full_integer=True
    )

    # Full-integer rewrite should make I/O int8.
    assert quantized.graph.input[0].type.tensor_type.elem_type == TensorProto.INT8
    assert quantized.graph.output[0].type.tensor_type.elem_type == TensorProto.INT8

    # The graph should contain a QLinearConv node because the example model's
    # compute op is Conv.
    op_types = [n.op_type for n in quantized.graph.node]
    assert "QLinearConv" in op_types


def test_quantize_onnx_model_hybrid_falls_back_to_qdq_for_reducemean():
    """hybrid format falls back to QDQ (or leaves fp32) for unsupported ops."""
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 1, 1])
    axes = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="axes")
    rm = helper.make_node("ReduceMean", ["input", "axes"], ["output"], keepdims=1)
    graph = helper.make_graph([rm], "rm", [input], [output], [axes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10

    quantized = quantize_onnx_model(model, quant_format="hybrid", full_integer=True)

    # ONNX Runtime does not quantize ReduceMean, so the model is unchanged and
    # QLinearReduceMean does not exist.  This verifies the hybrid picker did not
    # force qoperator for an unsupported op.
    op_types = [n.op_type for n in quantized.graph.node]
    assert "ReduceMean" in op_types
    assert "QLinearReduceMean" not in op_types
    assert quantized.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT


def test_parse_quant_dtype():
    """_parse_quant_dtype returns the expected (activation, weight) QuantType pair."""
    assert _parse_quant_dtype("a8w8") == (QuantType.QInt8, QuantType.QInt8)
    assert _parse_quant_dtype("A8W8") == (QuantType.QInt8, QuantType.QInt8)

    with pytest.raises(ValueError):
        _parse_quant_dtype("a16w8")


def test_discovery_config_from_config_and_args(tmp_path):
    """DiscoveryConfig.from_config_and_args layers config file under CLI args."""
    from argparse import Namespace
    from torq.gen_config._options import DiscoveryConfig

    # Write a config file with shared keys and gen_config section
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "model_path": str(TEST_MODEL),
        "work_dir": str(tmp_path / "work"),
        "chip": "SL5",
        "timeout": 30,
        "gen_config": {
            "skip_mode": True,
            "collect_timing": True,
        }
    }))

    # Create args with config file but no model (model comes from config)
    args = Namespace(
        config=[str(cfg_file)],
        model=None,
        output_dir=None,
        torq_hw=None,
        torq_hw_type=None,
        compiler_option=[],
        runtime_option=[],
        auto_convert_bf16=False,
        auto_convert_int32=False,
        log_file=None,
        verbose=False,
        skip_mode=False,
        skip_executors=None,
        skip_ops=None,
        save_bf16_model=None,
        subgraph_from=None,
        subgraph_to=None,
        collect_timing=False,
        timing_runs=1,
        recommend_by_timing=False,
        dedup_layers=False,
        quantize=False,
        per_channel=False,
        full_integer=False,
        quant_format="qdq",
        quant_dtype="A8W8",
        debug_ir=None,
        trace_buffers=False,
        debug_torq_compiler=0,
        compiler_timeout=300,
        runtime_timeout=240,
        ignore_binary_mtime=False,
        convert_io_dtypes=None,
        recompute_cache=False,
        debug_cache=False,
        cache_dir=None,
    )

    cfg = DiscoveryConfig.from_config_and_args(args)

    # Config file values should be present
    assert cfg.model_path == str(TEST_MODEL)
    assert cfg.output_dir == str(tmp_path / "work")
    assert cfg.torq_hw == "SL5"
    assert cfg.compiler_timeout == 30
    assert cfg.runtime_timeout == 30

    # Gen_config section values should be present
    assert cfg.skip_mode is True
    assert cfg.collect_timing is True


def test_discovery_config_cli_overrides_config_file(tmp_path):
    """CLI arguments override config file values."""
    from argparse import Namespace
    from torq.gen_config._options import DiscoveryConfig

    # Write a config file
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "model_path": str(TEST_MODEL),
        "chip": "SL5",
    }))

    # Create args with config file and explicit CLI values
    args = Namespace(
        config=[str(cfg_file)],
        model=str(TEST_MODEL),
        output_dir=str(tmp_path / "cli_out"),
        torq_hw="SL2610",  # Override the config file
        torq_hw_type=None,
        compiler_option=[],
        runtime_option=[],
        auto_convert_bf16=False,
        auto_convert_int32=False,
        log_file=None,
        verbose=False,
        skip_mode=False,
        skip_executors=None,
        skip_ops=None,
        save_bf16_model=None,
        subgraph_from=None,
        subgraph_to=None,
        collect_timing=False,
        timing_runs=1,
        recommend_by_timing=False,
        dedup_layers=False,
        quantize=False,
        per_channel=False,
        full_integer=False,
        quant_format="qdq",
        quant_dtype="A8W8",
        debug_ir=None,
        trace_buffers=False,
        debug_torq_compiler=0,
        compiler_timeout=300,
        runtime_timeout=240,
        ignore_binary_mtime=False,
        convert_io_dtypes=None,
        recompute_cache=False,
        debug_cache=False,
        cache_dir=None,
    )

    cfg = DiscoveryConfig.from_config_and_args(args)

    # CLI values should override config file
    assert cfg.torq_hw == "SL2610"  # CLI value wins
    assert cfg.output_dir == str(tmp_path / "cli_out")  # CLI value wins
    assert cfg.model_path == str(TEST_MODEL)  # Same in both


def test_discovery_config_remote_mapping(tmp_path):
    """Remote section from config file is mapped to torq_* fields."""
    from argparse import Namespace
    from torq.gen_config._options import DiscoveryConfig

    # Write a config file with remote section
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "model_path": str(TEST_MODEL),
        "remote": {
            "address": "root@192.168.1.1",
            "port": 2222,
            "private_key": "/path/to/key",
        }
    }))

    # Create args
    args = Namespace(
        config=[str(cfg_file)],
        model=None,
        output_dir=None,
        torq_hw=None,
        torq_hw_type=None,
        compiler_option=[],
        runtime_option=[],
        auto_convert_bf16=False,
        auto_convert_int32=False,
        log_file=None,
        verbose=False,
        skip_mode=False,
        skip_executors=None,
        skip_ops=None,
        save_bf16_model=None,
        subgraph_from=None,
        subgraph_to=None,
        collect_timing=False,
        timing_runs=1,
        recommend_by_timing=False,
        dedup_layers=False,
        quantize=False,
        per_channel=False,
        full_integer=False,
        quant_format="qdq",
        quant_dtype="A8W8",
        debug_ir=None,
        trace_buffers=False,
        debug_torq_compiler=0,
        compiler_timeout=300,
        runtime_timeout=240,
        ignore_binary_mtime=False,
        convert_io_dtypes=None,
        recompute_cache=False,
        debug_cache=False,
        cache_dir=None,
    )

    cfg = DiscoveryConfig.from_config_and_args(args)

    # Remote values should be mapped
    assert cfg.torq_addr == "root@192.168.1.1"
    assert cfg.torq_port == 2222
    assert cfg.torq_private_key == "/path/to/key"
