"""Torch-op fixture tests.

Python builders define the models. The harness imports their ONNX or PyTorch
graphs to MLIR at test time, using `import_onnx --data-prop` for ONNX.

CLI examples:
  pytest tests/ops/test_torch_ops.py -v -s --collect-only
  pytest tests/ops/test_torch_ops.py -v -s add-25x511-bf16.mlir
"""

import pytest

from .torch_ops_models import (
    _call_builder,
    _family_cases,
    torch_ops_case_data,
    torch_ops_onnx_model,
)

from torq.testing.comparison import compare_test_results
from torq.testing.versioned_fixtures import (
    versioned_generated_file_fixture,
    versioned_unhashable_object_fixture,
)

# IO dtype names of the torch builders, and the torch dtypes they stand for.
_TORCH_DTYPES = {
    "f32": "float32",
    "bf16": "bfloat16",
    "f16": "float16",
    "i1": "bool",
    "si8": "int8",
    "si16": "int16",
    "si32": "int32",
    "si64": "int64",
    "ui8": "uint8",
}


def pytest_generate_tests(metafunc):
    if "case_config" not in metafunc.fixturenames:
        return
    cases = sorted(_family_cases(), key=lambda c: (c.data["root"], c.data["group"], c.data["name"]))
    metafunc.parametrize("case_config", cases, indirect=True, ids=[c.name for c in cases])


@versioned_unhashable_object_fixture
def torch_ops_torch_module(request, torch_ops_case_data):
    data = torch_ops_case_data
    request.getfixturevalue("record_property")("compiler_input", f"torch:{data['builder']}")
    return _call_builder(data)


@versioned_generated_file_fixture("mlir")
def torch_ops_torch_mlir_model_file(request, versioned_file, torch_ops_torch_module, torch_ops_case_data):
    """Export a torch module via torch.export + FxImporter to Torch MLIR text."""
    import torch
    from iree.compiler import ir
    from iree.compiler.extras.fx_importer import FxImporter, FxImporterHooks

    io = torch_ops_case_data["io"] or []
    example = tuple(torch.zeros(dims, dtype=getattr(torch, _TORCH_DTYPES[dt])) for dims, dt in io)

    class _Hooks(FxImporterHooks):
        def prepare_module(self, module_op):
            module_op.attributes["torch.debug_module_name"] = ir.StringAttr.get(
                type(torch_ops_torch_module).__name__, context=module_op.context)

    exported = torch.export.export(torch_ops_torch_module, example)
    ctx = ir.Context()
    importer = FxImporter(context=ctx, hooks=_Hooks())
    importer.import_frozen_program(exported)
    with open(versioned_file, "w") as f:
        f.write(str(importer.module))


@pytest.fixture
def torch_ops_comparison_config(torch_ops_case_data):
    return torch_ops_case_data.data["config"].get("comparison", {})


@pytest.fixture
def case_config(request, runtime_hw_type, chip_config):
    data = request.param.data
    config = data["config"]

    # In-tree torch_ops cases run on the NSS/slice path only, so they fail loudly if a change
    # routes them to the host/CSS fallback. torch_ops_host_css/ and extras cases keep it.
    if data["group"] == "torch_ops" and data["root"] == "tests":
        compiler_options = ["--torq-disable-host", "--torq-disable-css"]
    else:
        compiler_options = []

    cfg = {
        "_case": data,
        "input_data": "tweaked_random_input_data",
        "torq_compiler_options": list(config.get("compiler_options", compiler_options)),
        "comparison_config": "torch_ops_comparison_config",
    }
    if "input_range" in config:
        cfg["tweaked_input_data_range"] = tuple(config["input_range"])
    if runtime_hw_type.data == "sim" and "sim_runtime_timeout" in config:
        cfg["torq_runtime_timeout"] = config["sim_runtime_timeout"]

    if data["kind"] == "onnx":
        cfg.update({
            "onnx_model": "torch_ops_onnx_model",
            "mlir_model_file": "onnx_mlir_model_file",
        })
    else:  # torch
        cfg.update({
            "mlir_model_file": "torch_ops_torch_mlir_model_file",
        })
    return cfg


@pytest.fixture
def reference_results(request, case_config):
    reference = case_config["_case"]["config"].get("reference")
    if reference:
        return request.getfixturevalue(reference)
    try:
        return request.getfixturevalue("llvmcpu_reference_results")
    except Exception:
        return request.getfixturevalue("torch_reference_results")


@pytest.mark.ci
@pytest.mark.fpga_ci
def test_mlir_files(request, torq_results, reference_results, case_config):
    compare_test_results(request, torq_results, reference_results, case_config)
