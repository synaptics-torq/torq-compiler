"""Collector and reference regressions that do not require the compiler."""

import sys
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from . import torch_ops_models as models
from torq.testing.numpy import _parse_uniform_bf16_constant_onnx


@pytest.fixture
def family_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "TOPDIR", tmp_path)
    monkeypatch.setattr(models, "_group_dirs", lambda: [("tests", tmp_path)])
    models._family_cases.cache_clear()
    yield tmp_path
    models._family_cases.cache_clear()


def family(path, name="example", kind="onnx"):
    path.write_text(f'KIND = "{kind}"\ndef build(): pass\nCASES = {{"{name}": build}}\nIO = {{"{name}": []}}\n')


def test_static_fixture_rejected(family_dir):
    (family_dir / "forgotten.mlir").touch()
    with pytest.raises(ValueError, match="static torch_ops"):
        models._family_cases()


def test_duplicate_stems_rejected(family_dir):
    family(family_dir / "a.py")
    family(family_dir / "b.py")
    with pytest.raises(ValueError, match="duplicate torch_ops case example"):
        models._family_cases()


def test_config_is_per_case(family_dir):
    family(family_dir / "a.py")
    with open(family_dir / "a.py", "a") as f:
        f.write('CONFIG = {"example": {"input_range": [0, 1]}}\n')
    assert models._family_cases()[0].data["config"] == {"input_range": [0, 1]}


@pytest.mark.parametrize("config", [
    '{"typo": {"input_range": [0, 1]}}',
    '{"example": {"tolerance": 1}}',
])
def test_bad_config_rejected(family_dir, config):
    family(family_dir / "a.py")
    with open(family_dir / "a.py", "a") as f:
        f.write(f"CONFIG = {config}\n")
    with pytest.raises(ValueError, match="CONFIG must map case stems"):
        models._family_cases()


def test_lookup_and_relative_cache_payload(family_dir):
    family(family_dir / "a.py", kind="torch")
    cases = models._family_cases()
    assert cases is models._family_cases()
    assert cases[0].data["path"] == "a.py"
    with pytest.raises(ValueError, match="unknown torch_ops case typo"):
        models.model_case("typo")
    with pytest.raises(ValueError, match="requires an ONNX builder"):
        models.model_case("example.mlir", family_dir.name)


def test_failed_import_is_retried(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text('raise ImportError("missing dependency")\n')
    for _ in range(2):
        with pytest.raises(ImportError, match="missing dependency"):
            models._import_module("torch_ops_broken_test", path)
        assert "torch_ops_broken_test" not in sys.modules


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("constant_node", [False, True])
def test_matmul_weight_ignores_unrelated_scalar(tmp_path, raw, constant_node):
    scalar = numpy_helper.from_array(np.array(0.5, dtype=ml_dtypes.bfloat16), "bias")
    weight = helper.make_tensor("weight", TensorProto.BFLOAT16, [2, 3], [0.25] * 6)
    if raw:
        weight = numpy_helper.from_array(np.full((2, 3), 0.25, dtype=ml_dtypes.bfloat16), "weight")
    nodes = [helper.make_node("MatMul", ["x", "weight"], ["y"])]
    initializers = [scalar]
    if constant_node:
        nodes.insert(0, helper.make_node("Constant", [], ["weight"], value=weight))
    else:
        initializers.append(weight)
    model = helper.make_model(helper.make_graph(nodes, "test", [], [], initializers))
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    assert _parse_uniform_bf16_constant_onnx(SimpleNamespace(file_path=path)) == (0.25, [2, 3])


@pytest.mark.parametrize("case", [c for c in models._family_cases() if c.data["kind"] == "onnx"],
                         ids=lambda c: c.name)
def test_onnx_builders_make_valid_models(case):
    onnx.checker.check_model(models._call_builder(case.data))


def test_direct_mlir_cache(request, monkeypatch):
    from torq.lab.model_tools.importers import onnx as importer

    calls = []
    data = {"name": "cache-regression.mlir", "src_hash": "v1", "path": "relative.py"}
    monkeypatch.setattr(models, "model_case", lambda *args: {"_case": data})
    monkeypatch.setattr(models, "_call_builder", lambda data: object())
    monkeypatch.setattr(onnx, "save", lambda model, path: path.write_bytes(b"onnx"))

    def convert(source, target):
        calls.append(target)
        target.write_text("module {}")

    monkeypatch.setattr(importer, "convert_onnx_to_mlir", convert)
    # Use a unique payload so a prior pytest run cannot satisfy the first lookup.
    data["src_hash"] = request.node.name + str(id(data))
    generate = models.torch_ops_mlir.__wrapped__(request)
    first = generate("cache-regression.mlir")
    assert generate("cache-regression.mlir") == first
    assert len(calls) == 1
    data["src_hash"] += "changed"
    assert generate("cache-regression.mlir") != first
    assert len(calls) == 2


@pytest.mark.parametrize("dtype", ["i1", "si8", "si16", "si32", "si64", "ui8", "bf16", "f16", "f32"])
def test_torch_export_input_dtypes(tmp_path, dtype):
    import inspect
    import torch
    from .test_torch_ops import torch_ops_torch_mlir_model_file

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x.clone()

    path = tmp_path / "model.mlir"
    inspect.unwrap(torch_ops_torch_mlir_model_file)(
        request=None, versioned_file=path, torch_ops_torch_module=Identity(),
        torch_ops_case_data={"io": [([2, 3], dtype)]})
    assert "func.func" in path.read_text()
