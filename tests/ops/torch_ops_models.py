"""Shared Torch operator builders for numeric and compiler regression tests.

Every .py file in a torch_ops group directory is a family module. The collector reads its
module-level literals without importing it:

- KIND: "onnx" (builders return an onnx.ModelProto) or "torch" (builders return an nn.Module).
- CASES: {case_stem: builder}, where builder is a function of the module or a
  functools.partial of one. Stems starting with "disable" are skipped.
- IO (torch only): {case_stem: [[shape, dtype], ...]}, the example inputs of torch.export.
- CONFIG (optional): {case_stem: {key: value}}, the settings of cases that need more than the
  defaults of test_torch_ops.py:
  - compiler_options: torq-compile flags, replacing the group default;
  - comparison: comparison tolerances (torq.testing.comparison keys);
  - reference: the name of the reference-results fixture;
  - input_range: [low, high] of the random inputs;
  - sim_runtime_timeout: runtime timeout in seconds on the simulator.
"""

import ast
import hashlib
import importlib.util
import sys
from functools import cache
from pathlib import Path

from filelock import FileLock

import pytest

from torq.lab import Case
from torq.testing.iree import TOPDIR
from torq.testing.versioned_fixtures import (
    VersionedFile,
    versioned_hashable_object_fixture,
    versioned_unhashable_object_fixture,
)


def _group_dirs():
    """(root_name, dir) pairs from the torch_ops.group files, in group order."""
    dirs = []
    for root_name, root in (("tests", TOPDIR / "tests"), ("extras", TOPDIR / "extras" / "tests")):
        group_file = root / "testdata" / "torch_ops.group"
        if not group_file.exists():
            continue
        for group in (x.strip() for x in group_file.read_text().splitlines()):
            d = root / "testdata" / group
            if d.is_dir():
                dirs.append((root_name, d))
    return dirs


_CONFIG_KEYS = {"compiler_options", "comparison", "reference", "input_range", "sim_runtime_timeout"}


def _parse_family_module(path: Path):
    """Lightweight AST scan of a builder module (no import at collection time).

    Returns (kind, {stem: builder_funcname}, {stem: io_spec}, {stem: config}) or None.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        raise ValueError(f"bad builder module {path}: {e}") from e
    metadata = {
        node.targets[0].id: node.value
        for node in tree.body
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name))
    }
    if "KIND" not in metadata or "CASES" not in metadata:
        return None
    try:
        kind = ast.literal_eval(metadata["KIND"])
        if kind not in ("onnx", "torch"):
            raise ValueError("KIND must be 'onnx' or 'torch'")
        case_dict = metadata["CASES"]
        if not isinstance(case_dict, ast.Dict) or not case_dict.keys:
            raise ValueError("CASES must be a nonempty dictionary")
        cases = {}
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        for key, value in zip(case_dict.keys, case_dict.values):
            stem = ast.literal_eval(key)
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "partial" and value.args):
                value = value.args[0]
            if (not isinstance(stem, str) or not stem or stem in cases
                    or not isinstance(value, ast.Name) or value.id not in functions):
                raise ValueError("CASES must map unique case stems to builder functions")
            cases[stem] = value.id
        ios = ast.literal_eval(metadata["IO"]) if "IO" in metadata else {}
        if not isinstance(ios, dict) or (kind == "torch" and cases.keys() - ios.keys()):
            raise ValueError("torch builders require IO for every case")
        configs = ast.literal_eval(metadata["CONFIG"]) if "CONFIG" in metadata else {}
        if (not isinstance(configs, dict) or configs.keys() - cases.keys()
                or any(not isinstance(c, dict) or c.keys() - _CONFIG_KEYS
                       for c in configs.values())):
            raise ValueError("CONFIG must map case stems to dicts with keys in "
                             + ", ".join(sorted(_CONFIG_KEYS)))
    except (ValueError, TypeError) as e:
        raise ValueError(f"bad builder metadata in {path}: {e}") from e
    return kind, cases, ios, configs


def _import_module(module_id: str, path: Path):
    if module_id not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_id, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_id] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[module_id]
            raise
    return sys.modules[module_id]


def _call_builder(data):
    mod = _import_module(data["builder"].partition(".")[0], TOPDIR / data["path"])
    return mod.CASES[data["name"].removesuffix(".mlir")]()


@cache
def _family_cases():
    cases = []
    seen = {}
    for root_name, d in _group_dirs():
        static = sorted(d.glob("*.mlir"))
        if static:
            raise ValueError(f"static torch_ops fixtures must be converted to builders: {static}")
        for f in sorted(d.glob("*.py")):
            # every .py in a group dir is a family module by convention; a
            # non-conforming one must fail loudly
            parsed = _parse_family_module(f)
            if parsed is None:
                raise ValueError(
                    f"{f} is not a valid family module: expected module-level "
                    "KIND = 'onnx'|'torch' and CASES = {case_stem: builder_fn}")
            kind, builders, ios, configs = parsed
            src_hash = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            mod_id = f"torch_ops_gen_{root_name}_{d.name}_{f.stem}"
            for stem, func in sorted(builders.items()):
                if stem.startswith("disable"):
                    continue
                if stem in seen:
                    raise ValueError(f"duplicate torch_ops case {stem}: {seen[stem]} and {f}")
                seen[stem] = f
                cases.append(Case(f"{stem}.mlir", {
                    "name": f"{stem}.mlir",
                    "root": root_name,
                    "group": d.name,
                    "kind": kind,
                    "path": str(f.relative_to(TOPDIR)),
                    "builder": f"{mod_id}.{func}",
                    "src_hash": src_hash,
                    "io": ios.get(stem),
                    "config": configs.get(stem, {}),
                }))
    return cases


@versioned_hashable_object_fixture
def torch_ops_case_data(case_config):
    # src_hash in the payload busts the cache when a family module changes
    return case_config["_case"]


@versioned_unhashable_object_fixture
def torch_ops_onnx_model(request, torch_ops_case_data):
    # versioned inputs arrive auto-unwrapped inside versioned generators
    data = torch_ops_case_data
    request.getfixturevalue("record_property")("compiler_input", f"onnx:{data['builder']}")
    return _call_builder(data)


def model_case(name, group="torch_ops"):
    """Use the same generated ONNX input in another fixture-based harness."""
    data = next((case.data for case in _family_cases()
                if case.name == name and case.data["root"] == "tests"
                and case.data["group"] == group), None)
    if data is None:
        raise ValueError(f"unknown torch_ops case {name} in {group}")
    if data["kind"] != "onnx":
        raise ValueError(f"torch_ops case {name} requires an ONNX builder")
    return {"_case": data, "onnx_model": "torch_ops_onnx_model",
            "mlir_model_file": "onnx_mlir_model_file"}


@pytest.fixture
def torch_ops_mlir(request):
    """Materialize an operator builder for tests that invoke the compiler directly."""
    import onnx
    from torq.lab.model_tools.importers.onnx import convert_onnx_to_mlir

    def generate(name, group="torch_ops"):
        data = model_case(name, group)["_case"]
        artifact = VersionedFile.build(request, "torch_ops_mlir", "mlir", data)
        with FileLock(str(artifact.file_path) + ".lock"):
            if not artifact.valid(request):
                onnx_path = artifact.file_path.with_suffix(".onnx")
                onnx.save(_call_builder(data), onnx_path)
                convert_onnx_to_mlir(onnx_path, artifact.file_path)
        return artifact.file_path

    return generate
