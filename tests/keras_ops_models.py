"""Keras operator test models, from the family modules in tests/testdata/keras_ops.

Each .py file there is a family module; its name is the pytest marker of its cases. The
collector imports it and reads:

- CASES: {case_stem: builder}, where builder() returns the Keras model (see
  torq.testing.keras_builder). CASES may be built with comprehensions, e.g. for sweeps.
- QUANTIZE (optional): the quantizations of every case, default ("int8", "int16"). Each one
  gives a test case named <case_stem>_<quantization>, with the int8 or int16 marker.
- CONFIG (optional): {case_stem: {key: value}}, the settings of cases that need more than the
  defaults of test_keras_ops.py:
  - quantize: the quantizations of this case; a single string ("int8") gives one test case
    named <case_stem>, with no suffix;
  - compiler_options: torq-compile flags;
  - comparison: comparison tolerances (torq.testing.comparison keys);
  - int8, int16: settings for one quantization only, merged over the others.

Cases listed in keras_known_failures.SKIP_TESTS (by test case name) are skipped.
"""

import hashlib
import importlib.util
from functools import cache

import pytest

from torq.testing.cases import Case
from torq.testing.iree import TOPDIR
from torq.testing.versioned_fixtures import (
    versioned_hashable_object_fixture,
    versioned_unhashable_object_fixture,
)

from .keras_known_failures import should_skip

KERAS_OPS_DIR = TOPDIR / "tests" / "testdata" / "keras_ops"

_QUANTIZATIONS = ("int8", "int16")
_CONFIG_KEYS = {"quantize", "compiler_options", "comparison", *_QUANTIZATIONS}


@cache
def _family_modules():
    modules = {}
    for path in sorted(KERAS_OPS_DIR.glob("*.py")):
        spec = importlib.util.spec_from_file_location(f"keras_ops_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.src_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        modules[path.stem] = module
    return modules


def _case_config(family, stem, config):
    if config.keys() - _CONFIG_KEYS:
        raise ValueError(f"keras_ops/{family}.py: unknown CONFIG keys for {stem}: "
                         f"{sorted(config.keys() - _CONFIG_KEYS)}")
    return config


def keras_ops_cases():
    """The pytest params of every keras_ops case, with their markers."""
    params, seen = [], {}
    for family, module in _family_modules().items():
        configs = getattr(module, "CONFIG", {})
        if configs.keys() - module.CASES.keys():
            raise ValueError(f"keras_ops/{family}.py: CONFIG for unknown cases "
                             f"{sorted(configs.keys() - module.CASES.keys())}")
        for stem in module.CASES:
            config = _case_config(family, stem, configs.get(stem, {}))
            quantize = config.get("quantize", getattr(module, "QUANTIZE", _QUANTIZATIONS))
            variants = ([(stem, quantize)] if isinstance(quantize, str)
                        else [(f"{stem}_{q}", q) for q in quantize])
            for name, q in variants:
                if name in seen:
                    raise ValueError(f"duplicate keras_ops case {name}: {seen[name]} and {family}")
                seen[name] = family
                settings = {k: v for k, v in config.items()
                            if k not in ("quantize", *_QUANTIZATIONS)}
                settings.update(config.get(q, {}))
                data = {"name": name, "family": family, "case": stem,
                        "quantize_to_int16": q == "int16", "config": settings,
                        "src_hash": module.src_hash}
                marks = [getattr(pytest.mark, family), getattr(pytest.mark, q)]
                skip, reason = should_skip(name)
                if skip:
                    marks.append(pytest.mark.skip(reason=reason))
                params.append(pytest.param(Case(name, data), marks=marks))
    return params


@versioned_hashable_object_fixture
def keras_ops_case_data(case_config):
    # src_hash in the payload busts the cache when a family module changes
    return case_config["_case"]


@versioned_unhashable_object_fixture
def keras_ops_model(request, keras_ops_case_data):
    data = keras_ops_case_data
    request.getfixturevalue("record_property")(
        "compiler_input", f"keras:keras_ops/{data['family']}.{data['case']}")
    return _family_modules()[data["family"]].CASES[data["case"]]()
