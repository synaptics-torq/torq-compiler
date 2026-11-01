# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json

import pytest

from typing import List, Dict

from torq.lab import Case, get_test_cases_from_files

__all__ = ["Case", "get_test_cases_from_files", "get_test_cases_from_tf_model",
           "drop_duplicate_cases", "keras_layer_key"]

"""

This module provides an helper fixture and class that allows to setup
non-exaustive tests for sets of parametrized fixtures.

Assume you have a fixture A that has 10 different parameterizations
and a fixture B that has 10 different parameterizations:

E.g.:

    @pytest.fixture(params=[1,2,3,4,5,6,7,8,9,10])
    def a_param(request):
        return request.param

    @pytest.fixture(params=[11,12,13,14,15,16,17,18,19,20])
    def b_param(request):
        return request.param

Pytest will generate 100 test cases (10x10).

However, you may want to only test a subset of these combinations. You can
use the Case class to define the combinations you want to test.

E.g.:

    @pytest.fixture(params=[
        Case("case1", {"a": 1, "b": 2}),
        Case("case2", {"a": 3, "b": 4}),
    ])
    def case_config(request):
        return request.param.data

    @pytest.fixture
    def a_param(case_config):
        return case_config['a']

    @pytest.fixture
    def b_param(case_config):
        return case_config['b']

Pytest will now only generate the test cases defined in the Case instances.

The plain ``Case`` dataclass and ``get_test_cases_from_files`` live in
``torq.lab`` and are re-exported here; this module adds
the hook and fixtures.

"""


def pytest_make_parametrize_id(config, val, argname):

    # if the parameter is an instance of a Case, use its name as id
    if isinstance(val, Case):
        return val.name

    # otherwise, let another plugin to handle it
    return None


@pytest.fixture
def case_config(request) -> Dict:
    """
    Default configuration in case a test is not parametetrized, this fixture is typically
    overridden in the test module to provide specific test cases.

    It is required to make sure that fixtures depending on the case_config always have a value.
    """

    return {}


def get_test_cases_from_tf_model(model, model_name, full_model=False) -> List[Case]:
    """
    Generates layer test cases from a tensorflow model, each layer becomes a test case.

    These test cases expect that the case_config fixture is overriden in the test module
    to provide the actual test configuration (by using the layer name in the model).
    """
    cases = []

    if full_model:
        full_model_config = model.get_config()
        cases.append(Case(f"full_model_{model_name}", full_model_config))
    else:
        from torq.testing.tensorflow import generate_layers_from_model
        layers = generate_layers_from_model(model)

        for layer_name, layer_config in layers.items():
            cases.append(Case(f"layer_{model_name}_" + layer_name, layer_config))

    return cases


def keras_layer_key(case):
    """Key of a case of get_test_cases_from_tf_model for drop_duplicate_cases: the configuration
    of its one-layer model (layer name normalized); None for a full model."""
    if case.name.startswith("full_model_"):
        return None
    return json.dumps(case.data, sort_keys=True)


def drop_duplicate_cases(cases, key):
    """The cases without those whose key an earlier case already has.

    Layer cases generated from several models often test the same layer twice (a block
    repeated in a model, or the same layer in two models). key(case) returns a hashable that is
    equal for two cases with the same definition, or None to always keep the case (e.g. a full
    model).
    """
    seen = set()
    kept = []
    for case in cases:
        k = key(case)
        if k is not None:
            if k in seen:
                continue
            seen.add(k)
        kept.append(case)
    return kept
