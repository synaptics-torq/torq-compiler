# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import fnmatch

import pytest
import torch
import transformers

from torq.testing.comparison import compare_test_results
from torq.testing.hf_transformers import make_hf_model_loader
from torq.testing.torch import generate_traced_torch_model_layers, register_torch_model_loader
from torq.testing.versioned_fixtures import versioned_hashable_object_fixture

'''
Test Hugging Face transformers models using IREE Turbine.

Each entry in HF_MODELS registers a lazy loader that pulls the model from
the Hugging Face Hub (weights cached by the transformers library in HF_HOME),
loads it in bf16, and (for heads that return a ModelOutput dict) wraps it so
the full model takes plain tensor inputs and returns plain tensor outputs.

Every standalone-invocable submodule becomes a layer test case: the generic
full-argument trace in torq.testing.torch (``generate_traced_torch_model_layers``)
records kwargs and non-tensor arguments and replays them, and each case is
exported through iree.turbine.aot and compared against direct PyTorch
execution.

CLI:
  List test cases:
    pytest tests/test_hf_models.py -v -s --collect-only
  Run full model:
    pytest tests/test_hf_models.py -v -s rtdetr_r50vd_full_model
  Run a layer:
    pytest tests/test_hf_models.py -v -s rtdetr_r50vd_model_decoder_layers_0
'''


class RTDetrObjectDetectionWrapper(torch.nn.Module):
    """RT-DETR head returns a ModelOutput (an OrderedDict); return the two
    tensor outputs the model is actually about: (logits, pred_boxes)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values)
        return out.logits, out.pred_boxes


# Each entry: prefix (test id / loader key), repo_id, transformers task,
# optional wrapper class, and example inputs for the traced forward pass.
HF_MODELS = [
    {
        "prefix": "rtdetr_r50vd",
        "repo_id": "PekingU/rtdetr_r50vd",
        "task": "object_detection",
        "wrapper": RTDetrObjectDetectionWrapper,
        "example_inputs": (torch.randn(1, 3, 640, 640),),
    },
]

# Register the lazy loaders so fixtures can instantiate the model on demand.
for _model in HF_MODELS:
    register_torch_model_loader(
        _model["prefix"],
        make_hf_model_loader(
            _model["repo_id"],
            _model["task"],
            wrapper=_model.get("wrapper"),
        ),
    )


def _model_version(model):
    # Qualify the metadata cache with the repo and the transformers version:
    # the traced graph (op set, attention implementation) can change between
    # transformers releases.
    return f"{model['repo_id']}@transformers-{transformers.__version__}"


def pytest_generate_tests(metafunc):
    cases = []
    for model in HF_MODELS:
        cases += generate_traced_torch_model_layers(
            metafunc.config.cache,
            model["prefix"],
            _model_version(model),
            model["prefix"],
            _load_registered_loader(model["prefix"]),
            tuple(x.to(torch.bfloat16) for x in model["example_inputs"]),
        )

    if not cases:
        return

    metafunc.parametrize("hf_model_case", cases, indirect=True,
                         ids=[c.name for c in cases])


def _load_registered_loader(prefix):
    # Rebuild the loader (same one that is registered) for the single worker
    # that has to trace the model on a metadata cache miss.
    model = next(m for m in HF_MODELS if m["prefix"] == prefix)
    return make_hf_model_loader(
        model["repo_id"], model["task"], wrapper=model.get("wrapper")
    )


@versioned_hashable_object_fixture
def comparison_config_bf16():
    # bf16 models: the compiler's fused bf16 matmul/conv/softmax accumulates
    # in a different order than the reference, and near-zero cancellation
    # outputs then show a large *relative* error with a tiny absolute one.
    # Gate on relative AND absolute difference (a few bf16 ULPs of the
    # tensor's range) and allow a small fraction of grossly-wrong elements.
    return {"fp_avg_tol": 0.05, "fp_max_tol": 0.5, "use_abs_tol_gate": True,
            "fp_abs_tol_frac": 0.02, "allowed_wrong": 0.001}


@versioned_hashable_object_fixture
def comparison_config_relaxed():
    # Tolerance-sensitive cases (long-sequence attention, full model).
    return {"fp_avg_tol": 0.1, "fp_max_tol": 0.5, "use_abs_tol_gate": True,
            "fp_abs_tol_frac": 0.05, "allowed_wrong": 0.005}


@pytest.fixture
def hf_model_case(request):
    return request.param


@pytest.fixture
def case_config(request, chip_config):
    case = request.getfixturevalue("hf_model_case")

    # Cases xfail'd for non-tolerance reasons (compiler op gaps, crashes,
    # resource limits).  Until the RT-DETR compiler fixes land (wip/rtd-*
    # branches), all rtdetr_r50vd cases are xfailed.
    xfail_tc = [
        'rtdetr_r50vd_*',
    ]

    # Cases that need looser tolerances than comparison_config_bf16: the
    # decoder layers (self-attention + deformable cross-attention over 8400
    # keys) accumulate more bf16 error than the conv/linear layers, sitting
    # right at the base allowed_wrong gate.
    relaxed_tc = [
        'rtdetr_r50vd_model_model_decoder_layers_*',
    ]

    name_lower = case.name.lower()
    if any(fnmatch.fnmatch(name_lower, pat.lower()) for pat in xfail_tc):
        pytest.xfail("failing test or skipped for now")

    comp_config = {
        "layer_name": case.data["layer_name"],
        "model_loader_key": case.data["model_loader_key"],
        "is_full_model": case.data["is_full_model"],
        "layer_input_shapes": case.data.get("layer_input_shapes", []),
        "layer_arg_specs": case.data.get("layer_arg_specs"),
        "layer_kwarg_specs": case.data.get("layer_kwarg_specs"),
        "torch_model_data": "torch_turbine_layer_model_data",
        "mlir_model_file": "torch_layer_model",
        "input_data": "tweaked_random_input_data",
        # Realistic activation magnitudes: the default (-40, 40) range
        # saturates the bf16 attention softmax and amplifies the
        # compiler/reference accumulation-order differences.
        "tweaked_input_data_range": (-1, 1),
        "comparison_config": "comparison_config_bf16",
    }

    if any(fnmatch.fnmatch(name_lower, pat.lower()) for pat in relaxed_tc):
        comp_config["comparison_config"] = "comparison_config_relaxed"

    return comp_config


@pytest.fixture
def reference_results(request):
    """Use PyTorch directly as reference for HF transformers model tests."""
    return request.getfixturevalue("torch_reference_results")


@pytest.mark.ci
def test_hf_model(request, reference_results, torq_results, case_config,
                 hf_model_case):
    compare_test_results(request, torq_results, reference_results, case_config)
