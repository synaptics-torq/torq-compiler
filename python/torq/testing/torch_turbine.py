# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IREE Turbine-based export helpers for the Torq torch model test path.

This module provides fixtures that export PyTorch layers/models to Torch MLIR
using ``iree.turbine.aot`` instead of the raw ``FxImporter`` path used by
``torq.testing.torch``.  It reuses the layer-extraction and reference-execution
machinery from ``torq.testing.torch``.
"""

import contextlib

import torch
import pytest

from .torch import (
    TorchLayerCase,
    _find_layer_by_name,
    _get_model_from_case_config,
)
from .versioned_fixtures import VersionedUncachedData

try:
    import iree.turbine.aot as _aot
    _HAS_TURBINE = True
except ImportError as _turbine_import_err:
    _aot = None
    _HAS_TURBINE = False
    _TURBINE_IMPORT_ERR = _turbine_import_err


@contextlib.contextmanager
def _bfloat16_numpy_workaround():
    """Temporarily allow np.array() on bfloat16 tensors.

    IREE Turbine's AOT exporter calls ``np.array(detached_tensor)`` when
    lifting constants to ``util.global`` ops.  NumPy does not support
    bfloat16, so for bf16 tensors we return a uint16 view of the same raw
    bits.  The exported MLIR type stays bf16, and the byte buffer is
    interpreted correctly by ``DenseResourceElementsAttr.get_from_buffer``.
    """
    original_array = torch.Tensor.__array__

    def _patched_array(self, *args, **kwargs):
        if self.dtype == torch.bfloat16:
            t = self.view(torch.uint16)
            if kwargs.get("copy"):
                t = t.contiguous().clone()
            return t.numpy()
        return original_array(self, *args, **kwargs)

    torch.Tensor.__array__ = _patched_array
    try:
        yield
    finally:
        torch.Tensor.__array__ = original_array


def export_submodule_with_turbine(submodule, example_inputs):
    """Export a torch.nn.Module to Torch MLIR using IREE Turbine.

    Args:
        submodule: The torch.nn.Module to export.
        example_inputs: A tuple of torch.Tensor example inputs.

    Returns:
        The MLIR module text as a string.
    """
    if not _HAS_TURBINE:
        raise ImportError(
            "iree-turbine is required for torch_turbine export. "
            "Install it with: pip install iree-turbine"
        ) from _TURBINE_IMPORT_ERR

    with _bfloat16_numpy_workaround():
        export_output = _aot.export(submodule, args=example_inputs)
    return str(export_output.mlir_module)


@pytest.fixture
def torch_turbine_layer_model_data(request, case_config):
    """Fixture that exports a single layer from a Torch model using IREE Turbine.

    Returns a VersionedUncachedData containing the Torch MLIR text.
    """
    layer_name = case_config["layer_name"]
    model = _get_model_from_case_config(case_config)
    layer_input_shapes = case_config.get("layer_input_shapes", [])

    submodule = _find_layer_by_name(model, layer_name) if layer_name else model
    if submodule is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")

    target_dtype = torch.bfloat16
    for t in list(model.parameters()) + list(model.buffers()):
        if t.is_floating_point():
            target_dtype = t.dtype
            break

    # Build random example inputs from recorded shapes using the model's dtype.
    example_inputs = []
    for shape in layer_input_shapes:
        if shape is None:
            continue
        example_inputs.append(torch.randn(*shape, dtype=target_dtype))

    if not example_inputs:
        raise ValueError(f"No input shapes available for layer '{layer_name}'")

    mlir_text = export_submodule_with_turbine(submodule, tuple(example_inputs))
    version = f"torch_turbine_layer_{layer_name}_{type(model).__name__}"
    return VersionedUncachedData(data=mlir_text, version=version)
