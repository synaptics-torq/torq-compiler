# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""TFLite intermediate tensor export.

Transforms a TFLite model so that specific intermediate tensors are exposed
as subgraph outputs.
"""

import copy
from pathlib import Path
from typing import List

import flatbuffers
import numpy as np

from .layers import TFLiteModelParser

class TFLiteTensorOutputExporter:
    """
    Transforms a TFLite model so that a specific tensor is exposed
    as a subgraph output.

    Usage::

        exporter = TFLiteTensorOutputExporter("model.tflite")
        exporter.get_output_tensor_names() # List all tensors with indices and names
        out_path = exporter.export("model_tensor_output.tflite", tensor_index=42)
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self._parser = TFLiteModelParser(model_path)
        self._model_obj = self._parser.get_model_object()

    def export(self, output_path: str, tensor_indexes: List[int]) -> Path:
        """
        Write a new TFLite model where a specific tensor is exposed as a subgraph output.

        The original model is not modified.  Existing subgraph outputs are
        kept first so the output ordering is stable and backward-compatible.

        Args:
            output_path: Destination path for the new ``.tflite`` file.
            tensor_indexes: List of indices of the tensors to expose as subgraph outputs.


        Returns:
            The resolved :class:`Path` of the written model.
        """
        new_model = copy.deepcopy(self._model_obj)

        if len(new_model.subgraphs) != 1:
            raise ValueError("Only single-subgraph models are supported")

        subgraph = new_model.subgraphs[0]
                
        subgraph.outputs = np.array(tensor_indexes, dtype=np.int32)            

        # Serialize.
        builder = flatbuffers.Builder(1024 * 1024)
        packed = new_model.Pack(builder)
        builder.Finish(packed, b"TFL3")
        buf = builder.Output()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            f.write(bytes(buf))

        return out
