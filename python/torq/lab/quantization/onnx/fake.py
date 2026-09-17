# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ONNX fake-quantization helpers.

Pure ONNX-model transforms. Requires the ``onnx`` extra.
"""

import onnx
from onnx import numpy_helper, TensorProto

from torq.lab.pipeline.io import ConvertIODTypesPolicy, cast_round_trip


def onnx_fake_quantize(model: onnx.ModelProto) -> onnx.ModelProto:
    """Round FP32/INT64/UINT64 initializers through BF16/INT32/UINT32.

    Fake-quantizes weights so the ONNX reference path matches the precision of
    a model compiled with ``--torq-convert-dtypes --torq-convert-io-dtype``.
    """
    for init in model.graph.initializer:
        if init.data_type not in (TensorProto.FLOAT, TensorProto.INT64, TensorProto.UINT64):
            continue
        data = numpy_helper.to_array(init).copy()
        down_dtype = ConvertIODTypesPolicy.convert_io_dtype(data.dtype)
        new_data = cast_round_trip(data, down_dtype, data.dtype)
        new_init = numpy_helper.from_array(new_data, init.name)
        init.CopyFrom(new_init)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    return model
