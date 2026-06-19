"""Parametric convolution sweep: geometry sampler + per-backend model builders.

Single model-source module for the opt-in conv sweep (tests/test_conv_sweep.py).

- Sampler: `_sample` draws one conv geometry from per-axis ranges (seeded,
  deterministic); the 5-category config sweeps square/non-square kernels and
  strides for Conv2D and two stride regimes for Conv1D. `conv_sweep_params()`
  returns a list of ConvModelParams.
- ONNX builder: `conv_onnx_model` builds a torch conv in memory and exports an
  onnx.ModelProto; the existing onnx_model -> onnx_model_file -> onnx_mlir_model_file
  chain + composite_reference_results oracle take it from there.
- TFLite flavor converter: `conv_sweep_tflite_model_file` produces the f32 /
  w8if32 (dynamic-range) / w8i8 flavors -- the shared quantize_model has no
  dynamic-range path, so the sweep needs its own converter.

Everything is built in memory at collection time; nothing here touches the repo.
"""

import io
import random

import numpy as np
import onnx
import tensorflow as tf
import torch
import torch.nn as nn

from torq.testing.onnx import convert_fp32_to_bf16, onnx_model_fixture
from torq.testing.versioned_fixtures import (
    versioned_generated_file_fixture,
    versioned_hashable_object_fixture,
)

# conv_model builds the keras model; keras_model_params feeds both the keras and
# (reused for) the onnx builder. Re-imported so the test module can pull all
# model fixtures from this one module.
from .keras_models import ConvModelParams, conv_model, keras_model_params  # noqa: F401

# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def _sample(rng, k, s, c_in, c_out, spatial, rank, padding, square=False,
            square_stride=False, po2_channels=False) -> dict:
    """Return one pure-conv geometry dict, e.g.
    {"rank":2,"c_in":3,"c_out":16,"spatial":(56,56),"kernel":(3,3),
     "stride":(1,1),"padding":"valid"}.  For 1D, kernel/stride/spatial are 1-tuples.

    Each axis is drawn independently from its [min,max] range. The kernel is
    clamped to <= spatial on each axis (a kernel larger than the input is invalid
    for "valid" padding). square=True (rank 2): square kernel + isotropic stride.
    square_stride=True (rank 2): isotropic stride only, kernel stays per-axis.
    po2_channels=True: draw channels from powers of two within [min,max].
    """
    ndim = rank

    def axis(rng_range):
        lo, hi = rng_range
        return rng.randint(lo, hi)

    def channel(rng_range):
        lo, hi = rng_range
        if not po2_channels:
            return rng.randint(lo, hi)
        choices = [1 << e for e in range(10) if lo <= (1 << e) <= hi]  # 1..512
        return rng.choice(choices) if choices else rng.randint(lo, hi)

    sp = tuple(axis(spatial[d]) for d in range(ndim))
    if square and ndim == 2:
        kv = min(axis(k[0]), min(sp))
        sv = axis(s[0])
        ker = (kv, kv)
        st = (sv, sv)
    else:
        ker = tuple(min(axis(k[d]), sp[d]) for d in range(ndim))
        if square_stride and ndim == 2:
            sv = axis(s[0])
            st = (sv, sv)
        else:
            st = tuple(axis(s[d]) for d in range(ndim))

    # padding="same" with stride > 1 is rejected by torch and not meaningful;
    # fall back to "valid" whenever any stride axis is > 1.
    pad = padding
    if pad == "same" and any(v > 1 for v in st):
        pad = "valid"

    return {"rank": rank, "c_in": channel(c_in), "c_out": channel(c_out),
            "spatial": sp, "kernel": ker, "stride": st, "padding": pad}


# ---------------------------------------------------------------------------
# Sweep config (5 categories: 3 Conv2D kernel/stride regimes + 2 Conv1D)
# ---------------------------------------------------------------------------

N = 50
CH = [1, 512]                 # 2^0 .. 2^9
SP2 = [[16, 64], [16, 64]]
SP1 = [[16, 128]]

CATEGORIES = [
    # tag,    sampler kwargs
    ("sqsq",  dict(rank=2, square=True,                      k=[[1, 5]],         s=[[1, 5]],         spatial=SP2, seed=11)),
    ("nsqsq", dict(rank=2, square=False, square_stride=True, k=[[1, 9], [1, 9]], s=[[1, 9]],         spatial=SP2, seed=22)),
    ("nsqns", dict(rank=2, square=False,                     k=[[1, 9], [1, 9]], s=[[1, 9], [1, 9]], spatial=SP2, seed=33)),
    ("c1ds1", dict(rank=1, square=False,                     k=[[1, 9]],         s=[[1, 1]],         spatial=SP1, seed=44)),
    ("c1ds9", dict(rank=1, square=False,                     k=[[1, 9]],         s=[[1, 9]],         spatial=SP1, seed=55)),
]


def _cfg_to_params(cfg) -> ConvModelParams:
    """Map a sampled geometry dict to ConvModelParams (channels-first NCHW view:
    spatial = (H, W) for rank 2, (L,) for rank 1)."""
    sp = cfg["spatial"]
    ker = cfg["kernel"]
    st = cfg["stride"]
    if cfg["rank"] == 1:
        (L,), (kw,), (sw,) = sp, ker, st
        return ConvModelParams(
            width=L, height=1, filters=cfg["c_out"], kernel_size=kw,
            input_channels=cfg["c_in"], rank=1, padding=cfg["padding"],
            kernel_w=kw, stride_w=sw,
        )
    (h, w), (kh, kw), (sh, sw) = sp, ker, st
    return ConvModelParams(
        width=w, height=h, filters=cfg["c_out"], kernel_size=kh,
        input_channels=cfg["c_in"], rank=2, padding=cfg["padding"],
        kernel_h=kh, kernel_w=kw, stride_h=sh, stride_w=sw,
    )


def conv_sweep_params(n=N):
    """Deterministic list of ConvModelParams across all sweep categories."""
    params = []
    for _tag, kw in CATEGORIES:
        sampler_kw = dict(kw)
        rng = random.Random(sampler_kw.pop("seed"))
        for _ in range(n):
            cfg = _sample(rng, c_in=CH, c_out=CH, padding="valid",
                          po2_channels=True, **sampler_kw)
            params.append(_cfg_to_params(cfg))
    return params


# ---------------------------------------------------------------------------
# ONNX backend (torch -> onnx.ModelProto, in memory)
# ---------------------------------------------------------------------------


@versioned_hashable_object_fixture
def conv_onnx_dtype(case_config):
    # 'f32' or 'bf16'
    return case_config.get('onnx_dtype', 'f32')


def _build_conv(p):
    Conv = nn.Conv2d if p.rank == 2 else nn.Conv1d
    if p.rank == 2:
        return Conv(p.input_channels, p.filters, kernel_size=(p.kh, p.kw),
                    stride=(p.sh, p.sw), padding=p.padding, bias=True)
    return Conv(p.input_channels, p.filters, kernel_size=p.kw,
                stride=p.sw, padding=p.padding, bias=True)


@onnx_model_fixture
def conv_onnx_model(request, keras_model_params, conv_onnx_dtype):
    # keras_model_params carries the ConvModelParams (shared with the keras path).
    p = keras_model_params
    torch.manual_seed(0)
    net = nn.Sequential(_build_conv(p)).eval()
    if p.rank == 2:
        x = torch.randn(1, p.input_channels, p.height, p.width)
    else:
        x = torch.randn(1, p.input_channels, p.width)

    buf = io.BytesIO()
    torch.onnx.export(net, x, buf, opset_version=17,
                      input_names=["input"], output_names=["output"])
    model = onnx.load_from_string(buf.getvalue())

    if conv_onnx_dtype == "bf16":
        model = convert_fp32_to_bf16(model)

    return model


# ---------------------------------------------------------------------------
# TFLite flavor converter (f32 / w8if32 dynamic-range / w8i8 full int8)
# ---------------------------------------------------------------------------


@versioned_hashable_object_fixture
def conv_sweep_flavor(case_config):
    return case_config["flavor"]


@versioned_generated_file_fixture("tflite")
def conv_sweep_tflite_model_file(request, versioned_file, keras_model, conv_sweep_flavor):
    model = keras_model.data if hasattr(keras_model, "data") else keras_model
    conv = tf.lite.TFLiteConverter.from_keras_model(model)

    if conv_sweep_flavor == "f32":
        pass
    elif conv_sweep_flavor == "w8if32":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif conv_sweep_flavor == "w8i8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        shapes = [[1 if d is None else d for d in inp.shape] for inp in model.inputs]
        _rng = np.random.default_rng(1234)
        conv.representative_dataset = lambda: (
            [_rng.random(s, dtype=np.float32) for s in shapes] for _ in range(16)
        )
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    else:
        raise ValueError(f"unknown conv sweep flavor: {conv_sweep_flavor}")

    with open(versioned_file, "wb") as f:
        f.write(conv.convert())
