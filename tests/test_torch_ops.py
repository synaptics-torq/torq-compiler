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

"""Torch-op fixture tests.

Python builders define the models. The harness imports their ONNX or PyTorch
graphs to MLIR at test time, using `import_onnx --data-prop` for ONNX.

CLI examples:
  pytest tests/test_torch_ops.py -v -s --collect-only
  pytest tests/test_torch_ops.py -v -s add-25x511-bf16.mlir
"""

# Tolerances ported from the original static MLIR directive comments.
_MLIR_DIRECTIVE_TOLS = {
    "conv2d-host.mlir": {"epsilon": 1e-3},
    "erf_mlp_dup_lram_load_bf16.mlir": {
        "fp_max_tol": 0.2, "fp_avg_tol": 0.03,
        "use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.001,
    },
    "conv1d_matmul_bf16_tile_and_fuse.mlir": {"fp_max_tol": 0.05},
    # A few outputs differ from the float reference by one int8 step: the gate allows one
    # step, not two.
    "conv-qdq-sliced-remainder-i8.mlir": {"use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.03},
    "conv1d-qdq-wide-kernel-k9-8x100-i8.mlir": {"use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.01},
    "conv1d-qdq-wide-kernel-k11-8x100-i8.mlir": {"use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.01},
    "Conv2d_bf16_1x1x64x8192.mlir": {"fp_max_tol": 1.1, "fp_avg_tol": 0.2},
    "cos-coarse-1024x32xbf16.mlir": {"allowed_wrong": 0.07},
    "dequant_matmul_zp_n6.mlir": {"fp_max_tol": 0.1, "allowed_wrong": 0.1},
    "dequant_matmul_zp_nested_tile.mlir": {"fp_max_tol": 0.1, "allowed_wrong": 0.1},
    "dequant_matmul_zp_wide_n.mlir": {"fp_max_tol": 0.1, "allowed_wrong": 0.1},
    "dynamicquantize_large.mlir": {"int_tol": 2.0},
    "dynamicquantize.mlir": {"int_tol": 2.0},
    "gemm_bf16.mlir": {"fp_max_tol": 0.015},
    "gemm-fc-row-channel-tiling-bf16.mlir": {"fp_max_tol": 0.05},
    # Since df56d91e the epilogue fusion of the matmuls stops at runtime data such as the
    # carried hidden state: one of the 32 outputs is one bf16 step off (23.25 for 23.375).
    "gru_s4_b1_h8_bf16.mlir": {"fp_max_tol": 0.035, "allowed_wrong": 0.05},
    "layernorm-nss-1x128x768-bf16.mlir": {
        "fp_max_tol": 0.02, "allowed_wrong": 0.0,
        "use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.01,
    },
    # Random affine weights expose the same bf16 rounding near cancellation
    # as the larger LayerNorm case (max absolute difference: 0.03125).
    "layernorm_8x1x32.mlir": {
        "fp_max_tol": 0.02, "allowed_wrong": 0.0,
        "use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.01,
    },
    "lstm_s4_b1_h8_bf16.mlir": {"fp_max_tol": 0.035},
    "matmul-1x128x3072x768-bf16.mlir": {
        "fp_avg_tol": 0.05, "fp_max_tol": 0.15, "allowed_wrong": 0.01,
    },
    "matmulinteger_depthwise_conv.mlir": {
        "use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.015,
    },
    "matmulinteger_depthwise_conv_wzp.mlir": {
        "use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.015,
    },
    "matmulinteger_fused_qkv.mlir": {
        "use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.015,
    },
    "matmulinteger_qkv_rank3.mlir": {
        "use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.015,
    },
    "matmulinteger_u8_weight_full_range_perchannel.mlir": {
        "use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.01,
    },
    "reducemean.mlir": {"fp_max_tol": 0.31, "fp_avg_tol": 0.03},
    "reducemean-reshape.mlir": {"fp_max_tol": 0.31, "fp_avg_tol": 0.03},
    "reducemean-tiling.mlir": {"fp_max_tol": 0.091},
    "reducemean_wide_kept_dims_bf16.mlir": {"fp_max_tol": 0.06, "fp_avg_tol": 0.02},
    "resize_nearest_upsample_fpn_bf16.mlir": {"fp_max_tol": 0.05, "allowed_wrong": 0.03},
    "sin-coarse-1024x32xbf16.mlir": {"allowed_wrong": 0.06},
}

_TORCH_DTYPES = {}  # populated lazily (avoids a torch import at collection time)


def _torch_dtypes():
    if not _TORCH_DTYPES:
        import torch
        _TORCH_DTYPES.update({
            "f32": torch.float32,
            "bf16": torch.bfloat16,
            "f16": torch.float16,
            "i1": torch.bool,
            "si8": torch.int8,
            "si16": torch.int16,
            "si32": torch.int32,
            "si64": torch.int64,
            "ui8": torch.uint8,
        })
    return _TORCH_DTYPES


# ---------------------------------------------------------------------------
# Collection: per-op-family builder modules
# ---------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    if "case_config" not in metafunc.fixturenames:
        return
    cases = _family_cases()
    cases.sort(key=lambda c: (c.data["root"], c.data["group"], c.data["name"]))
    metafunc.parametrize("case_config", cases, indirect=True, ids=[c.name for c in cases])


# ---------------------------------------------------------------------------
# Generated model fixtures
# ---------------------------------------------------------------------------

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
    dts = _torch_dtypes()
    example = tuple(torch.zeros(dims, dtype=dts[dt]) for dims, dt in io)

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


# ---------------------------------------------------------------------------
# Case configuration
# ---------------------------------------------------------------------------

@pytest.fixture
def comparison_config_for_encoder_conv(request):
    """Looser comparison for the encoder conv1d (runs with host/CSS fallback enabled)."""
    return {"fp_avg_tol": 0.2, "fp_max_tol": 1.0}

@pytest.fixture
def comparison_config_for_localavgpool(request):
    return {"fp_avg_tol": 0.2, "fp_max_tol": 1.004}

@pytest.fixture
def comparison_config_for_gelu(request):
    """Comparison config for GELU.

    GELU can produce tiny values in the far negative tail. The observed
    mismatch was:

        expected = -1.611188e-7, observed = -0.0, abs_diff = 1.611188e-7

    This is safe for GELU correctness because the mismatch is in the tail where
    the true GELU result is already effectively zero.
    """
    return {"epsilon": 2e-5}


@pytest.fixture
def comparison_config_for_matmul_dql(request):
    """Comparison config for block-quantized matmul (DequantizeLinear + MatMul).

    Block quantization introduces rounding at block boundaries that
    amplifies relative error for small output values.
    """
    return {"fp_max_tol": 0.5, "allowed_wrong": 0.2}


@pytest.fixture
def comparison_config_for_instancenorm(request):
    """Comparison config for InstanceNormalization.

    The normalized bf16 output carries ~1 bf16 ULP of error; against the fp32
    numpy reference a few near-zero elements just exceed the default fp_max_tol.
    """
    return {"fp_max_tol": 2e-2, "allowed_wrong": 1e-3}


@pytest.fixture
def comparison_config_for_encoder_stride64(request):
    """Comparison config for the stride-64 conv1d (audio-encoder layer).

    The S2D fold accumulates this conv's products in a different order than
    the llvm-cpu f32 reference, leaving a few near-cancellation outputs
    (~3.5e-5) ~1 bf16 ULP off the reference, beyond the default relative
    fp_max_tol.  Measured on sim-sl2610: 9/359712 differ (max relative 0.016).
    The small allowed_wrong fraction covers them without masking a real
    regression (which would affect thousands of elements).
    """
    return {"allowed_wrong": 1e-4}


@pytest.fixture
def comparison_config_for_qdq_dequant(request):
    """int8 QDQ dequant zp regression (deq_min_i8, qdq_min_i8): residual is bf16 MulOp rounding."""
    return {"fp_max_tol": 0.2, "allowed_wrong": 0.06}


@pytest.fixture
def comparison_config_for_qdq_conv_tiled(request):
    """Tiled int8 QDQ conv W-shift regression (conv1d_307_qdq_tiled): residual is bf16 requant rounding."""
    return {"fp_max_tol": 0.2, "allowed_wrong": 0.02}


@pytest.fixture
def comparison_config_for_qdq_conv_295(request):
    """Legacy-path tiled int8 QDQ conv (conv1d_295_qdq_tiled, kw=11). Its 1x64x6401 output spans a
    wide dynamic range (~1.0 down to ~1e-4), so the near-zero tail's bf16 requant rounding trips the
    relative-error metric on a few percent of elements, and the exact fraction varies by chip requant
    (e.g. ~2.0% on sr200 vs less on sl2610). Loosened accordingly -- a real W-shift regression is ~80%
    wrong, nowhere near this floor."""
    return {"fp_max_tol": 0.2, "allowed_wrong": 0.05}


@pytest.fixture
def comparison_config_for_qdq_izp(request):
    """int8 QDQ conv input-zero-point bias-fold regression (conv_izp_i8): residual is bf16 requant rounding."""
    return {"fp_max_tol": 0.2, "allowed_wrong": 0.06}


@pytest.fixture
def torch_ops_comparison_config(request, torch_ops_case_data):
    """Tolerances ported from the `// TORQ_*:` comments of the original static MLIRs."""
    return dict(_MLIR_DIRECTIVE_TOLS.get(torch_ops_case_data.data["name"], {}))


@pytest.fixture
def case_config(request, runtime_hw_type, chip_config):
    data = request.param.data
    name = data["name"]

    no_negative_input = [
        'sqrt-',
    ]

    extra_args = {}
    if any(s in name for s in no_negative_input):
        extra_args["tweaked_input_data_range"]  = (0, 100)

    # sin/cos are optimized for RoPE, which only runs on (often small)
    # positive input.  For that reason, negative inputs are not quite
    # as accurate as they could be.
    if 'sin-exact' in name:
        extra_args["tweaked_input_data_range"] = 0, 3
    if 'cos-exact' in name:
        extra_args["tweaked_input_data_range"] = 0, 1.5
    if 'sin-coarse' in name:
        extra_args["tweaked_input_data_range"] = 0, 12
    if 'cos-coarse' in name:
        extra_args["tweaked_input_data_range"] = 0, 12

    # The K-split matmul case accumulates chunk partials in bf16; keep inputs
    # positive so cancellation noise doesn't dominate the comparison (the
    # TORQ_* tolerance directives cover the residual rounding).
    if 'matmul-1x128x3072x768' in name:
        extra_args["tweaked_input_data_range"] = 0, 2

    no_slicing_tc = [
        # assertion for next 2 in function LData, file Kernel.cpp, line 824
        #  (elementType != DType::none && "Invalid elemen= DType::none)
        "topk-1x2100-bf16",
        "topk-1x2100-indices-only-bf16",
        # LRAM-limit regression pinned to the no-slicing path used for real
        # models (the sliced path fails this shape for an unrelated
        # pre-existing reason: an LRAM fragmentation failure).
        "erf_mlp_dup_lram_load_bf16",
        # LRAM-limit regression pinned to the no-slicing path used for real
        # models (the sliced path splits the same reduction across slices and
        # has an unrelated pre-existing numeric mismatch at this width).
        "reducemean_wide_kept_dims_bf16",

    ]
    extra_args["torq_compiler_options"] = []
    if any(s in name for s in no_slicing_tc):
        extra_args["torq_compiler_options"].append("--torq-disable-slicing")

    # The topk cases lower their sort to a Host/CSS software kernel. This is not
    # a compiler or runtime change: simulating that sort over 2100 elements on
    # the CModel is inherently slow, so it can exceed the 240s default runtime
    # timeout on a slower CI runner for the sl2610-v1 sim. The extra budget
    # applies to sim only, so the strict default still catches a genuinely slow
    # sort on fpga/hardware.
    if (
        runtime_hw_type.data == "sim"
        and any(s in name for s in ["topk-1x2100-bf16", "topk-1x2100-indices-only-bf16"])
    ):
        extra_args["torq_runtime_timeout"] = 600

    # Force in-tree torch_ops cases onto the NSS/slice path so we fail loudly
    # if a future change silently routes them to the host/CSS fallback.
    # torch_ops_host_css/ and extras testdata are exempted.
    # Per-case blocks below may replace this list when they need different flags.
    if data["group"] == "torch_ops" and data["root"] == "tests":
        extra_args["torq_compiler_options"].extend([
            "--torq-disable-host", "--torq-disable-css",
        ])

    # ONNX DynamicQuantizeLinear tensors are always f32, so DQL is only reachable
    # in a bf16 pipeline via --torq-convert-dtypes. The same holds for the
    # DQL -> MatMulInteger -> dequant chain.
    if any(s in name for s in ['dynamicquantize', 'matmulinteger']):
        extra_args["torq_compiler_options"].extend([
            "--torq-convert-dtypes", "--torq-convert-io-dtype",
            "--torq-disable-host", "--torq-disable-css",
        ])

    # Preserve fallback coverage for the relocated convolution probes.
    if name in {"conv2d-host.mlir", "conv1d-bf16-k127-s64.mlir",
                "conv1d-bf16-c288-k7-s3.mlir", "conv1d-bf16-c576-k3-s2.mlir"}:
        extra_args["torq_compiler_options"] = []

    # Preserve the comparison settings for the reduced strided Conv1D probe.
    if name == "conv1d-bf16-k127-s64.mlir":
        extra_args["comparison_config"] = "comparison_config_for_encoder_conv"

    if 'localavgpool.mlir' in name:
        extra_args["comparison_config"] = "comparison_config_for_localavgpool"
    
    if 'gelu' in name:
        extra_args["comparison_config"] = "comparison_config_for_gelu"

    if 'matmul_dql' in name:
        extra_args["comparison_config"] = "comparison_config_for_matmul_dql"

    if 'instancenorm' in name:
        extra_args["comparison_config"] = "comparison_config_for_instancenorm"
        # torch-mlir 3.12 decomposes InstanceNormalization in f32 inside the
        # dispatch (bf16 interfaces); the NSS elementwise path needs bf16, so
        # demote the cone back (3.10 decomposed in bf16 directly).
        extra_args["torq_compiler_options"] = ["--torq-convert-dtypes"]

    if "encoder_bf16_layer_Conv_2" in name:
        extra_args["comparison_config"] = "comparison_config_for_encoder_stride64"

    # Force the bf16-stash LayerNormalization (bert shape) onto the NSS/slice
    # path: the model carries stash_type=BFLOAT16 so torch-mlir decomposes it
    # directly in bf16 (no f32 cone), avoiding the need for a compiler demote.
    if 'layernorm-nss' in name:
        extra_args["torq_compiler_options"] = [
            "--torq-disable-host",
            "--torq-disable-css",
        ]

    # Force the Conv1D-as-matmul -> fully_connected lowering tests to run on
    # the NSS/slice path so they fail loudly if a future change silently routes
    # them to the host/CSS fallback.
    if 'conv1d_matmul_bf16_' in name:
        extra_args["torq_compiler_options"] = ["--torq-disable-host", "--torq-disable-css"]

    if 'softmax-1x2xbf16.mlir' in name:
        extra_args["torq_compiler_options"] = ["--torq-disable-css", "--torq-disable-host"]

    # Force the bf16 elementwise add onto the NSS/slice path so it fails loudly if a
    # future change stops lowering bf16 add via AddOpPattern (createBf16Add -> torq_hl.add).
    # (f32 two-tensor add is intentionally not lowered to NSS: the bf16-width data path
    # mis-strides f32 inputs and produces garbage.)
    if 'add-nss-25x511-bf16.mlir' in name:
        extra_args["torq_compiler_options"] = ["--torq-disable-host", "--torq-disable-css"]

    # Force the dynamic-indices gather onto the NSS/slice path so it fails loudly if
    # the value copy stops lowering to torq_hl.gather. CSS is left enabled on purpose:
    # the ONNX Gather index math (negative-index normalization on i64 plus the
    # i64->i32 demote) has no NSS data path and must run on CSS. Adding
    # --torq-disable-css would orphan those scalar ops and break serialization.
    if 'gather-dynamic-indices' in name:
        extra_args["torq_compiler_options"] = ["--torq-disable-host"]

    # int8 QDQ dequant zp regression: force onto NSS so a dropped-zp regression fails loudly.
    if any(s in name for s in ['deq_min_i8', 'qdq_min_i8']):
        extra_args["torq_compiler_options"] = ["--torq-disable-host", "--torq-disable-css"]
        extra_args["comparison_config"] = "comparison_config_for_qdq_dequant"

    # Tiled int8 QDQ conv W-shift regression: force onto NSS so the wide conv tiles along W.
    # Two variants guard the two conv lowering paths: conv1d_307 (kw=7) goes through the EK
    # kernel (Conv2DToHw.cpp); conv1d_295 (kw=11) fails hasEkLoweringConv (kw>7) and takes the
    # legacy Conv2DPattern.cpp path, whose baseOffset W-correction is what the fix restores.
    # Without that fix conv1d_295 is ~80% wrong; keep both so neither path can regress silently.
    if 'conv1d_307_qdq_tiled' in name:
        extra_args["torq_compiler_options"] = ["--torq-disable-host", "--torq-disable-css"]
        extra_args["comparison_config"] = "comparison_config_for_qdq_conv_tiled"

    if 'conv1d_295_qdq_tiled' in name:
        extra_args["torq_compiler_options"] = ["--torq-disable-host", "--torq-disable-css"]
        extra_args["comparison_config"] = "comparison_config_for_qdq_conv_295"

    # int8 QDQ conv input-zero-point regression: force onto NSS. The nonzero izp (si8 -23)
    # must fold into the bias via computeInputZpCorrection (NSS hardware does not subtract
    # izp); if that regresses, the NSS output diverges from the llvmcpu reference and fails.
    if 'conv_izp_i8' in name:
        extra_args["torq_compiler_options"] = ["--torq-disable-host", "--torq-disable-css"]
        extra_args["comparison_config"] = "comparison_config_for_qdq_izp"

    if 'isnan' in name:
        extra_args["torq_compiler_options"] = ["--torq-disable-host", "--torq-disable-css", "--torq-isnan-mode=remove"]

    # i64 scalar mul must run on host without slice lowering (see torch_ops_host_css/).
    if 'mul-i64-scalar.mlir' in name:
        extra_args["torq_compiler_options"] = ["--torq-disable-slices", "--torq-disable-css"]

    cfg = {
        "_case": data,
        "input_data": "tweaked_random_input_data",
        **extra_args,
    }

    if "comparison_config" not in cfg:
        cfg["comparison_config"] = "torch_ops_comparison_config"

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


def _case_name(case_config):
    return case_config.get("_case", {}).get("name", "")


@pytest.fixture
def reference_results(request, case_config):
    name = _case_name(case_config)

    if "gelu" in name.lower():
        return request.getfixturevalue("numpy_gelu_reference_results")

    if "globalaveragepool" in name.lower():
        return request.getfixturevalue("numpy_global_average_pool_reference_results")

    if "instancenorm" in name.lower():
        return request.getfixturevalue("numpy_instancenorm_reference_results")

    # The NPU accumulates reductions in fp32, while the llvm-cpu reference
    # accumulates in bf16. For these cases (large reductions, or small
    # reductions over large-amplitude inputs that cancel) the reference is
    # the inaccurate side, so use the fp32-accumulated numpy reference.
    if "reducemean-tiling" in name.lower() or "reducemean_wide_kept_dims_bf16" in name.lower():
        return request.getfixturevalue("numpy_reducemean_reference_results")

    if "matmul-1x128x" in name.lower():
        return request.getfixturevalue("numpy_matmul_const_weight_reference_results")

    try:
        return request.getfixturevalue("llvmcpu_reference_results")
    except Exception:
        return request.getfixturevalue("torch_reference_results")


@pytest.mark.ci
@pytest.mark.fpga_ci
def test_mlir_files(request, torq_results, reference_results, case_config):
    compare_test_results(request, torq_results, reference_results, case_config)
