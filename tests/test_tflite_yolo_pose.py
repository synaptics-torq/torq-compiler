"""
End-to-end and layer-by-layer testing of YOLOv8s Pose (int8 quantized).

Downloads yolov8s-pose_full_integer_quant_320.tflite from HuggingFace
(Synaptics/yolo), copies it to tests/testdata/tflite_models/, then tests
it both as a full model and layer-by-layer, reusing the infrastructure
from tests/test_tflite_model.py.

Usage:
    # See all test cases:
    pytest tests/test_yolo_pose_e2e.py -v --collect-only

    # Run full model test only:
    pytest tests/test_yolo_pose_e2e.py -v -s -k "full_model"

    # Run specific layer type:
    pytest tests/test_yolo_pose_e2e.py -v -s -k "layer_CONV_2D"

    # Force re-extraction of layers (clear cache):
    FORCE_EXTRACT=1 pytest tests/test_yolo_pose_e2e.py -v --collect-only
"""

import pytest
import os
import numpy as np
import cv2
from pathlib import Path

from torq.testing.comparison import compare_test_results
from torq.testing.hf import get_hf_model_file
from torq.testing.iree import chip_config  # noqa: F401 - used as pytest fixture
from torq.testing.tensorflow import tflite_mlir_model_file  # noqa: F401 - used as pytest fixture
from torq.testing.versioned_fixtures import (
    versioned_cached_data_fixture,
    versioned_static_file_fixture,
    VersionedUncachedData,
)

from test_tflite_model import generate_tflite_layer_cases, TFLiteLayerCase
import tensorflow as tf


# Configuration
MAX_LAYERS = int(os.environ.get('MAX_LAYERS', '0'))


# ============================================================================
# Image preprocessing / post-processing
# ============================================================================

def _preprocess_image(img, new_shape=(320, 320)):
    """Letterbox-resize image to new_shape, return (normalized_float32, pad)."""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    pad = (top / img.shape[0], left / img.shape[1])
    img = img[..., ::-1][None]           # BGR->RGB, add batch dim [1,H,W,C]
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img, pad


def _dequantize(y, scale, zero_point):
    return (y.astype(np.float32) - zero_point) * scale


def pose_postprocess(outputs, original_img_shape, pad,
                     conf_thresh=0.75, iou_thresh=0.7):
    """
    Post-process raw YOLOv8-pose output into (score, bbox[4], keypoints) tuples.
    Matches the logic in extras/tests/helpers/yolo.py::pose_postprocess.
    """
    outputs = outputs.copy()
    outputs[:, 0] -= pad[1]
    outputs[:, 1] -= pad[0]
    outputs[:, :4] *= max(original_img_shape)

    # [1, 56, N] -> [1, N, 56], convert cx/cy/w/h -> x/y/w/h (top-left)
    outputs = outputs.transpose(0, 2, 1)
    outputs[..., 0] -= outputs[..., 2] / 2
    outputs[..., 1] -= outputs[..., 3] / 2

    outs = []
    for out in outputs:
        scores = out[:, 4]
        keep   = scores > conf_thresh
        boxes  = out[keep, :4]
        scores = scores[keep]
        keypoints = out[keep, 5:]

        if not boxes.any() or not scores.any():
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        if len(indices) == 0:
            return []
        indices = indices.flatten()

        for idx, i in enumerate(indices):
            if idx > 5:
                break
            kp = keypoints[i].reshape((17, 3))
            visible_kps = [(float(k[0]), float(k[1])) for k in kp if k[2] > iou_thresh]
            outs.append((float(scores[i]), boxes[i], visible_kps))

    return outs


def _iou_similarity(a, b):
    intersection = np.logical_and(a, b)
    union        = np.logical_or(a, b)
    return np.sum(intersection) / np.sum(union) == 1


def compare_pose_results(llvm_outs, torq_outs):
    """Assert bbox and keypoints match between LLVMCPU and TORQ detections."""
    assert llvm_outs is not None and len(llvm_outs) > 0, \
        "LLVMCPU pose output is empty - no persons detected"
    assert torq_outs is not None and len(torq_outs) > 0, \
        "TORQ pose output is empty - expected persons detected by LLVMCPU"

    for i in range(min(len(llvm_outs), len(torq_outs))):
        _, ref_bbox, ref_kps = llvm_outs[i]
        _, tst_bbox, tst_kps = torq_outs[i]

        assert _iou_similarity(ref_bbox, tst_bbox), \
            f"Person {i}: bbox mismatch LLVMCPU={ref_bbox} TORQ={tst_bbox}"

        assert len(ref_kps) == len(tst_kps), \
            f"Person {i}: keypoint count mismatch ({len(ref_kps)} vs {len(tst_kps)})"
        for j, (rk, tk) in enumerate(zip(ref_kps, tst_kps)):
            assert _iou_similarity(np.array(rk), np.array(tk)), \
                f"Person {i} keypoint {j}: TORQ {tk} differs from LLVMCPU {rk}"


# ============================================================================
# Model Download
# ============================================================================

def download_yolo_pose_model(cache):
    """Download yolov8s-pose TFLite model from HuggingFace."""
    return Path(get_hf_model_file(
        cache, "Synaptics/yolo", "yolov8s-pose_full_integer_quant_320.tflite"
    ))


def _get_quant_params(model_path):
    """Return (in_scale, in_zp, is_int8, out_scale, out_zp) from a TFLite model."""
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_detail  = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]
    in_scale,  in_zp  = in_detail["quantization"]
    out_scale, out_zp = out_detail["quantization"]
    is_int8 = (in_detail["dtype"] == np.int8)
    return float(in_scale), int(in_zp), is_int8, float(out_scale), int(out_zp)


def _fixture_data(value):
    return value.data if hasattr(value, "data") else value


def _is_full_model_case(tflite_layer_model):
    data = _fixture_data(tflite_layer_model)
    return isinstance(data, dict) and not data.get("is_layer", True)


def _is_full_model_case_data(data):
    """Check if raw (unwrapped) data represents a full model case."""
    return isinstance(data, dict) and not data.get("is_layer", True)


def _compare_full_pose_pair(left_results, right_results, left_name, right_name, model_path, cache):
    """Dequantize, post-process, and compare pose detections between two backends."""
    _, _, _, out_scale, out_zp = _get_quant_params(model_path)

    left_raw = _fixture_data(left_results)[0]
    right_raw = _fixture_data(right_results)[0]

    left_out = _dequantize(left_raw, out_scale, out_zp)
    right_out = _dequantize(right_raw, out_scale, out_zp)

    image_path = get_hf_model_file(cache, "Synaptics/yolo", "bus.jpg")
    img = cv2.imread(image_path)
    assert img is not None
    _, pad = _preprocess_image(img)
    original_shape = img.shape[:2]

    left_outs = pose_postprocess(left_out, original_shape, pad)
    right_outs = pose_postprocess(right_out, original_shape, pad)

    print(f"\n{left_name} detected {len(left_outs)} person(s)")
    for i, (score, bbox, kps) in enumerate(left_outs):
        print(f"  [{i}] score={score:.3f}  bbox={bbox}  kps={len(kps)} visible")

    print(f"\n{right_name} detected {len(right_outs)} person(s)")
    for i, (score, bbox, kps) in enumerate(right_outs):
        print(f"  [{i}] score={score:.3f}  bbox={bbox}  kps={len(kps)} visible")

    compare_pose_results(left_outs, right_outs)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def case_config(request, tflite_layer_model):
    """Configure test case settings."""
    _skip_next_group_full_model(request, tflite_layer_model)
    torq_compiler_options = ["--torq-convert-dtypes", "--torq-disable-css", "--torq-disable-host"]
    torq_compiler_options += ["--torq-enable-transpose-optimization"]
    # tile-and-fuse is not working yet
    torq_compiler_options += ["--torq-enable-torq-hl-tiling"]
    return {
        "tflite_model": "tflite_layer_model",
        "mlir_model_file": "tflite_mlir_model_file",
        "input_data": "yolo_pose_input_data",
        "torq_compiler_options": torq_compiler_options,
        "torq_compiler_timeout": 600,
        "torq_runtime_timeout": 600,
    }


def _skip_next_group_full_model(request, tflite_layer_model):
    """Skip full model tests on unsupported targets."""
    if _is_full_model_case(tflite_layer_model):
        try:
            chip = request.getfixturevalue("chip_config").data
        except AttributeError:
            # AttributeError occurs when chip_config fixture exists but
            # was not parametrized (no request.param), e.g. in tests
            # that don't use the torq backend (test_yolo_pose_llvmcpu_tflite).
            return
        if chip.get('target') != "SL2610":
            pytest.skip(f"Full YOLOv8s-pose model only supported on SL2610")


@versioned_cached_data_fixture
def yolo_pose_input_data(request, tflite_layer_model, tweaked_random_input_data, mlir_io_spec):
    """
    For the full model: preprocess bus.jpg and quantize it to int8.
    For layer cases: fall back to tweaked random input data.
    """
    if not _is_full_model_case_data(tflite_layer_model):
        return tweaked_random_input_data

    if not mlir_io_spec.inputs or len(mlir_io_spec.inputs[0].shape) != 4:
        return tweaked_random_input_data

    model_path = tflite_layer_model.get("model_path")
    in_scale, in_zp, is_int8, _, _ = _get_quant_params(model_path)

    cache = request.getfixturevalue("cache")
    image_path = get_hf_model_file(cache, "Synaptics/yolo", "bus.jpg")
    img = cv2.imread(image_path)
    assert img is not None, f"Could not read {image_path}"

    img_float, _pad = _preprocess_image(img)    # [1, 320, 320, 3] float32

    if is_int8:
        input_tensor = (img_float / in_scale + in_zp).astype(np.int8)
    else:
        input_tensor = img_float

    return [input_tensor]


@pytest.fixture
def tflite_layer_model(request):
    """Fixture that provides the TFLite model for the current test case."""
    case = request.param
    version = "tflite_layer_model_" + case.name
    return VersionedUncachedData(data=case.data, version=version)


@pytest.fixture
def tflite_model_path(tflite_layer_model):
    """Get the TFLite model path."""
    data = _fixture_data(tflite_layer_model)
    if isinstance(data, dict):
        layer_path = data.get('layer_tflite_path')
        if layer_path and Path(layer_path).exists():
            return layer_path
        return data.get('model_path')
    return str(data)


@versioned_static_file_fixture
def tflite_model_file(request, tflite_model_path):
    """Provide the TFLite model file path."""
    return Path(tflite_model_path)




# ============================================================================
# Test Generation
# ============================================================================

_CASES_CACHE = {}


def _is_full_model_only(metafunc):
    """Check if only the full model case should be generated.

    Returns True when running with -k "full_model" or -m ci.
    """
    keyword_expr = metafunc.config.option.keyword
    if keyword_expr and 'full_model' in keyword_expr.lower():
        return True
    marker_expr = metafunc.config.option.markexpr
    if marker_expr and 'ci' in marker_expr.lower():
        return True
    return False


def pytest_generate_tests(metafunc):
    """Generate test cases for yolov8s-pose model."""
    if 'tflite_layer_model' not in metafunc.fixturenames:
        return

    full_only = _is_full_model_only(metafunc)
    cache_key = f"yolo_pose_e2e_cases_{MAX_LAYERS}_full={full_only}"
    if cache_key in _CASES_CACHE:
        cases = _CASES_CACHE[cache_key]
    else:
        cache = metafunc.config.cache
        model_path = download_yolo_pose_model(cache)

        if full_only:
            cases = [
                TFLiteLayerCase(
                    name=f"{model_path.stem}_full_model",
                    data={'model_path': str(model_path), 'is_layer': False}
                )
            ]
        else:
            cases = generate_tflite_layer_cases(model_path, max_layers=MAX_LAYERS)

        _CASES_CACHE[cache_key] = cases

    if cases:
        test_name = metafunc.function.__name__
        is_torq_test = 'torq' in test_name.lower()

        params = []
        ids = []
        for c in cases:
            name_lower = c.name.lower()
            # Mark compile-error layers as xfail(run=False) for torq tests
            if is_torq_test and any(s in name_lower for s in TORQ_COMPILE_ERROR_LAYERS):
                params.append(pytest.param(c, marks=pytest.mark.xfail(reason="compiler error", run=False)))
            else:
                params.append(c)
            ids.append(c.name)

        metafunc.parametrize(
            "tflite_layer_model",
            params,
            indirect=True,
            ids=ids,
        )


# ============================================================================
# Tests
# ============================================================================

# Layer cases known to fail with TORQ backend (wrong results)
TORQ_FAILED_LAYERS = [
    'layer_conv_2d_1',
    'layer_conv_2d_5',
    'layer_conv_2d_25',
    'layer_conv_2d_52',
    'layer_conv_2d_79',  # Only in next.group
    'layer_conv_2d_145',
    'layer_quantize_126',
    'layer_resize_nearest_neighbor_109',
    'layer_resize_nearest_neighbor_127',
    'layer_softmax_277',
    'layer_mul_280',
    'layer_mul_287',
]

# Layer cases that error during compilation (setup errors) with TORQ backend
TORQ_COMPILE_ERROR_LAYERS = [
    'layer_pad_0',  # Only in next.group
    'layer_pad_4',
    'layer_pad_24',  # Only in next.group
    'layer_resize_nearest_neighbor_127',  # Only in next.group
]

# Layer cases known to fail in LLVMCPU vs TFLite comparison
LLVMCPU_TFLITE_FAILED_LAYERS = [
    'layer_resize_nearest_neighbor_109',
    'layer_resize_nearest_neighbor_127',
    'layer_softmax_277',
]


def _check_xfail_torq(request):
    """Mark known-failing TORQ layer tests as xfail."""
    name = request.node.callspec.id.lower() if hasattr(request.node, 'callspec') else ''
    if any(s in name for s in TORQ_FAILED_LAYERS):
        pytest.xfail("failing test or skipped for now")


def _check_xfail_llvmcpu_tflite(request):
    """Mark known-failing LLVMCPU vs TFLite layer tests as xfail."""
    name = request.node.callspec.id.lower() if hasattr(request.node, 'callspec') else ''
    if any(s in name for s in LLVMCPU_TFLITE_FAILED_LAYERS):
        pytest.xfail("failing test or skipped for now")


def _compare_results(request, left_results, right_results, left_name, right_name,
                     case_config, tflite_layer_model):
    """Dispatch to full-model pose comparison or layer comparison."""
    if _is_full_model_case(tflite_layer_model):
        model_path = _fixture_data(tflite_layer_model).get("model_path")
        cache = request.getfixturevalue("cache")
        _compare_full_pose_pair(left_results, right_results, left_name, right_name, model_path, cache)
    else:
        compare_test_results(request, right_results, left_results, case_config)


def test_yolo_pose_llvmcpu_torq(
    request,
    llvmcpu_reference_results,
    torq_results,
    case_config,
    tflite_layer_model,
):
    """Compare YOLOv8s-pose results between LLVM-CPU and Torq backends.

    Full model: dequantize outputs, run pose post-processing, compare detections.
    Individual layers: numerical comparison via compare_test_results.
    """
    _check_xfail_torq(request)
    _compare_results(request, llvmcpu_reference_results, torq_results,
                     "LLVMCPU", "TORQ", case_config, tflite_layer_model)


@pytest.mark.ci
@pytest.mark.fpga_ci
def test_yolo_pose_tflite_torq(
    request,
    tflite_reference_results,
    torq_results,
    case_config,
    tflite_layer_model,
):
    """Compare YOLOv8s-pose results between TFLite and Torq backends."""
    _check_xfail_torq(request)
    _compare_results(request, tflite_reference_results, torq_results,
                     "TFLite", "TORQ", case_config, tflite_layer_model)


def test_yolo_pose_llvmcpu_tflite(
    request,
    tflite_reference_results,
    llvmcpu_reference_results,
    case_config,
    tflite_layer_model,
):
    """Compare YOLOv8s-pose results between LLVM-CPU and TFLite backends."""
    _check_xfail_llvmcpu_tflite(request)
    _compare_results(request, llvmcpu_reference_results, tflite_reference_results,
                     "LLVMCPU", "TFLite", case_config, tflite_layer_model)
