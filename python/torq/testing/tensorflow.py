import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass
from functools import wraps
import pytest
import tensorflow as tf
import numpy as np
import json

from . import keras_tflite_convert
from .versioned_fixtures import versioned_generated_file_fixture, versioned_cached_data_fixture, versioned_hashable_object_fixture, versioned_unhashable_object_fixture

"""
This module provides fixtures and utilities for testing TensorFlow/Keras models.
"""



def generate_layers_from_model(model: tf.keras.Model):    
    """
    Generates test cases for each unique layer configuration in the given Keras model and for the whole model

    :param full_model: A tf.keras.Model instance representing the full model
    :param module_globals: The globals() dictionary of the module where the test cases will be added
    """

    existing_cases = set()
    layer_configs = {}

    print("Total layers in model: ", len(model.layers))

    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        
        if not isinstance(layer.input, list):
            layer_inputs = [layer.input]
        else:
            layer_inputs = layer.input

        inputs = [ tf.keras.Input(shape=inp.shape[1:], name=f"input_{idx}", batch_size=1) for idx, inp in enumerate(layer_inputs) ]        
        config = layer.get_config()

        # normalize name to avoid duplicates
        config['name'] = "LayerUnderTest"
        cloned_layer = layer.__class__.from_config(config)        
        output = cloned_layer( inputs if len(inputs) > 1 else inputs[0] )

        model = tf.keras.Model(inputs=inputs, outputs=output, name=f"Model")

        model_config = model.get_config()

        json_str = json.dumps(model_config, sort_keys=True)

        # create one test case for each unique layer configuration
        if json_str not in existing_cases:
            existing_cases.add(json_str)
        else:
            # print(f"Skipping duplicate model for layer {layer.name}")
            continue

        layer_configs[layer.name] = model_config

    print("Generated unique tests: ", len(existing_cases))

    return layer_configs


def _convert_in_child_process():
    """True if the TFLite converter must run in a child process (see keras_tflite_convert).

    TORQ_KERAS_TFLITE_CONVERT_IN_CHILD=1 or 0 forces the choice.
    """
    forced = os.environ.get("TORQ_KERAS_TFLITE_CONVERT_IN_CHILD")
    if forced is not None:
        return forced == "1"
    return sys.platform == "darwin" and "iree.compiler" in sys.modules


def _input_shapes(model):
    return [[1 if x is None else x for x in t.shape] for t in model.inputs]


def _convert_in_child(model, quantize, quantize_to_int16):
    with tempfile.TemporaryDirectory(prefix="keras_tflite_") as tmp:
        tmp = Path(tmp)
        # The shapes of the model inputs, as from_keras_model uses: without a signature, export
        # makes the batch dim dynamic even for a model with batch 1. The spec names are the
        # signature input names.
        specs = [tf.TensorSpec(t.shape, t.dtype, name=t.name) for t in model.inputs]
        model.export(
            str(tmp / "saved_model"), verbose=False,
            input_signature=[specs] if len(specs) > 1 else specs
        )
        config = {
            "quantize": quantize,
            "quantize_to_int16": quantize_to_int16,
            "input_names": [t.name for t in model.inputs],
            "input_shapes": _input_shapes(model),
        }
        (tmp / "config.json").write_text(json.dumps(config))
        # -P: don't put the directory of the script on sys.path, its numpy.py would hide numpy
        cmd = [sys.executable, "-P", keras_tflite_convert.__file__, str(tmp / "saved_model"),
               str(tmp / "model.tflite"), str(tmp / "config.json")]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"TFLite conversion in a child process failed ({result.returncode}):\n"
                f"{result.stderr[-4000:]}"
            )
        return (tmp / "model.tflite").read_bytes()


def quantize_model(model, quantize=True, quantize_to_int16=False):
    """
    Convert a Keras model to TFLite, quantized to 8-bit signed integers unless quantize is False
    (16-bit activations with quantize_to_int16).

    :param model: the Keras model
    :return: The TFLite model
    """
    if _convert_in_child_process():
        return _convert_in_child(model, quantize, quantize_to_int16)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    keras_tflite_convert.configure_converter(
        converter, quantize, quantize_to_int16,
        lambda: keras_tflite_convert.calibration_samples(_input_shapes(model))
    )
    return converter.convert()


@pytest.fixture
def keras_model(request, case_config):
    return request.getfixturevalue(case_config['keras_model'])


@pytest.fixture
def tflite_model_file(request, case_config):
    return request.getfixturevalue(case_config.get('tflite_model_file', 'quantized_tflite_model_file'))    


@versioned_hashable_object_fixture
def tflite_to_tosa_conversion():
    # Part of the version of the converted MLIR: change it when the way the TOSA text is
    # produced changes, so that cached files are regenerated.
    return {"text": "bytecode printed with the compiler's MLIR"}


@versioned_generated_file_fixture("mlir")
def tflite_mlir_model_file(request, versioned_file, tflite_model_file, tflite_to_tosa_conversion):
    # Workaround until tosa-converter-for-tflite ships MR 95 (quantized Abs with
    # mismatched scales). See torq.testing.tflite_abs_workaround.
    from .tflite_abs_workaround import convert_tflite_to_tosa_mlir_with_abs_workaround

    convert_tflite_to_tosa_mlir_with_abs_workaround(tflite_model_file, versioned_file)

@versioned_hashable_object_fixture
def tflite_quantization_params(case_config):
    return {"quantize_to_int16": case_config.get("quantize_to_int16", False)}


@versioned_generated_file_fixture("tflite")
def quantized_tflite_model_file(request, versioned_file, keras_model, tflite_quantization_params):

    quantize_to_int16 = tflite_quantization_params.get("quantize_to_int16", False)
    
    tflite_model = quantize_model(keras_model, quantize_to_int16=quantize_to_int16)

    with open(versioned_file, "wb") as f:
        f.write(tflite_model)


@versioned_generated_file_fixture("tflite")
def float32_tflite_model_file(request, versioned_file, keras_model):

    tflite_model = quantize_model(keras_model.data, quantize=False)

    with open(versioned_file, "wb") as f:
        f.write(tflite_model)


def _model_needs_builtin_kernels(tflite_model_file):
    """True if the model has an int8 BATCH_MATMUL with an asymmetric rhs.

    XNNPACK's int8 BatchMatMul assumes a symmetric rhs and silently ignores a
    nonzero rhs zero-point, computing (a - a_zp) . b. The builtin reference
    kernels honor both zero-points and match the float model (e.g. YOLO26 goes
    blind on real images under XNNPACK but detects correctly under the builtin
    kernels). Only such models opt out of the default delegate so every other
    case keeps the stock interpreter behavior.
    """
    from tensorflow.lite.tools import flatbuffer_utils

    BATCH_MATMUL = 126
    TENSOR_TYPE_INT8 = 9

    model = flatbuffer_utils.read_model_with_mutable_tensors(str(tflite_model_file))
    for subgraph in model.subgraphs:
        for op in subgraph.operators:
            opcode = model.operatorCodes[op.opcodeIndex]
            code = opcode.builtinCode or opcode.deprecatedBuiltinCode
            if code != BATCH_MATMUL or len(op.inputs) < 2:
                continue
            rhs = subgraph.tensors[op.inputs[1]]
            q = rhs.quantization
            if (
                rhs.type == TENSOR_TYPE_INT8
                and q is not None
                and q.zeroPoint is not None
                and len(q.zeroPoint) == 1
                and int(q.zeroPoint[0]) != 0
            ):
                return True
    return False


def run_with_tflite(tflite_model_file, input_data):
    interpreter_args = {}
    if _model_needs_builtin_kernels(tflite_model_file):
        interpreter_args["experimental_op_resolver_type"] = (
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file), **interpreter_args)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i, input_data_array in enumerate(input_data):
        interpreter.set_tensor(input_details[i]['index'], input_data_array)

    interpreter.invoke()

    return [(out, interpreter.get_tensor(out['index'])) for out in output_details]

@versioned_cached_data_fixture
def tflite_reference_results(request, tflite_model_file, input_data):
    results = run_with_tflite(tflite_model_file, input_data)
    return [output[1] for output in results] 


@versioned_hashable_object_fixture
def keras_layer_data(case_config):
    return case_config['keras_layer_data']


@versioned_unhashable_object_fixture
def layer_model(request, keras_layer_data):
    # tf.keras.Model.from_config mutates the config dict it is given, which
    # corrupts the shared, cached keras_layer_data for subsequent tests and
    # yields an unbuilt model that fails TFLite conversion. Pass a copy so each
    # test reconstructs from a pristine config.
    return tf.keras.Model.from_config(copy.deepcopy(keras_layer_data))


def keras_model_fixture(fun):
    @versioned_unhashable_object_fixture
    @wraps(fun)
    def wrapper(**kwargs):
        request = kwargs.get("request")
        if request is None:
            raise ValueError("versioned_compiler_input_fixture requires a request fixture")

        record_property = request.getfixturevalue("record_property")
        record_property("compiler_input", f"keras:{fun.__module__}.{fun.__qualname__}")

        return fun(**kwargs)

    return wrapper
