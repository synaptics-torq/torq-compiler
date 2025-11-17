from dataclasses import dataclass
import pytest
import tensorflow as tf
import numpy as np
import subprocess
import json

import iree.compiler.tflite as iree_tflite_compile

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
            print(f"Skipping duplicate model for layer {layer.name}")
            continue

        layer_configs[layer.name] = model_config

    print("Generated unique tests: ", len(existing_cases))

    return layer_configs


def quantize_model(model, quantize=True, quantize_to_int16=False):
    """
    Quantize a TensorFlow model using TFLite.    
    
    The model is quantized to 8-bit signed integers.

    :param model_function: A tf.function that represents the model
    :return: The quantized TFLite model
    """

    # Function to create a representative dataset for quantization
    def q_dataset(input_specs):

        rng = np.random.default_rng(1234)

        for _ in range(10):
            # Generate random input data within the expected range [0, 1]
            data = []

            for ts in input_specs[0]:
                shape = [1 if x is None else x for x in ts.shape]
                data.append(rng.random(shape, dtype=np.float32))

            yield data

    # Convert the model to a TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Quantize and convert the modely

    if quantize:
        # Set optimization to quantize the model
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Provide a representative dataset to calibrate the quantization
        converter.representative_dataset = lambda: q_dataset([model.inputs])
        converter.allow_custom_ops = True

        if quantize_to_int16:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
            converter._experimental_disable_per_channel = True
            converter.experimental_new_quantizer = True
        else:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

    return converter.convert()


@pytest.fixture
def keras_model(request, case_config):
    return request.getfixturevalue(case_config['keras_model'])


@pytest.fixture
def tflite_model_file(request, case_config):
    return request.getfixturevalue(case_config.get('tflite_model_file', 'quantized_tflite_model_file'))    


@versioned_generated_file_fixture("mlir")
def tflite_mlir_model_file(request, iree_opt, versioned_file, tflite_model_file):    

    mlirb_model_path = str(versioned_file) + "b"
    iree_tflite_compile.compile_file(str(tflite_model_file), save_temp_iree_input=str(mlirb_model_path), import_only=True)

    subprocess.check_call([iree_opt, str(mlirb_model_path), "-o", str(versioned_file)])


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

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model.data)    
    tflite_model = converter.convert()

    with open(versioned_file, "wb") as f:
        f.write(tflite_model)


@versioned_cached_data_fixture
def tflite_reference_results(request, tflite_model_file, input_data):

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i, input_data_array in enumerate(input_data):
        interpreter.set_tensor(input_details[i]['index'], input_data_array)

    interpreter.invoke()

    return [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))] 


@versioned_hashable_object_fixture
def keras_layer_data(case_config):
    return case_config['keras_layer_data']


@versioned_unhashable_object_fixture
def layer_model(request, keras_layer_data):
    return tf.keras.Model.from_config(keras_layer_data)