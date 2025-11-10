import tensorflow as tf
import numpy as np
import subprocess
import json
import abc

import iree.compiler.tflite as iree_tflite_compile

from .compilation_tests import CompilationTestCase, CachedTestDataFile
from .iree import IREE_OPT, WithSimpleComparisonToReference, WithTorqCompiler, WithTorqRuntime, WithRandomUniformIntegerInputData, WithCachedMlirModel


def generate_tests_from_model(full_model: tf.keras.Model, module_globals, marks):    
    """
    Generates test cases for each unique layer configuration in the given Keras model and for the whole model

    :param full_model: A tf.keras.Model instance representing the full model
    :param module_globals: The globals() dictionary of the module where the test cases will be added
    """

    class TestFullModel(WithRandomUniformIntegerInputData, TfliteTestCase):

        pytestmark = marks

        def model(self):
            return full_model

    module_globals["TestFullModel"] = TestFullModel

    existing_cases = set()

    print("Total layers in model: ", len(full_model.layers))

    for layer in full_model.layers:

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

        json_str = json.dumps(model.get_config(), sort_keys=True)

        # create one test case for each unique layer configuration
        if json_str not in existing_cases:
            existing_cases.add(json_str)
        else:
            print(f"Skipping duplicate model for layer {layer.name}")
            continue

        class TestLayer(WithRandomUniformIntegerInputData, TfliteTestCase):
            
            pytestmark = marks

            def model(self):
                return model            

        module_globals[f"Test{layer.name}"] = TestLayer

    print("Generated unique tests: ", len(existing_cases))


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


class WithTfliteToMlir(WithCachedMlirModel):
    """
    Converts a TFLite model to an MLIR model using IREE's TFLite importer
    """

    @abc.abstractmethod
    def generate_tflite_model(self):
        pass
    
    def generate_mlir_model(self, mlir_file_path):

        print("Generating MLIR model from TFLite model...")

        mlirb_model_path = self.cache_dir / "model.mlirb"

        iree_tflite_compile.compile_file(str(self.tflite_model_file), save_temp_iree_input=str(mlirb_model_path), import_only=True)

        subprocess.check_call([IREE_OPT, str(mlirb_model_path), "-o", str(mlir_file_path)])


class WithQuantizedTfliteExport(WithTfliteToMlir):
    """
    Exports the model as a quantized TFLite model (the model must be of type tf.keras.Model)
    """

    def quantize_to_int16(self):
        return False

    tflite_model_file = CachedTestDataFile("quantized_model.tflite", "generate_tflite_model")

    def generate_tflite_model(self, tflite_file_path):
        print("Generating quantized TFLite model...")        

        tflite_model = quantize_model(self.model(), quantize_to_int16=self.quantize_to_int16())

        with open(tflite_file_path, "wb") as f:
            f.write(tflite_model)


class WithTfliteReference:
    """
    Computes reference results for a test case using TFLite
    """ 

    def generate_reference_results(self):
        print("Generating TFLite reference results...")

        interpreter = tf.lite.Interpreter(model_path=str(self.tflite_model_file))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        for i, input_data_array in enumerate(self.input_data):
            interpreter.set_tensor(input_details[i]['index'], input_data_array)

        interpreter.invoke()

        return [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))] 


class TfliteTestCase(WithTorqCompiler, WithTorqRuntime, WithQuantizedTfliteExport, 
                     WithTfliteReference, WithSimpleComparisonToReference,
                     CompilationTestCase):
    """
    Test case that compares the torq output against TFLite reference output
    """

    pass
