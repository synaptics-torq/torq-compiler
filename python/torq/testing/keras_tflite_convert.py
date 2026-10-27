"""Keras to TFLite conversion, in this process or in a child process.

TensorFlow's TFLite converter and IREE's compiler each bundle their own copy of MLIR. On macOS,
dyld merges the weak symbols of all the loaded libraries, among them the TypeIDs of MLIR types
and attributes, so whichever copy loads first serves both and the other one aborts:
- IREE loaded first: the converter aborts ("can't create Attribute 'mlir::StringAttr' because
  storage uniquer isn't initialized")
- TensorFlow loaded first: IREE aborts at its next MLIR context ("Trying to create a Type that
  was not registered in this MLIRContext")

torq.testing loads IREE first, so on macOS the conversion runs in a child process that never
imports IREE. The child gets the model as a SavedModel, which doesn't need the Python code of the
model's layers, and converts it with the same settings and calibration data as in process.

The child runs this file as a script: it must not import torq.testing, which imports IREE.
"""

import json
import sys

import numpy as np

# Calibration data for the quantization: random values in [0, 1)
CALIBRATION_SEED = 1234
CALIBRATION_SAMPLES = 10


def calibration_samples(input_shapes):
    """Yield the calibration samples, a list with one array per input, in the order of the inputs."""
    rng = np.random.default_rng(CALIBRATION_SEED)
    for _ in range(CALIBRATION_SAMPLES):
        yield [rng.random(shape, dtype=np.float32) for shape in input_shapes]


def configure_converter(converter, quantize, quantize_to_int16, representative_dataset):
    """Set the quantization options of a TFLite converter."""
    import tensorflow as tf

    if not quantize:
        return

    # Set optimization to quantize the model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Provide a representative dataset to calibrate the quantization
    converter.representative_dataset = representative_dataset
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


def _convert_saved_model(saved_model_dir, output_path, config):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # The samples are made in the order of the Keras model inputs and passed by input name, which
    # doesn't depend on the order of the inputs in the SavedModel signature
    def representative_dataset():
        for sample in calibration_samples(config["input_shapes"]):
            yield dict(zip(config["input_names"], sample))

    configure_converter(
        converter, config["quantize"], config["quantize_to_int16"], representative_dataset
    )
    with open(output_path, "wb") as f:
        f.write(converter.convert())


if __name__ == "__main__":
    # Usage: keras_tflite_convert.py <saved model dir> <output .tflite> <config json>
    with open(sys.argv[3]) as f:
        _convert_saved_model(sys.argv[1], sys.argv[2], json.load(f))
