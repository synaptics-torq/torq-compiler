import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.tensorflow import generate_layers_from_model

import tensorflow as tf


def get_full_model():
    # make sure tests use reproducible weights
    tf.keras.utils.set_random_seed(21321)

    inputs = tf.keras.Input(shape=(320, 320, 3), batch_size=1)
    return tf.keras.applications.MobileNetV2(weights=None, input_tensor=inputs, include_top=False)


layers = generate_layers_from_model(get_full_model())

@pytest.fixture(params=layers.keys())
def case_config(request):
 
  if "bn" in request.param.lower():
    pytest.xfail()

  if request.param in ["block_1_pad", "block_3_pad", 
                        "block_1_depthwise", "block_3_depthwise",
                        "block_6_depthwise", "block_13_depthwise"]:
    pytest.xfail()

  return {
            "keras_model": "layer_model",
            "keras_layer_data": layers[request.param],
            "mlir_model_file": "tflite_mlir_model_file",
            "tflite_model_file": "quantized_tflite_model_file",
            "input_data": "tweaked_random_input_data"
        }


def test_model(request, tflite_reference_results, torq_results, case_config):
    compare_test_results(request, torq_results, tflite_reference_results, case_config)
