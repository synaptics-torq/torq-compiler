import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.tensorflow import generate_layers_from_model
from torq.testing.iree import llvmcpu_reference_results
from torq.testing.cases import Case, drop_duplicate_cases, get_test_cases_from_tf_model, keras_layer_key
from torq.testing.versioned_fixtures import collection_data, versioned_hashable_object_fixture

import tensorflow as tf

@versioned_hashable_object_fixture
def comparison_config_for_efficientnetb0():
    return {'int_tol': 9, 'int_thld': 9}

def get_model_cases(model_name, input_shape, include_top=False, full_model=False):
    cases = []
    tf.keras.utils.set_random_seed(21321)
    inputs = tf.keras.Input(shape=input_shape, batch_size=1)

    model_class = {
        "resnet50": tf.keras.applications.ResNet50,
        "vgg16": tf.keras.applications.VGG16,
        "xception": tf.keras.applications.Xception,
        "inceptionv3": tf.keras.applications.InceptionV3,
        "inceptionresnetv2": tf.keras.applications.InceptionResNetV2,
        "densenet121": tf.keras.applications.DenseNet121,
        "nasnetmobile": tf.keras.applications.NASNetMobile,
        "efficientnetb0": tf.keras.applications.EfficientNetB0,
        "convnext_tiny": tf.keras.applications.ConvNeXtTiny
    }.get(model_name.lower())

    if model_class is None:
        raise ValueError(f"Unsupported model name: {model_name}")

    model = model_class(weights=None, input_tensor=inputs, include_top=include_top)

    return get_test_cases_from_tf_model(model, model_name, full_model)


# models whose layers are tested: (name, input shape, test the full model instead of its layers)
MODELS = [
    ("resnet50", (224, 224, 3), False),
    # full_model_resnet50 working with differences: 26357 out of 100352 [26.26%]
    # ("resnet50", (224, 224, 3), True),

    # full model_vgg16 hang as some conv2d ops hang
    ("vgg16", (224, 224, 3), False),

    # full model_xception hang/crash as some conv2d ops
    ("xception", (299, 299, 3), False),

    # full model inceptionv3 hang/crash as some conv2d hang and avgpool ops crash
    ("inceptionv3", (299, 299, 3), False),

    # full model_inceptionresnetv2(total 780 layers) hang/crash
    # as some conv2d hang and maxpool layer crash
    ("inceptionresnetv2", (299, 299, 3), False),

    # full model_densenet121(total 427 layers) hang as some avgpool ops hang
    ("densenet121", (224, 224, 3), False),

    # crash and hang because of many avgpool crash
    ("nasnetmobile", (224, 224, 3), False),

    # full model crash (total 237 layers)
    ("efficientnetb0", (224, 224, 3), True),

    # convnext serial model cannot work as many depthwise/pointwise ops crash
    # and mul op need to broadcast on two dims which unsupported now
    # (tensor<1x14x14x1xi32>, tensor<1x1x1x384xi32>) -> tensor<1x14x14x384xi32>
    # ("convnext_tiny", (224, 224, 3), False),
]


def _compute_test_cases():
    # the names of the layers, part of the test names, are numbered per process (conv2d_6): count
    # from 0 whatever was built before
    tf.keras.backend.clear_session()
    cases = []
    for model_name, input_shape, full_model in MODELS:
        cases += get_model_cases(model_name, input_shape, full_model=full_model)
    # a layer with the same configuration and input shapes as an earlier one (in the same model or
    # another one) is the same test
    cases = drop_duplicate_cases(cases, keras_layer_key)
    return [{"name": c.name, "data": c.data} for c in cases]


# the cases of this process, computed once for all the test functions
_test_cases = None


def pytest_generate_tests(metafunc):
    global _test_cases
    if "case_config" not in metafunc.fixturenames:
        return
    if _test_cases is None:
        # building the models takes about 20 s: cached across sessions (see collection_data)
        _test_cases = [Case(c["name"], c["data"]) for c in
                       collection_data(metafunc, "keras_app_cases", MODELS, _compute_test_cases)]
    metafunc.parametrize("case_config", _test_cases, indirect=True)


@pytest.fixture
def case_config(request, runtime_hw_type, chip_config):

  extra_args = {}
  if "full_model_efficientnetb0" in request.param.name.lower():
    extra_args["comparison_config"] = "comparison_config_for_efficientnetb0"

  if request.param.name.lower() == "layer_resnet50_conv3_block1_3_conv" and chip_config.data.get("target") == "SL2610":
    # Tiling this layer splits the large conv inputs into 64-element chunks
    # addressed through chained subviews; the buffer then no longer fits
    # alongside its neighbors, so the LRAM plan swaps it out and back in.
    # Regression for the fatal "LRAM plan: a view is used while its buffer is
    # swapped out" (synaptics-torq/torq-compiler-dev#2697).
    extra_args["torq_compiler_options"] = [
        "--torq-hl-tile-input-size-threshold=40000",
        "--torq-hl-tile-dim-size=64",
    ]

  aws_fpga = (runtime_hw_type.data == "aws_fpga")
  if aws_fpga:
    # Only run full model tests on AWS FPGA
    if "full_model" not in request.param.name.lower():
        pytest.skip("AWS FPGA only runs full model tests")

  compile_timeout = 60 * 10
  runtime_timeout = 60 * 10

  if "layer_" in request.param.name:
    compile_timeout = 60 * 5
    runtime_timeout = 60

  longer_test_timeout = [
      'inceptionv3_conv2d_6',
      'inceptionv3_conv2d_8',
      'layer_vgg16_block5_conv1',
      # Big VGG16 conv layers: the 1.9-2.5 GMAC block*_conv2 3x3 convs take
      # ~36-39s of single-threaded sim alone, leaving no headroom for the 60s
      # layer budget when CI's 18 workers oversubscribe the 16-vCPU runner.
      # The ~0.95 GMAC block*_conv1 convs take ~16-18s: same no-margin class.
      'layer_vgg16_block1_conv2',
      'layer_vgg16_block2_conv2',
      'layer_vgg16_block3_conv2',
      'xception_block2_sepconv2',
      'vgg16_block2_conv1',
      'vgg16_block2_conv2',
      'vgg16_block1_conv2',
      'vgg16_block3_conv1',
      'vgg16_block3_conv2',
      'vgg16_block4_conv2',
      'inceptionv3_conv2d_30',
      'inceptionresnetv2_conv2d_173',
      'inceptionresnetv2_conv2d_170',
      'resnet50_conv3_block1_1_conv',
      'resnet50_conv3_block1_0_conv',
      'xception_conv2d',
  ]
  if any(s in request.param.name.lower() for s in longer_test_timeout):
    compile_timeout = compile_timeout * 2
    runtime_timeout = runtime_timeout * 3

  keras_model = request.param.data

  return {
            "keras_model": "layer_model",
            "keras_layer_data": keras_model,
            "mlir_model_file": "tflite_mlir_model_file",
            "tflite_model_file": "quantized_tflite_model_file",
            "input_data": "tweaked_random_input_data",
            "quantize_to_int16": False,
            "torq_compiler_timeout": compile_timeout,
            "torq_runtime_timeout": runtime_timeout,
            **extra_args
        }

@pytest.mark.ci
def test_keras_app_tflite_torq(request, tflite_reference_results, torq_results, case_config):
    compare_test_results(request, torq_results, tflite_reference_results, case_config)


def test_keras_app_llvmcpu_torq(request, llvmcpu_reference_results, torq_results, case_config):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)


def test_keras_app_llvmcpu_tflite(request, tflite_reference_results, llvmcpu_reference_results, case_config):
    compare_test_results(request, tflite_reference_results, llvmcpu_reference_results, case_config)
