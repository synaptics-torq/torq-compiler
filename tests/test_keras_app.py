import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.tensorflow import generate_layers_from_model
from torq.testing.iree import llvmcpu_reference_results
from torq.testing.cases import get_test_cases_from_tf_model

import tensorflow as tf


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


test_cases = [
    *get_model_cases("resnet50", (224, 224, 3)),
    # full_model_resnet50 working with differences: 26357 out of 100352 [26.26%]
    # *get_model_cases("resnet50", (224, 224, 3), full_model=True),

    # full model_vgg16 hang as some conv2d ops hang
    *get_model_cases("vgg16", (224, 224, 3)),

    # full model_xception hang/crash as some conv2d ops
    *get_model_cases("xception", (299, 299, 3)),

    # full model inceptionv3 hang/crash as some conv2d hang and avgpool ops crash
    *get_model_cases("inceptionv3", (299, 299, 3)),

    # full model_inceptionresnetv2(total 780 layers) hang/crash
    # as some conv2d hang and maxpool layer crash
    *get_model_cases("inceptionresnetv2", (299, 299, 3)),

    # full model_densenet121(total 427 layers) hang as some avgpool ops hang
    *get_model_cases("densenet121", (224, 224, 3)),

    # crash and hang because of many avgpool crash
    *get_model_cases("nasnetmobile", (224, 224, 3)),

    # full model crash (total 237 layers)
    *get_model_cases("efficientnetb0", (224, 224, 3)),

    # convnext serial model cannot work as many depthwise/pointwise ops crash
    # and mul op need to broadcast on two dims which unsupported now
    # (tensor<1x14x14x1xi32>, tensor<1x1x1x384xi32>) -> tensor<1x14x14x384xi32>
    # *get_model_cases("convnext_tiny", (224, 224, 3))
]


@pytest.fixture(params=test_cases)
def case_config(request, chip_config):

  failed_str = [
    # resnet50
    # skip cases that will be fused with other ops
    'resnet50_pool1_pad',

    # wrong results
    'resnet50_conv1_conv',
    'resnet50_pool1_pool',

    # vgg16
    # hang
    'vgg16_block1_conv1',
    'vgg16_block1_conv2',
    'vgg16_block2_conv1',
    'vgg16_block2_conv2',
    'vgg16_block3_conv1',
    'vgg16_block3_conv2',
    'vgg16_block4_conv1',
    'vgg16_block4_conv2',

    # xception
    # wrong results
    'xception_block1_conv1',
    'xception_block2_pool',
    'xception_block4_pool',
    'xception_conv2d_3',
    'xception_block13_pool',

    # hang
    'xception_block1_conv2',
    'xception_block2_sepconv1',
    'xception_block2_sepconv2',
    'xception_block3_sepconv1',
    'xception_block3_sepconv2',
    'xception_block4_sepconv1',
    'xception_block4_sepconv2',

    # crash
    'xception_conv2d',
    'xception_conv2d_2',

    # inceptionv3
    # wrong results
    'inceptionv3_conv2d_4',
    'inceptionv3_conv2d_30',
    'inceptionv3_conv2d_33',
    'inceptionv3_conv2d_75',
    'inceptionv3_conv2d_79',
    'inceptionv3_max_pooling2d',
    'inceptionv3_max_pooling2d_1',
    'inceptionv3_max_pooling2d_2',
    'inceptionv3_max_pooling2d_3',

    # compiler too long
    'inceptionv3_conv2d_6',
    'inceptionv3_conv2d_8',

    # crash
    'inceptionv3_average_pooling2d',
    'inceptionv3_average_pooling2d_1',
    'inceptionv3_average_pooling2d_2',
    'inceptionv3_average_pooling2d_3',
    'inceptionv3_average_pooling2d_7',
    'inceptionv3_average_pooling2d_8',

    # nss unsupported ops
    'bn',
    'batch_normalization',

    # inceptionresnetv2
    # wrong results
    'inceptionresnetv2_conv2d',
    'inceptionresnetv2_max_pooling2d',
    'inceptionresnetv2_max_pooling2d_1',
    'inceptionresnetv2_max_pooling2d_3',
    'inceptionresnetv2_conv2d_72',
    'inceptionresnetv2_conv2d_75',
    'inceptionresnetv2_max_pooling2d_2',
    'inceptionresnetv2_conv2d_157',
    'inceptionresnetv2_conv2d_159',
    'inceptionresnetv2_conv2d_162',

    # crash
    'inceptionresnetv2_conv2d_2',
    'inceptionresnetv2_conv2d_4',
    'inceptionresnetv2_average_pooling2d',

    # densenet121
    # wrong results
    'densenet121_conv1_conv',
    'densenet121_pool1',

    # skip cases that will be fused with other ops
    'densenet121_zero_padding2d_1',

    # hang
    'densenet121_pool2_pool',
    'densenet121_pool3_pool',
    'densenet121_pool4_pool',

    # nasnetmobile
    # wrong result
    'nasnetmobile_separable_conv_1',
    'nasnetmobile_reduction_left2_reduce',

    #crash
    'nasnetmobile_reduction_left',
    'nasnetmobile_adjust_avg_pool',
    'nasnetmobile_normal_left3',
    'nasnetmobile_separable_conv_2_reduction_left1_stem_2',
    'nasnetmobile_separable_conv_2_reduction_right1_stem_2',
    'nasnetmobile_separable_conv_2_reduction_right1_reduce_4', # error: matching error reduction loops > 0!
    'nasnetmobile_separable_conv_2_reduction_right1_reduce_8',

    # efficientnetb0
    # wrong results
    'efficientnetb0_stem_conv',
    'efficientnetb0_block2a_dwconv',
    'efficientnetb0_block2a_se_squeeze',
    'efficientnetb0_block2b_se_squeeze',
    'efficientnetb0_block3a_dwconv',
    'efficientnetb0_block3a_se_squeeze',
    'efficientnetb0_block3b_se_squeeze',
    'efficientnetb0_block4a_dwconv',
    'efficientnetb0_block6a_dwconv',

    # crash
    'dwconv_pad',
    'drop',
    'normalization',
    'se_squeeze',
    'se_reduce',
    'se_expand',
    'se_excite',
    'efficientnetb0_block2a_expand_activation',
    'efficientnetb0_block2a_expand_conv',
  ]
  if any(s in request.param.name.lower() for s in failed_str):
    pytest.xfail("failing test or skipped for now")

  if chip_config.data['target'] != "SL2610":
    failed_str += [
      # Assertion failed: (fused >= count && "Could not fuse the requested number of dimensions")
      'resnet50_conv3_block1_1_conv',
      'resnet50_conv3_block1_0_conv',
      'resnet50_conv4_block1_1_conv',
      'resnet50_conv4_block1_0_conv',
      'resnet50_conv5_block1_1_conv',
      'resnet50_conv5_block1_0_conv',
        # Compiler too long
      'xception_block13_sepconv2',
        # error: unable to free enough space for results and operands
      'nasnetmobile_zero_padding2d_2',
      # doesnt work on github ci for unknown reason
      'xception_block14_sepconv2',
    ]

  if any(s in request.param.name.lower() for s in failed_str):
    pytest.xfail("failing test or skipped for now")

  compile_timeout = 60 * 15
  runtime_timeout = 60 * 15

  if "layer_" in request.param.name:
    compile_timeout = 60 * 3
    runtime_timeout = 30

  longer_test_timeout = [
      'inceptionv3_conv2d_6',
      'inceptionv3_conv2d_8',
  ]
  if any(s in request.param.name.lower() for s in longer_test_timeout):
    compile_timeout = compile_timeout * 2
    runtime_timeout = runtime_timeout * 2

  keras_model = request.param.data

  return {
            "keras_model": "layer_model",
            "keras_layer_data": keras_model,
            "mlir_model_file": "tflite_mlir_model_file",
            "tflite_model_file": "quantized_tflite_model_file",
            "input_data": "tweaked_random_input_data",
            "quantize_to_int16": False,
            "torq_compiler_options": ["--torq-mul-cast-i32-to-i16"],
            "torq_compiler_timeout": compile_timeout,
            "torq_runtime_timeout": runtime_timeout
        }

@pytest.mark.ci
def test_keras_app_tflite_torq(request, tflite_reference_results, torq_results, case_config):
    compare_test_results(request, torq_results, tflite_reference_results, case_config)


def test_keras_app_llvmcpu_torq(request, llvmcpu_reference_results, torq_results, case_config):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)


def test_keras_app_llvmcpu_tflite(request, tflite_reference_results, llvmcpu_reference_results, case_config):
    compare_test_results(request, tflite_reference_results, llvmcpu_reference_results, case_config)
