#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
import subprocess


def quantize_model(model_function, quantize=True):
    """
    Quantize a TensorFlow model using TFLite.
    The model is quantized to 8-bit signed integers.
    :param model_function: A tf.function that represents the model
    :return: The quantized TFLite model
    """

    # Function to create a representative dataset for quantization
    def q_dataset(input_specs):
        for _ in range(10):
            # Generate random input data within the expected range [0, 1]
            data = [np.random.rand(*ts.shape).astype(np.float32 if ts.dtype==tf.float32 else np.int32) for ts in input_specs[0]]
            yield data

    # Get the concrete function from the tf.function
    concrete_fn = model_function.get_concrete_function()

    # Convert the model to a TFLite model
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    
    # Quantize and convert the model


    if (quantize):
        # Set optimization to quantize the model
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Provide a representative dataset to calibrate the quantization
        converter.representative_dataset =  lambda : q_dataset(concrete_fn.structured_input_signature)

        # Specify that we want to use only integer operations in the converted model
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        # Set the input and output tensors to uint8
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        converter.allow_custom_ops = True

        # Convert the model to quantized tflite
    tflite_quantized_model = converter.convert()

    return tflite_quantized_model

def gen_mlir(model_path):

  tosa_file = str(model_path).replace(".tflite", ".tosa")
  tosa_mlir = str(model_path).replace(".tflite", ".tosa.mlir")

  cmd = ["iree-import-tflite", model_path, "-o", tosa_file]
  subprocess.run(cmd, capture_output=True, text=True)

  cmd = ["iree-opt", tosa_file, "-o", tosa_mlir]
  subprocess.run(cmd, capture_output=True, text=True)

  # remove tosa bin file
  subprocess.run(["rm -rf " + tosa_file], shell=True)

  # replace mlir dynamic batch_size with 1
  content = Path(tosa_mlir).read_text()
  new_content = content.replace("?", "1")
  Path(tosa_mlir).write_text(new_content)

  print(f"\nTOSA MLIR file created successfully:", tosa_mlir)

  # generate linalg op
  linalg_mlir = str(model_path).replace(".tflite", ".linalg.mlir")

  pipeline_str = "--iree-tosa-input-transformation-pipeline"

  cmd = ["iree-opt", tosa_mlir, pipeline_str, "-o", linalg_mlir]

  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    print(result.stderr)
    exit(1)
  else:
    print(f"\nLinalg MLIR file created successfully:", linalg_mlir)



# Define each model as a tf.function with the desired inputs and operations

# ADD
@tf.function(input_signature=[
    tf.TensorSpec(shape=[2, 3], dtype=tf.float32),
    tf.TensorSpec(shape=[2, 3], dtype=tf.float32)]
)
def model_definition_add(input1, input2):
    return tf.add(input1, input2)

# REDUCE_MAX
@tf.function(input_signature=[
    tf.TensorSpec(shape=[3, 4], dtype=tf.float32)]
)
def model_definition_reduce_max(input):
    return tf.reduce_max(input, axis=0)

# PW
conv_layer = tf.keras.layers.Conv2D(1280, (1, 1), padding="same")
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 7, 7, 320], dtype=tf.float32),
])
def model_definition_pw(input):
    return conv_layer(input)

# MAXPOOL2D
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 8, 8, 32], dtype=tf.float32)]
)

def model_definition_maxpool_2d(input):
    return tf.nn.max_pool2d(input, ksize=2, strides=2, padding='VALID')

# GATHER
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 256, 32], dtype=tf.float32),
    tf.TensorSpec(shape=[1, 16], dtype=tf.int32)
    ]
)

def model_definition_gather(values, indices):
    return tf.gather(values, indices, axis=1, batch_dims=1)

# CONV
conv_case = tf.keras.layers.Conv2D(24, (3, 3), padding="valid")
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 224, 224, 24], dtype=tf.float32),
])
def model_definition_conv(input):
    return conv_case(input)

# CONV KERNEL SIZE 4, STRIDE 4, SAME
conv_case_stride = tf.keras.layers.Conv2D(64, (4, 4), padding="same", strides=4, name="conv4")
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 256, 256, 1], dtype=tf.float32),
])
def model_definition_conv4(input):
    return conv_case_stride(input)


# DEPTHTOSPACE
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 8, 8, 4], dtype=tf.float32),
])
def model_definition_depthtospace(input):
    depth_to_space = tf.nn.depth_to_space(input, 2)
    return depth_to_space

# SPACETODEPTH
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 16, 16, 1], dtype=tf.float32),
])
def model_definition_spacetodepth(input):
    depth_to_space = tf.nn.space_to_depth(input, 2)
    return depth_to_space

#SCATTER
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 256, 32], dtype=tf.float32),
    tf.TensorSpec(shape=[1, 16, 3], dtype=tf.int32),
    tf.TensorSpec(shape=[1, 16], dtype=tf.float32)
    ]
)
def model_definition_scatter(values_in, indices, input):
    return tf.keras.ops.scatter_update(values_in, indices, input)

model_catalog = {
    "add": model_definition_add,
    "reduce_max": model_definition_reduce_max,
    "pw": model_definition_pw,
    "maxpool2d": model_definition_maxpool_2d,
    "gather" : model_definition_gather,
    "conv" : model_definition_conv,
    "conv4" : model_definition_conv4,
    "scatter" : model_definition_scatter,
    "depthtospace": model_definition_depthtospace,
    "spacetodepth": model_definition_spacetodepth
}


def main():
    """
    generate quantized TFLite model and TOSA/LINALG MLIR file
    cmd example: python3 gen-model.py -m add -o add.tflite
    it will generate add.tflite, add.tosa.mlir and add.linalg.mlir
    """
    # Parse argument for output file Name
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="add", choices=model_catalog.keys(), help="Model type")
    parser.add_argument("-o", "--output_file", type=str, default="model.tflite", help="Name of the output file")
    parser.add_argument("-q", "--quantize_model", type=int, default=1, help="Whether quantize the model")
    args = parser.parse_args()

    tflite_quantized_model = quantize_model(model_catalog[args.model], args.quantize_model)

    with open(args.output_file, "wb") as f:
        f.write(tflite_quantized_model)

    print("Quantize model:", args.quantize_model)
    print(f"\nTFLite {args.model.upper()} model created successfully:", args.output_file)


    gen_mlir(args.output_file)


if __name__ == "__main__":
    main()
