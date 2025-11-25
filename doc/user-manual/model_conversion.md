# Model Preparation

## Overview

The Torq compiler can compile models expressed in MLIR, which is a generic framework for intermediate representations. To compile a model with the Torq compiler, it must first be converted to MLIR.

This describes how to convert models from _TFLite_ and _Torch_ to MLIR.

## Convert a TFLite model to TOSA and MLIR

IREE provides a command line tool that can be used to convert a _TFLite_ model to a binary MLIR file
expressed in the {term}`TOSA` dialect.


- If not yet done, activate python environment as explained in  [Getting Started](./getting_started.md) (skip this step if using the Docker container).

- Navigate to the root directory of the [Release Package](./getting_started.md#release-package-ubuntu-24-04), or run the [Docker container](./getting_started.md#docker-image).
  For the Docker container, release package is located at:  
  ```
  $ cd /opt/release
  ```
- Convert the model to TOSA use the following command:
    
    **Model Source:** This model - MobileNetV2 is generated from tf.keras.applications using [tf_model_generator.py](https://github.com/synaptics-torq/iree-synaptics-synpu/blob/main/tests/model_generator/tf_model_generator.py). The dataset for int8 quantization is done using random data.

    ```{code} shell
    $ iree-import-tflite samples/hf/Synaptics_MobileNetV2/MobileNetV2_int8.tflite -o mobilenetv2.tosa
    ```
> **Note:** The `samples/` directory is only included in the [Release Package](./getting_started.md#release-package-ubuntu-24-04) and is not available in the github compiler repository.


- More details on the IREE TFLite tools can be found in the 
[official IREE website](https://iree.dev/guides/ml-frameworks/tflite/).

    ```{tip}
    The generated *.tosa* file is a *binary* MLIR file. To convert it to a text representation, run:
    ```{code}shell
    $ iree-opt mobilenetv2.tosa -o mobilenetv2.mlir
    ```

## Convert Torch Model to MLIR

- Torch models can be converted to MLIR using the `torch-mlir` toolchain.  
- This process involves exporting a PyTorch model and converting it to MLIR in various dialects such as TORCH, TOSA, LINALG_ON_TENSORS, or STABLEHLO.  
- The resulting MLIR file can then be used as input for the Torq compiler, depending on the supported dialects and features.

- Export torch-mlir Python Packages

    ```{code}shell
    $ export PYTHONPATH=`pwd`/build/tools/torch-mlir/python_packages/torch_mlir:`pwd`/test/python/fx_importer
    ```

- Create Torch Test Model and Output to different MLIR Dialect

    ```{code}python
    import torch
    from torch_mlir import torchscript

    class SimpleModule(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.ops.aten.abs(x)


    if __name__ == '__main__':
        test_input = torch.ones(2, 16)
        graph = SimpleModule()
        graph.eval()
        module = torchscript.compile(graph,
                                    test_input,
                                    torchscript.OutputType.TORCH,
                                    use_tracing=False,
                                    verbose=False)
        print(module.operation.get_asm())

        with open("./aten-torch.mlir", "w") as fp:
            fp.write(module.operation.get_asm())
    ```

- output type could be
    - TORCH
    - LINALG_ON_TENSORS
    - TOSA
    - STABLEHLO

## References

- [torch-mlir development guide](https://github.com/llvm/torch-mlir/blob/main/docs/development.md)

- More details on the IREE Pytorch tools can be found in the 
[official IREE website](https://iree.dev/guides/ml-frameworks/pytorch/).