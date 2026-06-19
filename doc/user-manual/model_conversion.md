# Model Conversion

## Overview

The Torq compiler compiles models expressed in MLIR, which is a generic framework for intermediate representations, into Torq VMFB files. To deploy a source model, first convert it to MLIR, then compile the MLIR to VMFB, then run the VMFB with the Torq runtime.

This page describes how to convert models from _TFLite_, _Torch_, and _ONNX_ to MLIR, compile them, and run them.

(model-selection-and-verification)=
## Model Selection and Verification

Before converting a model, check that both the model format and numeric type match the target path you plan to use.

- Prefer an already quantized INT8 model when targeting NPU-heavy image models. Verify the input and output shapes, dtypes, scales, and zero points with a model viewer such as Netron or with the source framework tooling.
- For FP32 models that need BF16 on NSS, use `torq-compile --torq-convert-dtypes`. Add `--torq-convert-io-dtype` when the public model inputs and outputs should also use converted dtypes; omit it when Torq should preserve the original public I/O dtypes with boundary casts.
- Check the operator and dtype coverage in [Supported Operations](./ops.md) before spending time debugging a model that uses unsupported operators.
- Keep a small set of representative inputs and reference outputs from the source framework. After conversion, compile the model, run the same inputs through Torq, and compare outputs with tolerances appropriate for INT8 or BF16 quantization.

For ONNX model exploration and layer-by-layer verification, use [torq-gen-config](./torq-gen-config.md). It can discover executor assignments, run layer and full-model checks, and report accuracy differences.

## Convert Model to MLIR Format

### Convert a TFLite model to TOSA and MLIR

A TFLite model must first be converted to TOSA MLIR before compilation. There are two approaches depending on which distribution you are using.

#### Using the Compiler Wheel

```{note}
Install the `[tflite]` extra to get the `tosa-converter-for-tflite` tool:
`pip install "torq_compiler-<version>-<platform>.whl[tflite]"`
```

Convert the model to TOSA bytecode:

```{code} shell
$ tosa-converter-for-tflite model.tflite --bytecode -o model.mlirbc
```

Use `--text` instead of `--bytecode` to produce a human-readable MLIR file:

```{code} shell
$ tosa-converter-for-tflite model.tflite --text -o model.mlir
```

#### Using the Release Package

- If not yet done, activate the Python environment as explained in [Getting Started](./getting_started.md) (skip this step if using the Docker container).

- Navigate to the root directory of the {ref}`Release Package <release-package-ubuntu-24-04>`, or run the {ref}`Docker container <docker-image>`.
  For the Docker container, the release package is located at:  
  ```
  $ cd /opt/release
  ```
- Convert the model to TOSA using the following command:
    
    **Model Source:** This model - MobileNetV2 is generated from tf.keras.applications using [tf_model_generator.py](https://github.com/synaptics-torq/iree-synaptics-synpu/blob/main/tests/model_generator/tf_model_generator.py). The dataset for int8 quantization is done using random data.

    ```{code} shell
    $ tosa-converter-for-tflite tests/hf/Synaptics_MobileNetV2/MobileNetV2_int8.tflite --text -o mobilenetv2.mlir
    ```

```{note}
The `tests/hf/` directory is only included in the {ref}`Release Package <release-package-ubuntu-24-04>` and is not available in the compiler GitHub repository.
```

(convert-torch-model-to-mlir)=
### Convert Torch Model to MLIR

- Torch models can be converted to MLIR using the `torch-mlir` toolchain.  
- This process involves exporting a PyTorch model and converting it to MLIR in various dialects such as TORCH, TOSA, LINALG_ON_TENSORS, or STABLEHLO.  
- The resulting MLIR file can then be used as input for the Torq compiler, depending on the supported dialects and features.

- Export torch-mlir Python Packages

    ```{code} shell
    $ export PYTHONPATH=`pwd`/build/tools/torch-mlir/python_packages/torch_mlir:`pwd`/test/python/fx_importer
    ```

- Create Torch Test Model and Output to different MLIR Dialect

    ```{code} python
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

### Convert ONNX Model to MLIR

IREE provides an ONNX importer that converts ONNX models into a text-based MLIR representation. The importer is available in the Torq compiler Python environment.

```{note}
**Compiler wheel users:** ONNX importing requires the `onnx` extra. Install with: `pip install "torq_compiler-<version>-<platform>.whl[onnx]"`
```

- If not using the Docker container, activate the Python environment as explained in [Getting Started](./getting_started.md).

- Run the IREE importer to convert an ONNX model to MLIR:

    ```{code} shell
    $ python -m iree.compiler.tools.import_onnx path/to/model.onnx -o path/to/model.mlir --data-prop
    ```

    ```{tip}
    Use the `-h` flag to view all available importer options.
    ```

    ```{code} shell
    $ python -m iree.compiler.tools.import_onnx -h
    ```

The output MLIR file can be compiled with Torq by specifying `onnx-torq` as the input dialect.

## Compile and Run

After converting a source model to MLIR, use [Step-by-Step Model Deployment Examples](./step_by_step_examples.md) for the detailed `torq-compile` and `torq-run-module` flow. That page covers input dialect selection, compiling MLIR to VMFB, running on Torq devices, and simulator-oriented builds.

See the [Input Guide](./input_guide.md) for literal inputs, multiple inputs, `.npy` files, raw binary files, and image conversion.

## References

- [torch-mlir development guide](https://github.com/llvm/torch-mlir/blob/main/docs/development.md)

- More details on the IREE Pytorch tools can be found in the 
[official IREE website](https://iree.dev/guides/ml-frameworks/pytorch/).
