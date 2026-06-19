# Input Guide

## Supported MLIR Inputs for `torq-compile`

The input type tells the compiler what kind of intermediate representation (IR) it's processing, such as TOSA or Torch dialects, and selects the appropriate frontend pipeline.

By default, `torq-compile` automatically detects the input type (`--iree-input-type=auto`), so you typically don't need to specify it. If auto-detection doesn't work for your model, you can explicitly override it with `--iree-input-type`:
- For TOSA MLIR: `--iree-input-type=tosa-torq`
- For Torch MLIR: `--iree-input-type=torch-torq`
- For ONNX MLIR: `--iree-input-type=onnx-torq`
- For Linalg MLIR: `--iree-input-type=linalg-torq`

For more options, run:
```shell
$ torq-compile --help
```

The Torq specific options can be retrieved by running:
```shell
$ torq-compile --help | grep torq
```

## Compile-Time Dtype Conversion

Use `--torq-convert-dtypes` when a model needs Torq compiler dtype conversion, such as converting FP32 tensors to BF16 for NSS execution. This is a compile-time choice; `torq-run-module` inputs must match the compiled model signature and are not reinterpreted into another dtype at runtime.

By default, Torq preserves the public input and output dtypes by inserting boundary casts where needed. Add `--torq-convert-io-dtype` only when the compiled model inputs and outputs should also use the converted dtypes.

```shell
$ torq-compile model.mlir -o model.vmfb --torq-convert-dtypes
```

```shell
$ torq-compile model.mlir -o model.vmfb --torq-convert-dtypes --torq-convert-io-dtype
```

For broader model selection and verification guidance, see {ref}`Model Selection and Verification <model-selection-and-verification>`.

---

## Input Structure for `torq-run-module`

This section explains how to provide inputs to the `torq-run-module` tool when running models. Match the expected shape, data type, and order of the model's inputs. Mismatches in input specifications, such as a wrong dtype or shape, will result in execution errors. Inputs can be provided directly as literals or loaded from `.npy` or raw `.bin` files.

- **Dtype:** Match the compiled model (e.g., `i8`, `i16`, `f32`). Mismatched dtypes will fail.
- **Shapes:** Use the exact input shapes expected by the model.

### Literal Inputs

- Pattern:  
  ```shell
  --input="<shape>x<dtype>=<value_or_list>"
  ```

- Example:  
  ```shell
  $ torq-run-module --module=model.vmfb --input="1x64x64x3xi8=0"
  ```

- List syntax (comma-separated) for tensors:  
  ```shell
  $ torq-run-module --module=model.vmfb --input="1x3xi8=[[1,2,3]]"
  ```

### Multiple Inputs

Order matters: pass inputs in the same order as the model’s signature.

- Example with two inputs (e.g., image tensor + scale):  
  ```shell
  $ torq-run-module --module=model.vmfb \
    --input="1x224x224x3xi32=0" \
    --input="1xi16=1"
  ```

### Feeding Input from Files

- **NPY (numpy arrays):** NumPy `.npy` files from `numpy.save`.
  ```shell
  $ torq-run-module --module=model.vmfb --input=@input.npy
  ```

- **Raw binary:** Raw binary files can be read to provide buffer contents.
  ```shell
  $ torq-run-module --module=model.vmfb \
    --input="1x224x224x3xf16=@input.bin"
  ```
  (Suppose `input.bin` holds float16 data for shape `1x224x224x3`)

(converting-images-to-npy-or-bin)=
### Converting Images to .npy or .bin

You can convert a PNG or JPEG image to `.npy` or `.bin` format using the provided tool.
Just run:

```shell
$ image_to_tensor.py --src=image.jpg --dst=input.npy --format=npy
```
or
```shell
$ image_to_tensor.py --src=image.png --dst=input.bin --format=rgb
```

- The script will automatically detect the image type and save the tensor in the desired format.
- Make sure to match the output shape and dtype with your model’s requirements.
- You can also specify the resize image size using the `--size` option (e.g., `--size=224x224`).
- `.npy` and `.npy_bgr` outputs include a batch dimension by default. Use `--no-batch` if your model expects an unbatched tensor.
- For more options, run:
  ```shell
  $ image_to_tensor.py --help
  ```

For more information, run:
```shell
  $ torq-run-module --help
  ```
