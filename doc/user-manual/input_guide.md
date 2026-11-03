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

## Preprocessing and Postprocessing Wrappers

Use `--torq-wrapper=wrapper.mlir` to replace the model's public entry points
with functions defined in a wrapper module. The wrapper declares the original
model function and can use Linalg operations, calls to defined helpers, external
CSS calls, or a combination of these for preprocessing and postprocessing.
Calls may have multiple inputs and outputs, and arguments can be passed through
unchanged or shared between calls.

### External Calls

For a model with signature `(tensor<4xbf16>) -> tensor<4xbf16>`:

```mlir
module {
  func.func private @model(tensor<4xbf16>) -> tensor<4xbf16>
  func.func private @normalize(tensor<4xi8>) -> tensor<4xbf16>
  func.func private @decode(tensor<4xbf16>) -> tensor<4xi32>

  func.func @wrapped(%input: tensor<4xi8>) -> tensor<4xi32> {
    %prepared = call @normalize(%input) : (tensor<4xi8>) -> tensor<4xbf16>
    %result = call @model(%prepared) : (tensor<4xbf16>) -> tensor<4xbf16>
    %output = call @decode(%result) : (tensor<4xbf16>) -> tensor<4xi32>
    return %output : tensor<4xi32>
  }
}
```

```shell
$ torq-compile model.mlir --torq-wrapper=wrapper.mlir \
  --torq-css-library=libprepost.a -o model.vmfb
```

The pass runs after Torq dtype conversion, before dispatch region creation and
ABI generation. The original model and its wrapper operations participate in normal
dispatch grouping together.
The model declaration must match the model signature at that point, including
any configured dtype conversions. Public wrapper definitions become the exported
entry points, and original model definitions become private. Wrapper definitions
must use names that do not collide with model symbols. The module may contain
only `func.func` declarations and definitions; recursive calls are not supported.

All original model entry points become private, including functions the wrapper
never calls. Unwrapped entry points are no longer exported in the VMFB. To keep
one available, define a public wrapper function that forwards to it.

Calls to declarations not present in the model become
`torq_hl.external_call` operations, with the declaration name identifying
the external CSS function. Model calls and calls to defined wrapper helpers are
inlined.

### Linalg Operations

A wrapper can implement its processing directly with `linalg.generic` and other
operations supported by the Torq input pipeline. No external declarations or
`--torq-css-library` option are needed. For a model with signature
`(tensor<4xi16>) -> tensor<4xi16>`, this wrapper adds one before the model and
doubles its result:

```mlir
module {
  func.func private @model(tensor<4xi16>) -> tensor<4xi16>

  func.func @wrapped(%input: tensor<4xi16>) -> tensor<4xi16> {
    %pre_empty = tensor.empty() : tensor<4xi16>
    %prepared = linalg.generic {
        indexing_maps = [affine_map<(index) -> (index)>,
                         affine_map<(index) -> (index)>],
        iterator_types = ["parallel"]}
        ins(%input : tensor<4xi16>) outs(%pre_empty : tensor<4xi16>) {
      ^bb0(%value: i16, %init: i16):
        %one = arith.constant 1 : i16
        %incremented = arith.addi %value, %one : i16
        linalg.yield %incremented : i16
    } -> tensor<4xi16>

    %result = call @model(%prepared) : (tensor<4xi16>) -> tensor<4xi16>

    %post_empty = tensor.empty() : tensor<4xi16>
    %output = linalg.generic {
        indexing_maps = [affine_map<(index) -> (index)>,
                         affine_map<(index) -> (index)>],
        iterator_types = ["parallel"]}
        ins(%result : tensor<4xi16>) outs(%post_empty : tensor<4xi16>) {
      ^bb0(%value: i16, %init: i16):
        %two = arith.constant 2 : i16
        %doubled = arith.muli %value, %two : i16
        linalg.yield %doubled : i16
    } -> tensor<4xi16>
    return %output : tensor<4xi16>
  }
}
```

```shell
$ torq-compile model.mlir --torq-wrapper=wrapper.mlir -o model.vmfb
```

The model is inlined into the wrapper before dispatch creation, allowing the
Linalg preprocessing and postprocessing to participate in normal dispatch
grouping, fusion, tiling, and kernel selection. Defined helper functions are
also inlined rather than converted to external calls. Wrapper bodies should
already use lowered operations such as Linalg, Tensor, and Arith; frontend
import and Torq dtype conversion have completed before wrapper insertion.

### Static CSS Libraries

Supply implementations in a RISC-V static archive using
`--torq-css-library=libprepost.a`. Repeat the option to link multiple archives.
These libraries are linked only into CSS programs, not host programs. Normal
archive extraction excludes unused members, and section garbage collection
removes unused sections from selected members. Compile library sources with
`-ffunction-sections -fdata-sections` to enable per-function and per-data removal.
The CSS code is linked with some libraries (including boostrap code and, on SL2610, 
libm), therefore it is better to use a custom prefix for all functions names 
to avoid collision with existing symbols.

External functions use the C ABI with a `void` return type. Pass one pointer
for each input, followed by one pointer for each result, in declaration order.
Inputs are read-only and the function must completely initialize all outputs.
There are no memref descriptors, shapes, strides, or scalar return values in
the call. For the external-call wrapper above, the C signatures are:

```c
void normalize(const signed char *input, unsigned short *output);
void decode(const unsigned short *input, int *output);
```

The `unsigned short` pointers here carry raw BF16 payloads; the compiler does
not convert the data to C floating-point values. C++ implementations must use
`extern "C"` to preserve the declared symbol names.

Build the archive for the selected CSS target: `SL2610` uses the `ilp32` ABI,
and `SR200` uses `ilp32f`. Query the compiler for the matching Clang target
flags without providing an input model:

```shell
$ torq-compile --torq-print-css-target-flags=SL2610
--target=riscv32-unknown-elf -march=rv32im -mabi=ilp32
```

The query accepts a named platform from `torq-compile --torq-hw=list` and exits
after printing a single line of flags. For example, build an SL2610 library with:

```shell
$ clang $(torq-compile --torq-print-css-target-flags=SL2610) \
  -ffreestanding -ffunction-sections -fdata-sections -O2 -c prepost.c -o prepost.o
$ llvm-ar rcs libprepost.a prepost.o
```

External calls execute once on CSS without LRAM/DTCM tiling. Arguments and
results must be statically shaped ranked tensors with byte-addressable integer
or floating-point elements. Bufferization packs non-identity input layouts into
dense row-major memrefs and allocates dense output buffers. Library functions
must fit the available CSS instruction and scratch memory; disabling CSS is an
error when external calls are present.

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
