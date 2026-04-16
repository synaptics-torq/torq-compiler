# Torq Runtime Python API (Beta)

The `torq-runtime` Python package provides bindings for loading and running compiled `.vmfb` models on a Torq device directly from Python.

```{warning}
`torq-runtime` is currently in beta and is not yet available on PyPI.
```

## Installation

The `torq-runtime` package is included in the GitHub release. Install the runtime wheel directly from any release snapshot.

## Quick Start

```python
import numpy as np
from torq.runtime import VMFBInferenceRunner

# Load the compiled model
runner = VMFBInferenceRunner("mobilenetv2.vmfb", device_uri="torq")

# Prepare input data
input_data = np.random.randint(0, 255, size=(1, 224, 224, 3), dtype=np.int8)

# Run inference
outputs = runner.infer([input_data])
print(f"Inference took {runner.infer_time_ms:.2f} ms")
```

## API Reference

### `VMFBInferenceRunner`

The main class for loading and running `.vmfb` models via the IREE runtime.

```python
VMFBInferenceRunner(
    model_path,
    *,
    function="main",
    device_uri="torq",
    n_threads=None,
    load_method="preload",
    load_model_to_mem=True,
    runtime_flags=None,
    device_outputs=False,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str \| PathLike` | *(required)* | Path to the `.vmfb` file. |
| `function` | `str` | `"main"` | Exported function name inside the module. |
| `device_uri` | `str` | `"torq"` | IREE device identifier. |
| `n_threads` | `int \| None` | `None` | Worker thread count (only for llvm-cpu device). |
| `load_method` | `"preload" \| "mmap"` | `"preload"` | `"preload"` copies into memory; `"mmap"` memory-maps the file. |
| `load_model_to_mem` | `bool` | `True` | Whether to load the model into memory during initialization. |
| `runtime_flags` | `Iterable[str] \| None` | `None` | Extra IREE runtime flags. |
| `device_outputs` | `bool` | `False` | If `True`, `infer()` returns on-device `DeviceArray` objects instead of NumPy arrays, avoiding device-to-host transfers. Useful for pipelines where outputs are fed back as inputs (e.g. KV-cache in autoregressive decoding). |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `model_path` | `PathLike` | Path to the loaded model file. |
| `infer_time_ms` | `float` | Elapsed time in milliseconds for the last call to `infer()`. |
| `device` | `HalDevice` | The underlying IREE HAL device. |
| `inputs_info` | `list[TensorInfo] \| None` | Input tensor metadata extracted from the model, or `None` if unavailable. |
| `outputs_info` | `list[TensorInfo] \| None` | Output tensor metadata extracted from the model, or `None` if unavailable. |

**Methods:**

#### `infer(inputs)`

Run inference and return the output arrays.

- **inputs** — Either an iterable of NumPy arrays (or `DeviceArray` objects) or a mapping of name to array.
- **Returns** — A list of NumPy arrays by default. When `device_outputs=True`, returns a list of on-device `DeviceArray` objects instead.

#### `allocate_device_array(array)`

Allocate a device buffer and copy a host NumPy array into it.

- **array** — A NumPy array to upload to the device.
- **Returns** — A `DeviceArray` that can be passed directly to `infer()` without further copies.

### `profile_vmfb_inference_time`

Load a `.vmfb` model and run inference multiple times for profiling.

```python
profile_vmfb_inference_time(
    model_path,
    inputs=None,
    *,
    n_iters=5,
    do_warmup=True,
    function="main",
    device="torq",
    n_threads=None,
    load_model_to_mem=True,
    runtime_flags=None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str \| PathLike` | *(required)* | Path to the `.vmfb` file. |
| `inputs` | `Iterable[NDArray] \| None` | `None` | Input arrays. Generated randomly from model metadata when `None`. |
| `n_iters` | `int` | `5` | Number of timed inference iterations. |
| `do_warmup` | `bool` | `True` | Whether to run one untimed warmup pass first. |
| `function` | `str` | `"main"` | Exported function name inside the module. |
| `device` | `str` | `"torq"` | IREE device URI. |
| `n_threads` | `int \| None` | `None` | Worker thread count (only for llvm-cpu device). |
| `load_model_to_mem` | `bool` | `True` | Whether to load the model into memory during initialization. |
| `runtime_flags` | `Iterable[str] \| None` | `None` | Extra IREE runtime flags. |

**Returns:** Average wall-clock inference time in milliseconds.

### `run_vmfb`

Run a `.vmfb` model via the `iree-run-module` CLI and return wall-clock time.

```python
run_vmfb(
    model_path,
    inputs,
    outputs,
    device="torq",
    n_threads=None,
    iree_binary=None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str \| PathLike` | *(required)* | Path to the `.vmfb` file. |
| `inputs` | `Iterable[str]` | *(required)* | Input descriptors forwarded as `--input` flags. |
| `outputs` | `Iterable[str]` | *(required)* | Output descriptors forwarded as `--output` flags. |
| `device` | `str` | `"torq"` | IREE device URI. |
| `n_threads` | `int \| None` | `None` | Worker thread count (only for llvm-cpu device, defaults to `os.cpu_count()`). |
| `iree_binary` | `str \| PathLike \| None` | `None` | Path to the `iree-run-module` binary. Resolved from `PATH` if not provided. |

**Returns:** Elapsed wall-clock time in milliseconds.

### `TensorInfo`

Dataclass holding dtype and shape metadata for a tensor.

```python
@dataclass
class TensorInfo:
    dtype: DTypeLike
    shape: list[int | str]
```

| Field | Type | Description |
|-------|------|-------------|
| `dtype` | `DTypeLike` | NumPy-compatible dtype. |
| `shape` | `list[int \| str]` | Tensor dimensions. |

**Methods:**

- `is_valid()` — Returns `True` if every dimension is an integer (i.e., no dynamic dimensions).

### Utility Functions

#### `random_inputs_from_info(inputs_info)`

Generate random NumPy arrays matching the given tensor metadata. Useful for testing.

- **inputs_info** — Iterable of `TensorInfo`.
- **Returns** — List of NumPy arrays with appropriate shapes and dtypes.

## Examples

### Inspecting Model Inputs and Outputs

```python
from torq.runtime import VMFBInferenceRunner

runner = VMFBInferenceRunner("model.vmfb", device_uri="torq")

if runner.inputs_info:
    for i, info in enumerate(runner.inputs_info):
        print(f"Input {i}: dtype={info.dtype}, shape={info.shape}")

if runner.outputs_info:
    for i, info in enumerate(runner.outputs_info):
        print(f"Output {i}: dtype={info.dtype}, shape={info.shape}")
```

### Profiling Inference Latency

```python
from torq.runtime import profile_vmfb_inference_time

avg_ms = profile_vmfb_inference_time(
    "model.vmfb",
    n_iters=10,
    do_warmup=True,
    device="torq",
)
print(f"Average inference time: {avg_ms:.2f} ms")
```

### Running with Custom Inputs

```python
import numpy as np
from torq.runtime import VMFBInferenceRunner

runner = VMFBInferenceRunner("model.vmfb", device_uri="torq")

# Load preprocessed input from a .npy file
input_data = np.load("preprocessed_input.npy")
outputs = runner.infer([input_data])
```

### Zero-Copy Device Outputs

When model outputs are fed back as inputs (e.g. KV-cache tensors in autoregressive decoding), use `device_outputs=True` to keep data on the device and avoid unnecessary host transfers:

```python
import numpy as np
from torq.runtime import VMFBInferenceRunner

runner = VMFBInferenceRunner("model.vmfb", device_uri="torq", device_outputs=True)

# Allocate an initial on-device buffer (e.g. for a KV cache)
kv_cache = runner.allocate_device_array(np.zeros((1, 8, 256, 64), dtype=np.float16))

# Run inference — outputs stay on device as DeviceArray objects
results = runner.infer([input_data, kv_cache])
logits, new_kv = results[0], results[1]

# Use the output DeviceArray directly as next step's input
kv_cache = new_kv

# Only bring the logits to host for sampling
logits_np = logits.to_host()
```
```
