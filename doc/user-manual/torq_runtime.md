# Torq Runtime

The Torq runtime provides the means for loading and running compiled `.vmfb` models on a Torq device. It is built on top of [IREE](https://iree.dev)'s own runtime: the `torq-runtime` Python package wraps IREE's Python bindings, and C/C++ applications use IREE's public runtime API together with a Torq-specific IREE HAL driver. Pick the section below that matches your language.

## Python API

The `torq-runtime` Python package provides bindings for loading and running compiled `.vmfb` models on a Torq device directly from Python.

### Installation

`torq-runtime` is published on [PyPI](https://pypi.org/project/torq-runtime/):

```bash
$ pip install torq-runtime
```

### Quick Start

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

### Examples

#### Inspecting Model Inputs and Outputs

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

#### Profiling Inference Latency

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

#### Profiling System Resources

```python
from torq.runtime import profile_vmfb_resources

stats = profile_vmfb_resources(
    "model.vmfb",
    n_iters=10,
    do_warmup=True,
    device="torq",
)
print(f"Average inference time: {stats.avg_inference_time_ms:.2f} ms")
print(f"Average DRAM footprint: {stats.avg_dram_footprint_bytes / 1024 / 1024:.1f} MB")
print(f"Peak DRAM footprint: {stats.peak_dram_footprint_bytes / 1024 / 1024:.1f} MB")
print(f"Average memory usage: {stats.avg_anon_mem_bytes / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {stats.peak_anon_mem_bytes / 1024 / 1024:.1f} MB")
print(f"Average CPU usage: {stats.avg_cpu_percent:.1f}%")
if stats.avg_npu_load_percent is not None:
    print(f"Average NPU usage: {stats.avg_npu_load_percent:.1f}%")
```

#### Running with Custom Inputs

```python
import numpy as np
from torq.runtime import VMFBInferenceRunner

runner = VMFBInferenceRunner("model.vmfb", device_uri="torq")

# Load preprocessed input from a .npy file
input_data = np.load("preprocessed_input.npy")
outputs = runner.infer([input_data])
```

#### Zero-Copy Device Outputs

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

### API Reference

#### `VMFBInferenceRunner`

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
    torq_hw_type="astra_machina",
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
| `torq_hw_type` | `str` | `"astra_machina"` | Torq HW backend for the `torq` device. Defaults to `astra_machina` (the SoC NPU). Other backends such as `sim` (x86 CModel simulator) or `aws_fpga` must be set explicitly, and must be compiled into the installed runtime wheel. |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `model_path` | `PathLike` | Path to the loaded model file. |
| `infer_time_ms` | `float` | Elapsed time in milliseconds for the last call to `infer()`. |
| `device` | `HalDevice` | The underlying IREE HAL device. |
| `function_names` | `list[str]` | Exported VMFB function names discovered after the module is loaded. Defaults to an empty list. |
| `inputs_info` | `list[TensorInfo] \| None` | Input tensor metadata extracted from the model, or `None` if unavailable. |
| `outputs_info` | `list[TensorInfo] \| None` | Output tensor metadata extracted from the model, or `None` if unavailable. |

**Methods:**

##### `infer(inputs)`

Run inference and return the output arrays.

- **inputs** — Either an iterable of NumPy arrays (or `DeviceArray` objects) or a mapping of name to array.
- **Returns** — A list of NumPy arrays by default. When `device_outputs=True`, returns a list of on-device `DeviceArray` objects instead.

##### `allocate_device_array(array)`

Allocate a device buffer and copy a host NumPy array into it.

- **array** — A NumPy array to upload to the device.
- **Returns** — A `DeviceArray` that can be passed directly to `infer()` without further copies.

#### `profile_vmfb_inference_time`

Load or reuse a `.vmfb` model and run inference multiple times for profiling.

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
    device_io=False,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str \| PathLike \| VMFBInferenceRunner` | *(required)* | Path to the `.vmfb` file or an existing runner. |
| `inputs` | `Iterable[NDArray] \| None` | `None` | Input arrays. Generated randomly from model metadata when `None`. |
| `n_iters` | `int` | `5` | Number of timed inference iterations. |
| `do_warmup` | `bool` | `True` | Whether to run one untimed warmup pass first. |
| `function` | `str` | `"main"` | Exported function name inside the module. |
| `device` | `str` | `"torq"` | IREE device URI. |
| `n_threads` | `int \| None` | `None` | Worker thread count (only for llvm-cpu device). |
| `load_method` | `"preload" \| "mmap"` | `"preload"` | `"preload"` copies into memory; `"mmap"` memory-maps the file. |
| `load_model_to_mem` | `bool` | `True` | Whether to load the model into memory during initialization. |
| `runtime_flags` | `Iterable[str] \| None` | `None` | Extra IREE runtime flags. |
| `device_io` | `bool` | `False` | Exclude some host copies from profiling by using iree DeviceArray objects for I/O. |

**Returns:** Average wall-clock inference time in milliseconds.

#### `profile_vmfb_resources`

Load or reuse a `.vmfb` model and run inference multiple times, collecting DRAM, CPU, and Torq NPU statistics alongside timing. Requires a Linux target for process statistics (reads from `/proc`). NPU load is reported when `/sys/class/misc/torq/statistics/torq_inference_time` is available.

```python
from torq.runtime import profile_vmfb_resources

profile_vmfb_resources(
    model_path,
    inputs=None,
    *,
    n_iters=5,
    do_warmup=True,
    function="main",
    device="torq",
    n_threads=None,
    load_method="preload",
    load_model_to_mem=True,
    runtime_flags=None,
    device_io=False,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str \| PathLike \| VMFBInferenceRunner` | *(required)* | Path to the `.vmfb` file or an existing runner. |
| `inputs` | `Iterable[NDArray] \| None` | `None` | Input arrays. Generated randomly from model metadata when `None`. |
| `n_iters` | `int` | `5` | Number of timed inference iterations. |
| `do_warmup` | `bool` | `True` | Whether to run one untimed warmup pass first. |
| `function` | `str` | `"main"` | Exported function name inside the module. |
| `device` | `str` | `"torq"` | IREE device URI. |
| `n_threads` | `int \| None` | `None` | Worker thread count (only for llvm-cpu device). |
| `load_method` | `"preload" \| "mmap"` | `"preload"` | `"preload"` copies into memory; `"mmap"` memory-maps the file. |
| `load_model_to_mem` | `bool` | `True` | Whether to load the model into memory during initialization. |
| `runtime_flags` | `Iterable[str] \| None` | `None` | Extra IREE runtime flags. |
| `device_io` | `bool` | `False` | Exclude some host copies from profiling by using iree DeviceArray objects for I/O. |

**Returns:** A `ProfileStats` dataclass (see below).

#### `ProfileStats`

Dataclass returned by `profile_vmfb_resources` containing profiling results. All resource metrics are **process-wide** and include overhead from the Python interpreter, the sampling thread, and any other activity in the process.

```python
from torq.runtime.profiling import ProfileStats
```

| Field | Type | Description |
|-------|------|-------------|
| `avg_inference_time_ms` | `float` | Average wall-clock inference time in milliseconds. |
| `avg_dram_footprint_bytes` | `int` | Average total DRAM footprint in bytes, including file-backed pages (e.g. mmap'd model data). |
| `peak_dram_footprint_bytes` | `int` | Peak total DRAM footprint in bytes. |
| `avg_anon_mem_bytes` | `int` | Average anonymous memory (heap/stack) in bytes, excluding file-backed pages. |
| `peak_anon_mem_bytes` | `int` | Peak anonymous memory in bytes. |
| `avg_cpu_percent` | `float` | Average CPU utilisation as a percentage (across all cores) during timed iterations. |
| `npu_inference_time_us` | `int \| None` | Torq NPU execution time accumulated during timed iterations, or `None` when unavailable. |
| `avg_npu_load_percent` | `float \| None` | Average Torq NPU utilisation during timed iterations, or `None` when unavailable. |

**Methods:**

##### `summary()`

Return a human-readable summary string with formatted values (times in ms, memory in MB, CPU as a percentage).

```python
stats = profile_vmfb_resources("model.vmfb")
print(stats.summary())
# Avg inference time:  12.345 ms
#
# Process-wide resource usage (includes Python overhead):
#   Avg DRAM footprint:  85.2 MB  (includes mmap'd model file)
#   Peak DRAM footprint: 91.7 MB
#   Avg memory usage:    23.4 MB  (heap/stack only, excludes model file)
#   Peak memory usage:   28.1 MB
#   Avg CPU usage:       47.3%
#   Avg NPU usage:       82.4%
```

#### `ResourceSampler`

A reusable background sampler for collecting process-wide DRAM and CPU metrics plus optional Torq NPU load around arbitrary workloads. Used internally by `profile_vmfb_resources` but available for custom profiling scenarios.

```python
from torq.runtime.profiling import ResourceSampler
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pid` | `int` | *(required)* | Process ID to monitor. |
| `interval` | `float` | `0.01` | Sampling interval in seconds. |
| `npu_inference_time_path` | `str \| PathLike \| None` | Torq sysfs path | Path to the Torq cumulative NPU inference-time counter, or `None` to disable NPU load reporting. |

**Properties (available after `stop()`):**

| Property | Type | Description |
|----------|------|-------------|
| `avg_rss` | `int` | Average total RSS (DRAM footprint) in bytes. |
| `peak_rss` | `int` | Peak total RSS in bytes. |
| `avg_anon_rss` | `int` | Average anonymous RSS in bytes. |
| `peak_anon_rss` | `int` | Peak anonymous RSS in bytes. |
| `avg_cpu_percent` | `float` | Average CPU utilisation percentage. |
| `npu_inference_time_us` | `int \| None` | Torq NPU execution time accumulated while the sampler was running, or `None` when unavailable. |
| `avg_npu_load_percent` | `float \| None` | Average Torq NPU utilisation while the sampler was running, or `None` when unavailable. |

#### `run_vmfb`

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

#### `TensorInfo`

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

#### Utility Functions

##### `random_inputs_from_info(inputs_info)`

Generate random NumPy arrays matching the given tensor metadata. Useful for testing.

- **inputs_info** — Iterable of `TensorInfo`.
- **Returns** — List of NumPy arrays with appropriate shapes and dtypes.

## C/C++ API

There is no separate Torq-specific application API for C/C++. Instead, C/C++ programs use [IREE](https://iree.dev)'s own public runtime API (`iree_runtime_session_t`, `iree_hal_device_t`, etc.) and simply request the `torq` HAL driver by name — the same driver that the Python `torq.runtime` package requests via `device_uri="torq"`. Torq supplies:

- An IREE HAL driver/device plugin (`runtime/driver/`) that implements the `torq` device, registered as an IREE "external HAL driver".
- A hardware/simulator backend (`runtime/torq_hw/`), selected with the `--torq_hw_type` flag.
- The `torq-run-module` CLI tool (`runtime/tools/torq-run-module-main.cpp`), a Torq-flavored fork of IREE's own `iree-run-module`.

### Building with the Torq Driver

Enable the driver when configuring the runtime build:

```bash
cmake -DIREE_EXTERNAL_HAL_DRIVERS=torq ...
```

Any binary that links the `driver::registration` target (as `torq-run-module` does) gets the `torq` driver registered into IREE's standard driver registry automatically — no explicit call to `iree_hal_torq_driver_module_register()` is needed in application code as long as the instance is created with `iree_runtime_instance_options_use_all_available_drivers()`.

### Quick Start

#### Running a Model via the CLI

The quickest way to run a `.vmfb` without writing any code is the `torq-run-module` tool built alongside the runtime:

```bash
torq-run-module \
    --module=mobilenetv2.vmfb \
    --device=torq \
    --input=1x224x224x3xsi8=@input.bin
```

#### Embedding the Runtime

Applications that embed the runtime directly use IREE's high-level `iree_runtime_session_t` API and request the `"torq"` device by name:

```cpp
#include "iree/runtime/api.h"

iree_status_t run_inference(const char* vmfb_path) {
  // Shared across all sessions. Enumerates every driver linked into the
  // binary, including "torq" once the runtime is built with
  // -DIREE_EXTERNAL_HAL_DRIVERS=torq.
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance));

  // Equivalent of VMFBInferenceRunner(..., device_uri="torq").
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view("torq"), &device));

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session));
  iree_hal_device_release(device);

  // Equivalent of VMFBInferenceRunner(vmfb_path, ...).
  IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_file(
      session, vmfb_path));

  // Equivalent of runner.infer([input_array]). The function name uses the
  // "<module_name>.<function_name>" form, matching the exported name in the
  // compiled module (compare to `function="main"` in VMFBInferenceRunner).
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.main"), &call));

  iree_hal_buffer_view_t* input = NULL;
  // ... allocate/populate `input`, e.g. with
  //     iree_hal_buffer_view_allocate_buffer_copy() ...
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_inputs_push_back_buffer_view(&call, input));
  iree_hal_buffer_view_release(input);

  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(&call, /*flags=*/0));

  iree_hal_buffer_view_t* output = NULL;
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_outputs_pop_front_buffer_view(&call, &output));
  // ... read `output` back to host, e.g. with iree_hal_device_transfer_d2h() ...
  iree_hal_buffer_view_release(output);

  iree_runtime_call_deinitialize(&call);
  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  return iree_ok_status();
}
```

See IREE's own [`hello_world_terse.c`](https://github.com/iree-org/iree/blob/main/runtime/src/iree/runtime/demo/hello_world_terse.c) demo for a complete, runnable version of this pattern, and `runtime/tools/torq-run-module-main.cpp` in this repository for how `torq-run-module` drives the same device through IREE's `iree_tooling_run_module_from_flags()` helper instead of hand-written session code.

### Device Selection

Hardware backend and allocator selection are exposed as process-wide IREE flags rather than constructor parameters:

| Flag | Default | Description |
|------|---------|-------------|
| `--torq_hw_type` | `astra_machina` | Torq hardware/simulator backend: `astra_machina` (SoC NPU), `sim` (x86 CModel simulator), `aws_fpga`, etc. — the C/C++ equivalent of the Python `torq_hw_type` parameter. Declared in `runtime/driver/TorqIO.h`. |
| `--torq_device_allocator` | `auto` | Buffer allocator backing the `torq` device: `auto` (probe for a usable DMA-BUF heap), `cpu`, or `dmabuf`. Declared in `runtime/driver/registration/driver_module.c`. |

Set them on the command line (as with `torq-run-module --torq_hw_type=sim ...`), via a flags file, or by calling IREE's `iree_flags_parse_checked()` yourself before creating the device.

### Profiling

There is no C/C++ equivalent of `infer_time_ms`, `profile_vmfb_inference_time`, or `profile_vmfb_resources`. To reproduce the same measurements in C/C++:

- **Inference latency** — time the call yourself around `iree_runtime_call_invoke()` (e.g. with `iree_time_now()` or `std::chrono`), the same way `infer_time_ms` is computed around `infer()` in Python.
- **NPU/CPU/DRAM usage** — the raw sources the Python `ResourceSampler` reads are available directly on Linux: process-wide CPU/DRAM from `/proc/<pid>/stat` and `/proc/<pid>/smaps_rollup`, and cumulative Torq NPU execution time from `/sys/class/misc/torq/statistics/torq_inference_time`. The runtime does not provide a reusable C/C++ helper for this; an application needs to read these itself around the inference call.

### Public Headers

| Header | Contents |
|--------|----------|
| `runtime/driver/registration/driver_module.h` | `iree_hal_torq_driver_module_register()` — registers the `torq` HAL driver factory. |
| `runtime/driver/torq_device.h` | `iree_hal_torq_device_params_t`, `iree_hal_torq_device_create()` — device creation, used internally by the driver factory. |
| `runtime/driver/TorqIO.h` | `FLAG_torq_hw_type` and other Torq runtime flags. |

For everything else — sessions, calls, buffer views, modules — use IREE's own public headers: `iree/runtime/api.h`, `iree/hal/api.h`, and `iree/vm/api.h`.
