# TORQ Network Framework

C++17 APIs for loading VMFB models and running inference on Linux and Zephyr.
The engine manages tensor allocation and device synchronization.

```cpp
#include "torq/network.hpp"

torq::Network network;
if (!network.load_model(model_bytes, model_size)) return false;
if (!network.inputs[0].assign(input_bytes, input_size)) return false;
if (!network.predict()) return false;
```

Assign prepared data to each input tensor; sizes are in bytes. After a
successful prediction, read `network.outputs` and apply any postprocessing
your application needs. Input preparation and output processing are
independent of the engine.

## Preprocessing And Postprocessing

- `torq::preprocess` provides `torq::Preprocessor` to prepare and assign input
  tensors before `predict()`. Configure input layout and conversion for your
  model; image decoding and resizing remain application responsibilities.
- `torq::postprocess` provides output-processing helpers, including classifiers
  and detectors. Apply the appropriate processor to `network.outputs` after
  prediction succeeds, using your model's parameters.
- Both libraries are optional. Applications can prepare inputs and process
  outputs directly with their own logic.

## API Basics

- `load_model(model_bytes, model_size)` copies the model and reads tensor
  shapes and types from VMFB reflection. Linux also supports
  `load_model(std::string(model_path))`.
- Assign every input with the exact tensor size, type, and layout before
  calling `predict()`. Resizing and image decoding are application work.
- `predict()` runs synchronously and allocates output buffers automatically.
  Read `network.outputs` only after success, before the next prediction or
  model reload.
- Optional `Tensor::set_buffer(std::shared_ptr<Buffer>)` attaches custom
  storage; `set_buffer(nullptr)` restores default allocation. Borrowed memory
  and custom allocators must outlive their buffers and inference.
- Serialize access to each network and its tensors. Model reload invalidates
  tensor references.

## Integration

Link `torq::engine`; add `torq::preprocess` and `torq::postprocess` when needed.

On Linux, set `TORQ_LINUX_RUNTIME_TARGET` and provide matching IREE headers
through that target or `TORQ_IREE_INCLUDE_DIR`. `TORQ_LINUX_DEVICE` defaults
to `torq`; use `local-task` for CPU-compiled models. Engine-owned NPU tensors
reuse HAL buffers directly; host/custom buffers use a copy path.

On Zephyr, size `APSRAM_TORQ_XRAM`, `APSRAM_TORQ_HEAP`, and
`CONFIG_TORQ_HEAP_SIZE` for the model. Keep `APSRAM_TORQ_HEAP` exclusive to
TORQ, without `DT_MEM_SW_*` attributes.

Linux logging defaults to warnings and errors. Set `TORQ_LOG_LEVEL=1` for
info logs or `0` for verbose logs.



