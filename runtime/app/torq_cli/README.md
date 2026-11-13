# torq_cli MobileNetV2 sample

Linux sample that loads a VMFB and a raw input tensor, runs inference, and
prints timing and the top five class scores. Tensor metadata comes from VMFB
reflection.

## Build

From the `torq-compiler-dev` directory, using an already configured SoC
build directory:

```sh
source ../venv/bin/activate
cmake -S . -B ../iree-build-soc \
  -DTORQ_BUILD_CLI=ON \
  -DTORQ_LINUX_DEVICE=torq
cmake --build ../iree-build-soc --target torq_cli
```

The executable is `../iree-build-soc/runtime/app/torq_cli/torq_cli`; run this
cross-compiled binary on the target board with the Torq driver installed.
The default `TORQ_LINUX_DEVICE=torq` selects the NPU. Use `local-task` with a
CPU-compiled VMFB for CPU inference.

## Run

```sh
./torq_cli mbv2.vmfb cat.bin
# Explicit quantization parameters for the output tensor:
./torq_cli mbv2.vmfb cat.bin 0.00390625 -128
```

Use a VMFB compiled for the target hardware with static tensor metadata.
The sample requires one input and one output. Input must be preprocessed and
match the model's tensor size, layout, and type exactly.

Optional `OUTPUT_SCALE OUTPUT_ZERO_POINT` arguments set output dequantization.
Defaults are scale `1/256`, with zero point `-128` for int8 or `0` for uint8.
Float32 outputs are read directly. Class indices use the model's indexing.

## Buffers And Logging

Engine-owned NPU tensors use the Torq HAL allocator and reuse the same HAL
buffers for binding, without wrapping or input copies. Astra allocations use
DMA-heap memory. Host/custom buffers use a copy path; DMA-BUF import failures
return an error. Output copies are skipped only when the result aliases the
attached buffer.

Logging defaults to warnings and errors. Set `TORQ_LOG_LEVEL=1` for lifecycle
and buffer logs, or `0` for verbose logging.