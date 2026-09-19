# Release Notes

## Version 2.2.0 (2026-09-19)

### Torq changes since v2.1.0

#### Highlights
- New `torq-lab` / `torq lab` unified model workflow CLI, shipped as a wheel with a pytest adapter and CI coverage.
- New `torq-decode-llm` runtime driver: model-agnostic autoregressive decode for LLMs.
- `torq-gen-config` is now a standalone CLI shipped in the compiler wheel, plus a new `torq-mlir-query` tool.
- Broader ONNX quantized coverage: `DynamicQuantizeLinear` (per-tensor, dynamic, uint8) and integer MatMul via `MatMulInteger` + dequantization.
- LayerNorm now runs on the NSS through bf16 stash retargeting.
- Large-input tiling, size-aware matmul tiling, program deduplication and NSS program packing in XRAM reduce memory pressure and code size.
- QEMU simulation replaced by MPACT; kernel-module dependencies upgraded to Astra 2.5.
- NPU devfreq is boosted during active jobs on SL2610.

#### Added
- Lowering for unsigned i1/i8 → bf16/f32 casts on pure NSS.
- ONNX `IsNaN` lowering and removal.
- Nearest-mode `Resize` raised to broadcast+collapse so it runs on the NSS.
- Elementwise-binary broadcast extended to `arith.and`/`or`/`xor`.
- `pow(x, 0.5)` transformed to `sqrt` for NSS support.
- Row-blocked `fully_connected` kernel.
- Efficient broadcasting on the inner dimension (EK).
- Packed sub-byte matmul weights streamed through the W bus.
- `--torq-host-vectorize` to tile and vectorize host programs.
- Slice count exposed as a slicing option.
- Print support from CSS.
- IREE Turbine model-zoo tests and a `util.global` lowering pass.
- `AsyncAccessOpInterface` with DMA optimizations, `ConditionallySpeculatable` on TorqHL, and `MemoryEffectsOpInterface` on TorqHL load/store/copy.
- Type and address annotation in the Perfetto profiling trace.
- Auto mode for `FLAG_torq_device_allocator`.
- EK utilities: `bitCast()`, `WRam::transpose()`, `createSliceTaskOp()`, `maxDivisor()`.

#### Changed
- Compile-time constants are outlined only over the loops they vary over.
- Producer transposes are fused into consumer slicing groups.
- i8 table precalculation applied to any i8 elementwise region, including SiLU folded into its sigmoid table.
- Parallel dimensions are tiled when reductions cannot fit LRAM; reduction tiling is skipped where it cannot help.
- Matmul tiling uses a reduction-aware shrink order with k-tiling; fast-matmul optimized for int16 and supports M < 4.
- Strided LRAM ends of `memref.copy` are staged through a dense buffer; `SwapExtractAndConvert` re-enabled behind an LRAM residency guard.
- Tile-and-fuse gains a max-producers fallback, an executor boundary check, and a widened fit-shrink reorder gate.
- XRAM pre-read skipped for write-only host arguments; host program fallback overhead reduced.
- `+sve` dropped from the default host CPU features; FMA ops unfused only on soft-float CSS targets.
- Diagnostics: tile-fit probe failures deferred to the real pipeline, each probe sees only its own diagnostics, failure-driven recovery no longer runs inside the probe, and LRAM allocation failures are classified as OOM vs fragmentation.
- `gen_config` keeps the console concise and routes full diagnostics to a log file.
- Symbol lookups cached during address resolution; DMA/slice cycle annotation split into a dedicated pass.
- Kernel-module dependencies built from the internal Gerrit tree.

#### Fixed
- Depthwise conv: H=1 bias broadcast in NHWC→NCHW, stride-2 with KH=1, asymmetric-pad Conv1D, and large-kernel Conv1D on the NSS path.
- NSS quantized Conv1D lowering, and conv-family NCHW kernels now require dense H-W planes.
- Silent maxpool corruption from mixed H/W pad phase and from an unscaled stride-1 kernel-x stride.
- Negative rescale shift overflowing the ACT `act_rsh`.
- Transposed matmul RHS crash, unsupported float matmuls gated in the Conv1D FC pattern, and zero-bias FC channel dim declared consistently with its buffer.
- Rank-0/scalar tensor handling, sub-byte tile offsets, and the unit-extent reshape wrapper around bf16 depthwise conv.
- Infinite loop when lowering oversized strided host copies.
- Elementwise operand realignment after folding rank-changing reshapes, full NHWC→NCHW loop-dim permutation for broadcast indexing maps, and permute+broadcast copy generics lowered via transpose+broadcast.
- `keepdims` reductions in pattern fuse groups, im2col tiling, and dropped-level const chains rebuilt at their retained shell level.
- Two-axis ConvTranspose kept off the interleave path; conv kernel axes capped separately on the EK path.
- Frees ordered before allocations in the LRAM peak sweep; Codegen dialect loaded before creating `TranslationInfoAttr`.
- NDL consistency check; `add` scale_bias width checked instead of asserted.
- Runtime auto mode returning zeros or crashing with SIGSEGV; build without MPACT; FP MPACT simulation.
- llvm-project updated with a TOSA Shape operation fix.

#### Testing, CI and tooling
- torch/TOSA op tests split into NSS-only and host_css folders; CSS `program_code` sharing test repaired; xfails updated.
- YOLO26 nano and small covered in the OD model tests (yolo26n temporarily disabled, then re-enabled with model checks keyed on the case name).
- Keras conv2d/depthwise/transpose test model scripts parametrized.
- Builtin TFLite kernels used for asymmetric-rhs BatchMatMul.
- `torq gen-config` ONNX conversion extended with schema-aware int64→int32 support, clamping INT64_MIN/MAX sentinels instead of failing, and LayerNormalization `stash_type` retargeted to bf16.
- `torq_compiled_model_dir` cache keying fixed; tool invocation timeouts now kill the whole process group.
- CI: GitHub runners on test bench machines, prebuilt MPACT download, Astra build file with ko/lib release, ko dependencies upgraded to Astra 2.5, FPGA runner upgrade, build-ko-dependencies cache miss fixed, artifact download command updated, ssh params passed in `check_board_liveness`, GitHub CLI added as a required dependency.

#### Documentation
- Torq setup steps for macOS users.
- User manual corrections.

#### Internal cleanup
- Refactors of virtual memory conversion, constant tracing, quantized patterns and `setDerivedMemrefAddress`; removal of unused TosaToTorqHL patterns and the `PromoteScalarsTo1D` pattern.

**Full Changelog**: https://github.com/synaptics-torq/torq-compiler/compare/v2.1.0...v2.2.0


## Version 2.1.0 (2026-07-31)

### General Notes

This release is heavily focused on robustness and production readiness, with dense compiler and CI improvements in addition to feature work.

#### Link with Astra-SDK build system

Astra-SDK for Synaptics SL2610 SoC family is automatically retrieving the source code from this [github repo](https://github.com/synaptics-torq/torq-compiler/).

In the same way that Torq framework v2.0.x was released to support [Astra-SDK v2.4.0](https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.4.0), v2.1.x is supporting [Astra-SDK v2.5.0](https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.5.0).

#### release packages and new pip wheels

We don't produce nor use the release tarball anymore (release.tar.gz); this tarball contained the binary elements of the release compiled for Ubuntu-24.04. Instead we recommended to use the various PIP wheel packages:

- astra-sl-release: contains the binary runtime for the SL2610 board:
   - lib/syna_npu.ko: precompiled Linux kernel module for Torq NPU
   - tool/astra-sl-torq-run-module: precompiled userspace command for Torq runtime
- astra-sl-runtime-wheel: PIP wheel package of Torq Runtime for SL2610 AstraMachina board.
- host-compiler-wheel: PIP wheel package of Torq Compiler supporting x86-64 Ubuntu 24.04
- host-runtime-wheel: PIP wheel package of Torq Runtime for Host based simulation (x86-64 Ubuntu 24.04)
- host-turbine-wheel: new Python backend based on iree-turbine (supporting only Host based Ahead-Of-Time compilation)

#### About performance of compiled models

In terms of performance v2.1.0 may not as good as v2.0.0 for some models. While this is one of the major focus of this release the reason for this potential regression is the switch to generic linalg-slicing pass instead of homemade torq_hl-slicing (slicing is process of parallelisation between 2 slices - aka MAC engines - available in SL2610 SoCs). This new slicing pass is going to offer a nice boost in performance on the SL2610 AstrMachina or Google-Coral board but if you see any performance issue or regression this can be mitigated by disabling slicing using the compiler option **--torq-disable-slicing**.

Moreover, if you are using an official Astra image v2.5.x you will see a nice boost of performance thanks to the increased default frequency of the Torq NPU from 800MHz to 1GHz by default (this is governed by the newly introduced devfreq support).


### Torq Compiler v2.1 changes since vs v2.0.0:

#### Highlights
- Major Slicing improvements (parallelization across our 2 MAC engines - aka Slices).
- Major expansion of quantized model support (Especially ONNX int8 quantized models).
- Added int4 and int8 block quantization support, validated with Google Gemma3 LLM.
- Added torq-turbine, a new PyTorch backend using forked iree-turbine.
- Broad compiler correctness fixes across lowering and codegen paths.
- Runtime improvements for x86 and Linux SL2610.
- Significant CI, release pipeline, and test reliability upgrades.

#### Added
- ONNX int8 quantization support.
- int4/int8 block quantization support for LLM-oriented workflows.
- New quantized lowerings (torch-onnx path), including Quantized MaxPool, GEMM, GlobalAveragePool and ElementWise Add.
- torq-turbine test/backend path using forked iree-turbine.
- Linux NPU devfreq support for SL2610.
- CI support for versioned release tags and version-branch synchronization workflows.

#### Changed
- Compiler pipeline behavior around elementwise coalescing and slicing.
- QConv2D behavior to defer constant resolution to JIT in relevant flows.
- EK-related behavior in HDIM and append/modulo paths.
- Runtime Python behavior with x86 CModel inference enablement and improved version checks.
- CI/release architecture toward cache/artifact-driven workflows, split Astra build jobs, and more wheel-based test execution.

#### Fixed
- Grouped 1x1 conv miscompile via matmul-as-conv.
- ConvTranspose fp32 bias-add lowering failures on NPU.
- ReduceMean batch-dimension handling.
- valid-to-same padding bug.
- Rank-0 scalar broadcast in elementwise binary ops.
- Clamp epilogue marking issue.
- Gather lowering issue.
- Depthwise stride-2 and asymmetric SAME-padding miscompile issues.
- InstanceNormalization pure NSS lowering issue.
- NDL cycle estimation issues.
- Shared-config corruption in Keras layer tests.
- Torchvision and test dependency robustness issues.
- SoC build kernel-module configuration issue.
- CI issues around release tests, docker build handling, and status report execution.

**Full Changelog**: https://github.com/synaptics-torq/torq-compiler/compare/v2.0.0...v2.1.0_beta


## Version 2.0.0 (2026-06-20)

This is the second major release of the Torq Framework.

### Important note about v1.5.x runtime compatibility

While we are confident that v2.0.0 compiler is way more capable and efficient than previous release, we understand that switching to a new major version may have unforeseen effect.
To help with the transition we made sure that the v2.0.0 kernel driver can support both v1.5 and v2.0 runtime and compiled models.

### Link with Astra-SDK build system

Astra-SDK for Synaptics SL2610 SoC family is automatically retrieving the source code from this [github repo](https://github.com/synaptics-torq/torq-compiler/).

In the same way that Torq framework v1.5.x was released to support [Astra-SDK v2.3.0](https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.3.0), v2.0.x is
supporting [Astra-SDK v2.4.0](https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.4.0)

### About performance

In terms of performance **v2.0.0** is much better than **v1.5.1** for most models, most notably YOLOv8 TFLite models are twice as fast. The performance gap is even bigger for LLMs and Transformers in general. As was demonstrated during the GoogleIO event Moonshine speech2text and Google-Gemma3 LLM are now fully accelerated on the NPU and deliver impressive performance
on the SL2610 board. See our [ready to use torq examples for those models](https://github.com/synaptics-torq/torq-examples) and [Google's Coral board example page](https://developers.google.com/coral/products/SL2610-demos-examples)

### Main changes in this release

The changes from v1.5.1 release are too numerous to list here, we will prepare extended release note as part of v2.0.0 but here are the main changes:
* Updated to use [IREE v3.10](https://github.com/iree-org/iree/releases/tag/v3.10.0)
  - Next version will upgrade to latest IREE
* Vastly improved robustness, performance and model coverage
  - v2.0.0 has close to 6000 passing test cases (up from 1425 in v1.5.1).
* [torq-gen-config](torq-gen-config.md)
  - This is new tool for the generation of working configurations for the compilation of ONNX models using heterogenous inference.
  - Working generated config are maintained in new git: [torq-model-configs](https://github.com/synaptics-torq/torq-model-configs) 
* Support for [TOSA 1.0.1](https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html)
* Switch to use [tosa-converter-for-tflite](https://gitlab.arm.com/tosa/tosa-converter-for-tflite) from ARM
  - Warning: Don't use older method iree-import-tflite as this will produce uncompilable TOSA files.
* Support for StableHLO input
* LLM support: Google Gemma3 is our flagship model
* Much improved Tile&Fuse and removed old torq-hl-tiling algorithm for tiling
  - As a consequence you cannot change the memory available for tiling using [the --torq-hw option](custom_hw.md), this is now done automatically depending on the LRAM size.
* Better Transformer model support
* New or improved accelerated bf16 operations (sin, cos, sqrt, etc)
* New generic linalg-slicing
  - This is not enabled by default as it needs to be optmized but you can try it by giving these options to torq-compile:
    > torq-compile <...> --torq-disable-slicing=true --torq-disable-linalg-slicing=false


## Version 1.5.1 (2026-04-20)

* various fixes to Github CI
* Update torq-runtime wheel name
* Update model preload logic in python runtime helper script
* Doc updates and runtime fixes


## Version 1.5.0 (2026-04-03)

This is the second public release of the Torq Compiler.

v1.5.0 is using an old version of IREE from [july 4th, 2024]( https://github.com/synaptics-torq/iree/tree/torq-20240704.944).
We branched out from main in order to move forward with the upgrade of IREE which will take some time to stabilize. In the mean time branch v1.5 will be maintained with bug fixes and maybe some small features for at least a couple of month in order to support the usage of [ASTRA SDK 2.3](https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.3.0).

v1.5.0 offers stable and good performances on a number of models including:
* MobilenetV2: int8 and bf16
* YOLOv8s: Body Pose and ObjectDetection
* YOLOv8n: ObjectDetection
* Moonshine
* SmolLM2
* A big number of confidential customer models

> **WARNING**
> v1.5.0 switched to the new generic algorithm for tiling that was formerly named as
> [Super-Tiling](https://synaptics-torq.github.io/torq-compiler/v/latest/dev-manual/super_tiling.html) and is now named Tile&Fuse or in short T&F.
> The deprecated torq-hl-tiling algorithm still offers better performances for 
> models can still be enabled using the `--torq-enable-torq-hl-tiling` option.

For MobilenetV2 and YOLOv8 models we recommend to use `--torq-enable-torq-hl-tiling` for maximum performance; `--torq-enable-transpose-optimization` is also recommended for YOLOv8 performance.


## Version 1.1.0 to 1.4.0

These correspond to Synaptics internal release were not formally tagged in the public git.


## Version 1.0.0

v1.0.0 was the initial release as opensource in github, we never tagged nor created a binary package for it
but it corresponds more or less to the 'initial' tag:

https://github.com/synaptics-torq/torq-compiler/releases/tag/initial


## Version 0.9.0

### Key Features & Enhancements

- Initial release of the IREE-based MLIR compiler targeting Synaptics Torq.
- Memory-aware tiling/slicing to fit on-chip memory.
- Compile-time profiling for approximate clock-cycle estimates; runtime profiling for measured execution.
- End-to-end validated on MobileNetV2.
- Automatic CSS fallback for unsupported operators.
- Supertiling (experimental): groups adjacent tiles into larger macro-tiles to improve locality and reduce launch/DMA overhead.
- Support for compiling the model on custom Synaptics SoC hardware configurations.

