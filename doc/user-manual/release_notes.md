# Release Notes

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

