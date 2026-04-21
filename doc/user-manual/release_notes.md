# Release Notes

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

### Fixed Issues

- None.	

### Known Issues

- None.

### Breaking Changes in This Release
- None.

### Deprecated Features
- None.

### Documentation
- None.

