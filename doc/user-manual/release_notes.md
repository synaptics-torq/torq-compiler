# Release Notes

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

