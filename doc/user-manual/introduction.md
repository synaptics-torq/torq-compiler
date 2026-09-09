## Introduction

The {term}`Torq` compiler is based on {term}`MLIR`, a framework designed to support
compilation for heterogeneous hardware. Built as a plugin on top of {term}`IREE`, the Torq compiler enables efficient deployment of {term}`ML` models across various architectures, with special optimizations for Synaptics hardware.

This comprehensive guide will help you understand, install, and effectively use the Torq compiler system for deploying machine learning models on Synaptics hardware.

### Technical Background

The compilation is implemented as a sequence of {term}`pass`es that {term}`lower` a high-level
representation to an {term}`IR` expressed using {term}`dialect`s that can be customized
to represent target-specific operations.
MLIR itself is part of the {term}`LLVM` ecosystem, it bring important components like the Linalg, TOSA and LLVM-IR dialects, some important generic algorithm like tiling, and of course the LLVM compiler itself which is used to compile part of a graph targeting the CSS or the Host CPU. To leverage these features, Torq compiler is built as a plugin on top of
{term}`IREE`, an MLIR-based end-to-end compiler and runtime specialized for {term}`lower`ing
{term}`ML` models to a variety of architectures including CPUs, GPUs and custom hardware.

### Distribution Contents

The Torq compiler and runtime are distributed as separate Python wheels in the
GitHub release assets:

#### Compiler Wheel (`torq-compiler`)
- Provides `torq-compile`, `torq-lab`, and the compiler Python tools
- Supports optional extras for ONNX, TFLite, TensorFlow, and profiling
- Currently provided for x86-64 compiler hosts

#### Runtime Wheel (`torq-runtime`)
- Provides `torq-run-module`, the runtime Python API, and the host simulator where supported
- Available for supported x86-64 hosts and aarch64 boards

Install both wheels when compiling and running models on the host. See
[Getting Started](./getting_started.md) for installation instructions.

### System Requirements

- Supported operating systems:
  - x86-64 Linux hosts for compilation
  - Supported Linux x86-64 hosts and aarch64 boards for runtime execution
- Supported hardware: Synaptics Torq hardware (SL2610 SoC families)

### Getting Started
Refer to the [Getting Started](./getting_started.md) for installation and setup instructions. Ensure your system meets the requirements above before proceeding.

### Support and Contact
For technical support, questions, or to report issues, please contact your Synaptics support representative or use the available support channels.
