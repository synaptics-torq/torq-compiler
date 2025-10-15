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

The Torq compiler is delivered in two formats:

> **Note:** The Docker approach is recommended for most users, as it provides a pre-configured environment with all dependencies.

#### Release Package
- Pre-compiled Torq Compiler binary
- Runtime Simulator and supporting libraries
- Useful scripts for analysis and utility tasks
- User manual
- Sample models

#### Docker Image
- Contains all contents of the Release package, pre-configured in a containerized environment
- Pre-installed dependencies and tools

### System Requirements

- Supported operating systems:
  - For the release package: Ubuntu 24.04
  - For the Docker image: Any system that supports Docker
- Supported hardware: Synaptics Torq hardware (SL2610 SoC families)

### Getting Started
Refer to the [Getting Started](./getting_started.md) for installation and setup instructions. Ensure your system meets the requirements above before proceeding.

### Support and Contact
For technical support, questions, or to report issues, please contact your Synaptics support representative or use the support channels provided with your release package.
