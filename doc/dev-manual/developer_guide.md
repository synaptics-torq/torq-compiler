# Developer Guide

This guide is intended for developers working on the Torq compiler and runtime. It covers the main workflows and extension points for adding new functionality, optimizing compute patterns, and integrating custom kernels. Each section provides step-by-step instructions, code examples, and references to relevant source files to help you get started quickly.

**Topics covered:**
- How to add support for new operators and convert them to Torq kernels.
- Guidelines for developing efficient kernels using Torq hardware APIs.
- Instructions for creating and registering new compiler passes.
- Methods for implementing and testing compute optimizations.

Explore the sections below for detailed guides and practical examples:


```{toctree}
:caption: 1. Operator and Kernel Support
:maxdepth: 2

TOSA to Torq Conversion Guide <adding_ops.md>
Kernel Development Guide <kernel_dev.md>
```

```{toctree}
:caption: 2. Compiler Passes and Optimizations
:maxdepth: 2

Creating a New Compiler Pass <add_pass.md>
Implementing Compute Optimizations <passes_optimizations.md>
Super Tiling Optimization <super_tiling.md>
```