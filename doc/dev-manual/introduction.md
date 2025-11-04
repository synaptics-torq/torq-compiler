## Introduction


The {term}`Torq` compiler is based on {term}`MLIR`, a framework designed to support
compilation for heterogeneous hardware.
The compilation is implemented as a sequence of {term}`pass`es that {term}`lower` an high-level
representation to an {term}`IR` expressed using {term}`dialect`s that can be customized
to represent target-specific operations.
MLIR itself is part of the {term}`LLVM` ecosystem and, being completely generic, does not provide
any specific support for features commonly needed in machine learning application.
To avoid developing all these features from scratch, Torq compiler is built as a plugin on top of
{term}`IREE`, an MLIR-based end-to-end compiler and runtime specialized for {term}`lower`ing
{term}`ML` models to a variety of architectures including CPUs, GPUs and custom hardware.
