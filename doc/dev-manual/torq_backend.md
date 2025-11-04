## Torq compiler backend

(overview)=
### Overview

IREE provides build-in support to convert a TFLite model to an MLIR representation expressed in the
[TOSA dialect](https://mlir.llvm.org/docs/Dialects/TOSA/).
In this process TFLite layers are converted in one or more TOSA operators. This conversion preserves
all the quantization information present in the original model, although they are not expressed
as tensor attributes but in the form of "Rescale" operators.
A similar process is available to convert Torch/ONNX models to TOSA, even if the resulting MLIR
presents some differences in the way operators are used.

For these reasons the TOSA dialect is a very good input for our compiler, which can compile it for
our Torq HW independently of the actual format of the input model.

There are two approaches possible:

1. let IREE lower TOSA to standard lower-level dialects such as
    [linalg](https://mlir.llvm.org/docs/Dialects/Linalg/),
    and then convert these lower level contructs to something that can be implemented on the Torq

2. lower each TOSA operator directly to a custom implementation for the Torq, skipping
    intermediate representations

The beauty of the first approach is that it would allow us to can take advantage of many features
that come built-in with IREE, such as tiling and simplifications. The other advantage is that
if we are able to convert these low-level constructs we are in principle able to compile any
model that maps to them out of the box, without being constrained to implement each layer type by hand.
The main issue is that it may not be feasible or easy to match all these low-level constructs 
to the Torq HW, or the resulting compiled model might be inefficient in terms of inference time
or memory usage.

The second approach is easier to implement since the list of TOSA operators is well defined and
each of them can be lowered to a specific hand-optimized Torq {term}`kernel`.
The price to pay is a large number of kernels to be developed which must also explicitly support
tiling and other optimizations.

Our overall approach is to combine the two approaches above: directly lower the most common operators
to hand-optimized kernels, and try to fall back to an implementation (possibly less efficient) based
on lower-level dialects for all the other cases.


### Dialects

Torq Plugin defines two {term}`dialect`s:

- **TorqHL** used to represent high-level operations or _kernels_. These roughly corresponds to
    TOSA operators (e.g.: ``Conv2d``, ``FullyConnected``) but their inputs and attributes
    more closely match the architecture of the NPU. These operations are defined in
    [TorqHLOps.td](https://github.com/synaptics-torq/torq-compiler/blob/main/compiler/torq/Dialect/TorqHL/TorqHLOps.td)

- **TorqHW** used to represent low-level operations implemented in the NPU HW, for example it 
    includes operations such as ``NSSTask`` and ``SliceTask`` with the corresponding configuration
    for each {term}`NDL`.
    These operations are defined in
    [TorqHWOps.td](https://github.com/synaptics-torq/torq-compiler/blob/main/compiler/torq/Dialect/TorqHW/TorqHWOps.td)

The purpose of the *TorqHL* dialect is to put in a single place all the information required to 
implement an operation in HW. These information may come from multiple TOSA operators, or TOSA
may have these information in a format that is not directly compatible with our HW. Having all these
parameters together in the right format greatly simplifies the subsequent compilation steps.

The purpose of the *TorqHW* dialect is to be as close as possible to the available {term}`HW API`.
This might seem a bit redundant at first since *TorqHL*  could as well be converted to
HW bytecodes by calling the HW API directly, but representing the low-level operations in MLIR
is actually quite useful for multiple reasons:

1. in some cases we want to lower other dialects and we don't want all of them to depend directly
on the HW API.

2. lowering *TorqHL*  or other dialects directly to HW bytecodes via the HW API without intermediate
steps would result in a binary black box. We would need a separate listing file or disassembler
in order to be able to understand the content of the compiled binary file.
With the *TorqHW* dialect we can just generate this low-level representation
(the equivalent of assembly listing for CPU) using standard MLIR tools.

3. having this low-level representation inside MLIR allows to add additional low-level optimizations
that would not be possible if we went directly to HW bytecodes.



### Passes

In MLIR the transformation from the input representation to the final output representation is done
via a sequence of {term}`pass`es.
Passes represent the basic infrastructure for transformation and optimization. The idea is to keep
each pass as simple as possible and just do one thing. This makes it easy to check its behaviour
in isolation, independently of other transformations. The power of MLIR comes from the application
of multiple passes and from the ability to have different levels of abstractions coesisting in
the same {term}`IR`.

After the last pass the resulting IR is serialized, that is converted to a {term}`VMFB` binary file
that can be executed by the IREE runtime {term}`VM`.

### Data Layout

The default data layout for Torq is ``NCHW``.
For convolutions weights the layout is ``OIHW``, which is converted to ``(O/4)IHW4`` to be sure that the weights
that are processed together are contiguous in memory. The same transformation is applied for
depthwise and fully-connected layers.

Since TFLite and TOSA networks are ``NHWC``, each time a TOSA layer is replaced by
a corresponding TorqHL ``NCHW`` layer (e.g. ``Conv2d``) it is wrapped by a pair of transpose operations,
one before it to convert the incoming tensor from ``NHWC`` to ``NCHW``, and one after it to convert
back ``NCHW`` to ``NHWC``. This might seem extremely inefficient at first, but in practice the
tansposes of adjacent layers cancels out most of the time and they can be removed by a later
optimization pass, so that very few transposes remain in the final compiled model.


### Memory Allocation and Planning

Currently LRAM memory is allocated independently for each _dispatch_.
In IREE a dispatch is the basic unit of execution
that corresponds roughly to one layer, or a set of fused layers, or a part of it in case tailing
is applied. No attempt is done to share LRAM areas between different dispatches, even if this could
improve perfomances and it is an area of investigation for future improvements.
At the beginning of each dispatch the required data and weights are loaded from DRAM to LRAM,
then the required computations are executed and finally the results are stored back to DRAM.


### Strided Convolutions

The HW {term}`NDL`s in the ALU are not able to efficiently ignore unused input data
when perfoming convolutions with stride greater than 1.
For this reason the input tensor of a stride 2 convolutions is partitioned
in 4 so that all pixels having even row and column index are contiguous to each other, followed by
pixels having even row and odd column index, then odd- even-, and finally odd- odd-. Similar
transformations can be designed for convolutions with stride 3 or greateer.

This partitioning is done by a segmentation layer which is added at conversion time.
When possible this layer is fused with the preceeding layer so that no penalty is introduced.

### Torq HAL driver

The driver is based on the standard HAL API from IREE. For more information please refer to the 
[IREE official doc](https://iree.dev/developers/design-docs/vm/)
