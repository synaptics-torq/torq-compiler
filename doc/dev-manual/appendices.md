# References

The official MLIR documentation:\
https://mlir.llvm.org/docs

Some useful MLIR tutorials:\
https://mlir.llvm.org/docs/Tutorials/

Simplified overview of the code generation capabilities available in MLIR:\
https://discourse.llvm.org/t/codegen-dialect-overview

_Linalg_ is one of the most fundamental dialects in MLIR and IREE:\
https://iree.dev/community/blog/2024-01-29-iree-mlir-linalg-tutorial/#iree-mlir-linalg-tutorial

A real full-fledged course on compiler technology:\
https://www.cs.cornell.edu/courses/cs6120/2020fa/self-guided/


## Glossary

:::{glossary}
dialect
: dialects are the mechanism used in {term}`MLIR` to define new attributes, operations, and types
and allow to model different levels of abstractions in an uniform way in the same {term}`IR`\
https://mlir.llvm.org/docs/LangRef/#dialects
:::

:::{glossary}
IR
: Intermediate Representation
:::

:::{glossary}
IREE
: Intermediate Representation Execution Environment, based on {term}`MLIR`\
https://iree.dev
:::

:::{glossary}
LLVM
: originally an acronym for Low Level Virtual Machine, it has broadened in scope and
is now a toolkit for building compilers\
https://en.wikipedia.org/wiki/LLVM
:::

:::{glossary}
lower
: in compiler technology _to lower_ means to rewrite complex operations in term of simpler ones
:::

:::{glossary}
HW API
: Application Programming Interface to create binary code for the Torq HW.
This API provides a more abstract interface than accessing HW registers directly.
:::

:::{glossary}
kernel
: in this context the code that implements a given layer or operation in a model. The words 
_kernel_, _layer_ and _operation_ are often used as synonims.
:::


:::{glossary}
ML
: Machine Learning
:::

:::{glossary}
MLIR
: Multi-Level Intermediate Representation project, based on {term}`LLVM`\
https://mlir.llvm.org/
:::

:::{glossary}
NDL
: N-Dimensional Loop: an NDL agent is an HW block able to generate an address sequence
corresponding to multiple levels of nested for-loops, each specified with a counter and a stride.
:::

:::{glossary}
pass
: In {term}`MLIR` passes represent the basic infrastructure for transformation and optimization of
{term}`IR`s\
https://mlir.llvm.org/docs/PassManagement/
:::

:::{glossary}
SSA
: Static Single Assignment form, is a common representation used in compilers where each variable
is assigned exactly once\
https://en.wikipedia.org/wiki/Static_single-assignment_form
:::

:::{glossary}
Torq
: Proprietary Synaptics NPU
:::

:::{glossary}
TOSA
: Tensor Operator Set Architecture: introduced by ARM, it is a standard specifying a set of
tensor operations commonly employed by Neural Networks\
https://www.mlplatform.org/tosa/
:::

:::{glossary}
VM
:  IREE Virtual Machine executed at runtime to run the compiled {term}`VMFB`\
https://iree.dev/developers/design-docs/vm/#virtual-machine-vm
:::


:::{glossary}
VMFB
:  Virtual Machine FlatBuffer: this is the default output format of files generated
by the {term}`IREE` compiler.
VMFB files have ``.vmfb`` extension and can be executed on the IREE runtime {term}`VM` 
:::