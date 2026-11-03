# ADR0004 - Statically link external CSS functions with a dense pointer ABI

## Status

Proposed | Accepted | **Implemented**

## Context

Some computations are already available as precompiled C or C++ code, or are
more practical to implement in a library than as compiler-recognized tensor
operations. They should execute on CSS alongside the model without requiring a
dedicated Torq kernel implementation or a runtime plugin for each function.

The compiler must preserve these opaque computations through dispatch creation,
route them to CSS, allocate their result buffers, and generate calls matching
the library's ABI. The library cannot interpret MLIR-specific memref descriptors
or arbitrary strided tensor views unless that complexity becomes part of its
public interface.

CSS programs have limited instruction and scratch memory. Linking an entire
library into every program would unnecessarily consume that memory. Also,
IREE's LLVM conversion normally rewrites calls to declarations as runtime
imports, which is not the desired mechanism for code supplied in a static
archive.

## Decision

Represent an opaque call with `torq_hl.external_call`. Its `function_name`
attribute names the library symbol, and its tensor operands and results specify
the inputs and outputs.

### Relationship to ADR0003

[ADR0003](0003-model-wrapper-injection.md) defines how a wrapper is injected,
how its public entry points replace the model interface, and why injection
happens before dispatch creation. This ADR extends that mechanism with optional
library-backed calls and defines their CSS execution, buffer ABI, and linking.
It does not change the wrapper placement or public-entry-point rules.

During wrapper injection, distinguish three cases:

- A declaration naming a defined model function must have an identical type;
  calls to it are inlined as described in ADR0003. A signature mismatch is an
  error, not a request to call a library function.
- A helper defined in the wrapper is also inlined, preserving its MLIR body.
- A declaration whose name is absent from the model module identifies a
  library function. Calls to it become `torq_hl.external_call` operations
  under this ADR rather than being inlined.

Wrapper injection does not require this feature. A wrapper containing only
Linalg, Tensor, and Arith computations and model/helper calls uses ADR0003
without needing a CSS library. A wrapper may also mix such computations with
library calls; only the latter use the ABI and linking rules below.

### Dispatching and CSS Routing

Use the pre-dispatch wrapper injection point established by ADR0003 so these
calls are part of dispatch formation rather than left as unhandled operations
in the host module.

Allow `torq_hl.external_call` through the CPU-program outlining filter,
which otherwise excludes TorqHL operations. Force its outlining group to use
CSS, and report an error if CSS execution is disabled. The resulting program
uses the existing CSS executable compilation, linking, and invocation paths.

Do not implement LRAM/DTCM tiling for this operation. An invocation operates on
its complete buffers; the library and its working set must fit the available
CSS resources. The compiler does not infer a tileable implementation from an
opaque function name.

### Bufferization and ABI

Require statically shaped ranked tensor inputs and results with byte-addressable
integer or floating-point elements. Bufferization obtains the input buffers,
packs non-identity input layouts into dense row-major memrefs, and allocates
dense result buffers. Inputs are read-only, results do not alias them, and the
library must fully initialize every output.

The bufferized operation has no SSA results. Its operands are the input memrefs
followed by the output memrefs, each in the original declaration order.

Lower the bufferized operation to a direct LLVM call to a C function returning
`void`. Pass only the data pointers, not memref descriptors, shapes, strides,
element counts, or scalar results. For example:

```mlir
%output = "torq_hl.external_call"(%left, %right)
    <{function_name = "combine"}>
    : (tensor<4xi16>, tensor<4xi16>) -> tensor<4xi32>
```

uses the C signature:

```c
void combine(const short *left, const short *right, int *output);
```

The library must know the shapes and element representations through its own
contract. BF16 pointers carry raw BF16 payloads; passing a pointer does not
convert these payloads to C floating-point values. C++ implementations must
export C-linkage symbols.

Validate that lowered operands are dense memrefs with byte-addressable elements
and that declarations of the same symbol within a CSS module have compatible
LLVM signatures. Pointer types erase element and shape information, so the
compiler and linker cannot verify the library implementation's complete tensor
contract.

Mark LLVM declarations with `hal.import.static`. This preserves direct symbol
references through IREE's LLVM conversion instead of rewriting calls to its
runtime import ABI.

### Archive Linking

Add the repeatable `--torq-css-library=library.a` compiler option. Read each
archive and add it to the existing CSS linker inputs. These archives are not
linked into host programs.

Use normal archive extraction, without whole-archive linking, and enable
`--gc-sections`. Unreferenced archive members are not extracted, and unused
sections in extracted members can be discarded. Library producers should use
`-ffunction-sections -fdata-sections` for fine-grained removal.

Libraries must match the selected CSS target and ABI. Missing files produce
compiler diagnostics; unresolved symbols and incompatible object ABIs are 
reported by the linker.
The linked code is embedded through the normal CSS program binary path, so
the runtime does not need to load the archive separately.

## Alternatives

- **Introduce a dedicated Torq operation and lowering for every library
  function.** This permits function-specific optimization but requires compiler
  changes for every library and is unnecessary for opaque, whole-buffer calls.
- **Use the application host for library execution.** This avoids CSS linking
  but requires a host implementation and may add host synchronization and data
  transfers.
- **Use runtime imports or dynamically loaded plugins.** This enables late
  binding but requires a symbol-resolution and loading contract on CSS which is
  unreasonable. Static linking resolves dependencies at compilation and reuses 
  the existing program loading mechanism.
- **Pass memref descriptors to the library.** This supports dynamic shapes and
  strides but may introduce unnecessary complexity and runtime overheads.
  Dense pointers provide a small, conventional C ABI.
- **Pass arbitrary views as plain pointers.** This avoids packing but silently
  changes results when the library assumes contiguous storage. Packing preserves
  the tensor's logical element order.
- **Automatically tile opaque calls.** The compiler has no contract describing
  how a function behaves on slices, including cross-element dependencies. The
  implemented interface therefore requires whole-buffer execution.

## Consequences

### Positive

- Precompiled functions integrate without function-specific compiler passes or
  runtime plugins.
- The ABI is expressible directly in C and supports multiple inputs and outputs.
- Strided tensor inputs retain their logical values through explicit packing.
- Unused library code does not consume CSS program space when it can be removed
  at archive-member or section granularity.
- Existing CSS scheduling, binary loading, and C-model execution are reused.

### Negative

- Dynamic shapes, scalar call operands/results, and packed sub-byte element
  types are outside the tensor interface implemented here.
- Packing inputs and allocating output buffers can add copies and scratch usage.
- Opaque calls cannot be fused into a Torq kernel or automatically tiled.
- Library authors are responsible for matching target ABI, tensor shapes,
  element representations, output initialization, and resource limits.
- Statically linked library code may be duplicated across separate CSS programs.
