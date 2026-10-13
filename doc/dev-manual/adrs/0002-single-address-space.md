# ADR0002 - Use single address space for a Torq hardware context

## Status

Proposed  | **Accepted** | Implemented

## Context

The compiler currently generates executables that assume they run in a
dedicated address space. This relies on the runtime switching between
different address spaces as needed when executing different
executables, by switching the IOMMU page table used by the NPU.
Because all addresses are known at compile time, this assumption means
the descriptors in the executable never need to be changed at runtime.

This approach has several drawbacks:

- Switching the IOMMU page table at each dispatch adds overhead.

- All dispatch inputs must be aligned to 4K so they can be mapped at
  fixed locations.

- On MMU-less systems, buffer addresses must be known at compile time.
  This makes it impossible to use one model with multiple input
  buffers without performing additional allocations and copies.

- Multiple dispatches cannot be chained on the hardware side, since
  switching page tables requires interrupts on the host side.

- Calling the same executable with multiple input buffers requires
  modifying the page table at each invocation, which makes
  de-duplication of executables inefficient.

- It is very hard to reuse scratch memory across executables
  as it is not clear which dispatches will be executed and when.

More broadly, this per-executable address space model doesn't map
well to the HAL designs supported by IREE, which typically assume a
single address space per hardware context. Adopting a single hardware
address space instead makes it possible to schedule command buffers
containing multiple dispatches without switching page tables between
them, enabling pure hardware-level chaining of dispatches without host
intervention, for the parts of a model that only involve NPU-side
operations (see Decision below for how dispatches that still require a
host action are handled).

## Decision

We adopt a single address space per Torq hardware context, so each
Torq HAL device instance has exactly one hardware context.

Torq (device) buffers allocated within the context, as well as host
buffers imported into it, are automatically mapped into that single
address space.

The compiler generates executables assuming that two slots in LRAM
are reserved for NSS descriptors. Management of these slots is
handled by the runtime. This avoids requiring global planning of LRAM
across all executables.

For each executable, the compiler generates a list of relocations
that lets its sections be loaded at any address within the address
space and its bindings be used at arbitrary addresses. Three kinds of
sections are generated:

- uninitialized sections: used as scratch space by the executable at
  runtime; not serialized into the model file

- relocatable sections: loaded from the vmfb and relocated at runtime
  (essentially NSS CFG programs)

- read-only sections: contain data that is constant at runtime (e.g.
  constants, slice descriptors, etc.)

Relocations come in two kinds: relocations for bindings and
relocations for sections. Each relocation carries a reference to a
base buffer and either substitues an offset within it or its size.
The exact list of relocation kinds, including which section and byte
offset each one patches and the patched width, is expected to be
refined during implementation; in general we expect to need to patch
the absolute address of a buffer, an absolute address plus an offset
within a buffer, and the size of a buffer (the latter is needed, for
example, to patch a DMA descriptor). If a binding relocation is
missing from the list, the corresponding address must not silently
resolve to a stale address; instead, address 0 of the address space is
reserved and intentionally left unmapped, so an unresolved binding
points there and faults loudly instead of being quietly used. Whether
this can be surfaced to the host as an IOMMU page-fault interrupt, as
opposed to some other failure mode, is still to be determined.
The relocatable sections of executables need to be loaded by the 
runtime or the NPU in LRAM. Relocatable sections do not load or jump to
other sections on their own, chaining of execution must be 
orchestrated by the runtime.

In addition to creating executables for all the dispatches the
compiler also generates some special relocatable sections that 
can be used to perform transfers during command buffer execution. These 
sections have relevant relocations for the buffers they reference. 
The compiler also adds to each model an additional relocatable section
to perform "load and jump" operations that can be appended by the runtime
to other relocatable sections to perform NPU side chaining of
executable sections.

When an executable is loaded, all of its read-only sections are
mapped into the single address space.

When a Torq command buffer is recorded, the runtime first 
tracks all the dispatches and transfers that are required building
a list executable segments that will be part of the command buffer.
When recording transfers it appends the corresponding transfer 
executables to the list of executable segments. During recording
the runtime also tracks the amount of uninitialized section 
space required by each dispatch to find the largest amount required.

While finalizing the recording the runtime starts by allocating
a dedicated region of the address space. This region will contain
all the executable sections and the uninitialized segments area.
It then copies all executable sections, appending at the end of 
each section the "load and jump" segment configured to jump
to the next executable section which may be of the same executable,
another executable, or a transfer segment. Finally, it relocates
all references to read-only and uninitialized sections to their 
location in the address space. Dispatches re-use the same 
uninitialized sections area as they don't run in parallel.
While the command buffer is being constructed, the runtime tracks 
and records the binding relocations it requires which will be 
resolved before submission.

Each section plus its appended "load and jump" segment forms the
chunk that gets loaded into one of the two reserved LRAM slots. While
one slot is executing, the NPU loads the next chunk into the other
slot ahead of time.

When a command buffer is submitted to a queue, the runtime resolves
all binding relocations before submitting the command buffer to the
hardware for execution. Resolving bindings at submission time, rather
than earlier when the command buffer is finalized, is necessary
because IREE normally records command buffers once, at initialization
time (e.g. inside a global initializer block), rather than once per
inference. The same recorded command buffer is then submitted many
times with different buffers, for example the same recording reused
for every token during autoregressive decoding. If bindings had to be
fixed by the time recording finishes, the command buffer would need to
be re-recorded whenever the caller supplies different buffers, which
defeats the purpose of recording ahead of time.

The chaining described above covers dispatches and transfers that only
require the NPU. Currently models also contain some operations that still
require the host CPU (such as host fallback).
In the medium term we intend to lift host CPU dispatches out of the
torq codegen entirely, as a separate stream of work. Until that lands,
the runtime handles host actions in the middle of a chain by looking
at the action list already generated for each executable. If an
executable's action list only contains NSS start/wait actions, it can
be stitched into the chain like any other NPU-only section, with no
runtime round-trip to the host. If the action list also contains host
actions, the runtime inserts an interrupt into the chain immediately
before the host action: the chain stops there, the host performs its
action, and then resumes the chain by loading and jumping to the next
chunk. In effect, the command buffer becomes a list of interleaved
chunks, some of which run purely on the NPU and some of which are host
actions, stitched together from the action lists of the executables
involved. The "no host intervention" chaining described above applies
to the NPU-only chunks; a host action always introduces a stop and
resume. Host-program arguments can themselves contain buffer
addresses, so they also need relocation coverage, the same as NPU
bindings.

For debugging purposes the runtime may decide to not chain
sections together and insert instead an explicit return to host.

## Alternatives

A first alternative is to keep per-executable address spaces at compile
time but patch each executable's descriptors on the host before every 
dispatch, instead of switching page tables. This removes the alignment and
MMU-less requirements, since buffer addresses no longer need to be
fixed at compile time, and it makes de-duplicating executables easier.
However, it requires the host to patch descriptors between every pair
of dispatches, which reintroduces host intervention and defeats the
goal of chaining dispatches purely on the hardware side.

Another alternative is to allocate a fresh single address space per
command buffer rather than one shared address space per hardware
context. This keeps the address space small and scoped to the buffers
actually used by a given command buffer, but it still requires a page
table switch whenever the hardware moves from one command buffer to
another, and doesn't simplify the relocation code. It would also 
complicate the lifetime of imported host buffers, which would need to 
be re-imported for every command buffer instead of once.

A third alternative is to keep fixed, compile-time-assigned addresses
but compute them for an entire model instead of a single executable,
so that all executables belonging to the same model agree on a shared
layout. This avoids relocations entirely, but it requires a global
memory plan that doesn't fit in IREE model and still forces inputs to be
4K-aligned and addresses to be fixed for MMU-less targets, so it only
narrows the scope of the drawbacks above rather than removing them.
It also doesn't allow to de-duplicate the executables as each instance
will have unique addresses for its buffers.

None of these alternatives remove host intervention between
dispatches, support MMU-less targets, and enable de-duplication of
executables and their scratch memory at the same time, which is why a
single address space per hardware context, with relocations generated
by the compiler and resolved by the runtime, was chosen instead.

## Further Work

- Investigate strategies for re-using LRAM resident buffers across dispatches. This can
  be achieved by defining standard locations for input/outputs within the LRAM so that
  the runtime can elide transfers if input/outputs happen to be already resident.

- Chain execution of submitted command buffers to minimize host intervention.

- Fully remove host fallback from executables to ensure all operations are handled by the NPU.

## Consequences

### Positive

- Reduce host intervention between dispatches
- Reduced memory scratch memory footprint
- Allow dynamic buffer locations on MMU-less targets
- Allow arbitrary bindings addresses
- Allow easier import of host buffers (limited by cache line alignment)
- Multiple models in the same process can efficiently share the same address space
  if desired, or use separate address spaces if the user requests multiple
  HAL devices.
- Multi-process works as before, each process gets its own HAL device,
  only one device can be executing on the NPU at any given time.

### Negative

- Increased complexity in the runtime
- Network and buffers must fit within 4 GB of addressable space
- Potential fragmentation of the single address space over time
- Since there is no global LRAM memory planning and relocations it is not
  possible to re-use LRAM resident buffers across different dispatches,
  this will be handled in a separate ADR. For the moment we assume
  that in this case we fuse all operations that use LRAM resident
  buffers within a single dispatch.
