# The Tile and Fuse Pass

The tile and fuse pass tiles operations that would not normally fit in the
available memory. When an operation is tiled, for efficiency, we also
tile(/fuse) with it (in the same tiling loop) operations that it uses directly,
and the operations that those operations use, and so on. The operation that
triggered the tiling is called the consumer, and the operations it uses are
called producers.

At the core of the pass is the function
`scf::tileConsumerAndFuseProducersUsingSCF`. The function takes an operation
that has an `TilingInterface` implementation (most of the linalg dialect, and
some tensor operations) to be tiled, and an options object that controls the
function's execution. After tiling the operation the function recursively fuses
producers, as long as they have an `TilingInterface` implementation, and the
control function allows it.

Since the scf tiling function fuses producers, we should only tile an operation
when we are sure that none of its consumers needs to be tiled. This suggests a
reverse order traversal. Hence, T&F iterates over all `TilingInterface`
operations in the function in reverse order of appearance. This guarantees that
when we tile an operation, we have already considered all its users. An
operation is tiled if it hasn't already been fused, and it does not fit in
memory.

:::{attention} TODO
What if an operation feeds multiple consumers? Once an operation was
fused, it will not be considered for tiling.

      [a]
      / \
    [b] [c]

If _a_ and _b_ implement `TilingInterface`, and both need to be tiled to fit in
memory, we will tile _b_ first, and fuse _a_ with it. _a_ still needs to be
tiled in order to drive _c_, but currently we will not tile _a_ again, which
will cause the compiler to fail later due to memory overflow. Note that if _c_
also implements `TilingInterface` and is too big to fit in memory, we will tile
_c_ and fuse _a_ with it, avoiding the overflow. The problem is only when _a_
derives an operation that is not going to be tiled. :::

## Fuse groups

Later in the pipeline, the passes that do the lowering from linalg to TorqHL use
a few rewrite patterns to do their transformations. In order for those passes to
work, T&F has to make sure that when it tiles an operation that is part
of a pattern, all the other operations that belong to the same pattern are fused
in the same tiling loop (Note that this implies all those operations must
implement `TilingInterface`). To achieve that, the relevant rewrite patterns are
executed in two modes:

1. The original rewrite mode (performs the transformation); and
2. A marking mode (performs no rewrites) that places an attribute
   (`torq-fuse-group`) on the operations that make up the pattern.

The `MarkPatternsForSuperTilingPass` pass executes the relevant patterns in the
marking mode before the T&F pass.

In most cases a rewrite pattern starts from a principal operation (e.g. a
convolution), walks forward to some output value, and walks backwards to some
set of input values. In the rewrite mode all the operations between those values
are replaced with their TorqHL equivalent. In the marking mode, the function
`markFuseGroupBackward` is used to mark those operations as belonging to a fuse
group.

In principal, an operation can belong to multiple patterns. Hence, the attribute
we use for marking the groups is an array attribute. Each group is identified by
a UID that is assigned to the principal operation (the one the pattern matching
starts from) at the beginning of the pass. This UID is recorded by the
`torq-fuse-group-id` attribute.

## Memory Footprint Check

Instead of statically approximating the memory footprint, the pass uses a
precise approach by executing the actual address assignment pipeline on a cloned
subset of the IR.

1. **Extraction**: The function `extractOpsForMemoryCheck` creates a temporary
   `ModuleOp` containing the operation to be checked (along with any fused
   producers).
2. **Simulation**: The temporary module is passed to `checkModuleFitsInMemory`,
   which runs the `AssignLramAddresses` pipeline.
3. **Verification**: If the address assignment succeeds without errors (e.g., no
   memory overflow), the configuration is deemed valid.

This ensures that the memory footprint check takes into account actual memory
layouts and optimizations performed by later passes.

## Finding Optimal Tile Sizes

The pass determines optimal tile sizes for all tileable dimensions.

### Tiling Order
The consumer operation determines which dimensions are tiled and in what order.
If the consumer is a member of a pattern fuse group, the principal operation of
the fuse group determines which dimensions are tiled and in what order. This is
abstracted by the `getTilingIterDomainsOrder` function.

### Tiling Search (`fitTileToMemory`)
Finding the best tile size involves a two-phase search:

1. **Shrink Pass**: The pass starts with full iteration domain sizes. If the
   configuration exceeds memory constraints, it iteratively reduces tile sizes
   along the target dimensions to `1` until the simulated module fits in memory.
2. **Grow-Back Pass**: Starting from the reduced sizes, the pass attempts to
   maximize the dimensions using a binary search (`searchTileSizeForDim`) to
   find the largest factor that still fits in memory.

## Producers Fuse Mode

The command line option `torq-tile-and-fuse-producers-fuse-mode` controls how
aggressively producers are fused into the consumer's tiling loop. It supports
the following modes:

- **`max-producers`** (Default): Prefers fusing more producers over maximizing
  tile size. The tile size is determined by the biggest size that can
  accommodate all producers.
- **`max-size`**: Prefers a larger tile size over fusing more producers. The
  tile size is set to maximize the consumer's footprint in memory, and producers
  are fused only if they fit within that tile size.
- **`only-patterns`**: Fuses producers only when strictly required to preserve
  pattern fuse groups.
- **`no-fuse`**: Disables producer fusion entirely.

## Loop Reduction for Memory Simulation

During the memory footprint check, the pass clones the tiled IR and runs a
subset of the lowering pipeline to determine if the tiles fit in LRAM. As we are
only concerned with the memory usage of the IR at this point, we do not need to
compile all the tiles, if they all have the same memory requirements. Therefore,
to speed up the compilation time, after the peeling pass, when the tiling for
loops have no longer any dynamic shapes, we replace each of the remaining loops
with its first iteration.
