# The Super Tiling Pass

The super tiling pass tiles operations that would not normally fit in the
available memory. For efficiency, when possible, if one operation is tiled, we
also tile with it (in the same tiling loop) all the operations that directly
feed it, and the operations that directly feed them, and so on. The feeding
operations are called producers, and tiling them in the same loop as the
original operation is called fusion.

At the core of the pass is the function
`scf::tileConsumerAndFuseProducersUsingSCF`. The function takes an operation
that has an `mlir::TilingInterface` implementation and needs to be tiled, and an
options object that controls the function's execution. After tiling the
operation the function recursively fuses producers, as long as they have an
`mlir::TilingInterface` implementation.

Since the scf tiling function fuses producers, we should only tile an operation
when we are sure that none of its consumers needs to be tiled. This suggests a
post-order traversal. Hence, super tiling does a DFS traversal over the
operations data flow DAG, and processes the operations in post-order. An
operation is tiled if it hasn't already been fused, and it does not fit in
memory (more on that later).

:::{attention} TODO
Currently we first collect all the `TilingInterface` operations in a
post-order, and then we process them. We can't do the processing inline as we
will be modifying the graph while waking it. I think mlir has pattern drivers,
such as `mlir::walkAndApplyPatterns`, that can handle the walk for us, but I'm
not sure they do the walk in the right order. The documentation does say the
walk is post-order, but it does not say over what graph (data-flow?).
:::

:::{attention} TODO
What if an operation feeds multiple consumers? Once an operation was
fused, it will not be considered for tiling.

      [a]
      / \
    [b] [c]

If _a_ and _b_ implement `mlir::TilingInterface`, and both need to be tiled to
fit in memory, we will tile _b_ first, and fuse _a_ with it. _a_ still needs to
be tiled in order to drive _c_, but currently we will not tile _a_ again, which
will cause the compiler to fail later due to memory overflow. Note that if _c_
also implements `mlir::TilingInterface` and is too big to fit in memory, we will
tile _c_ and fuse _a_ with it, avoiding the overflow. The problem is only when a
derives an operation that is not going to be tiled.
:::

## Fuse groups

Later in the pipeline, the passes that do the lowering from linalg to TorqHL
use a few rewrite patterns to do their transformations. In order for those
passes to work, super tiling has to make sure that when it tiles an operation
that is part of a pattern, all the other operations that belong to the same
pattern are fused in the same tiling loop (Note that this implies all those
operations must implement `mlir::TilingInterface`). To achieve that, we have
modified the relevant rewrite patterns to operate in two modes: the original
rewrite mode; and a new marking mode where no rewrites are done, except for
placing an attribute (`torq-fuse-group`) on the operations that make up the
pattern. The `MarkPatternsForSuperTilingPass` pass executes the patterns in the
marking mode before the super tiling pass.

In most cases a rewrite pattern starts from a principal operation (e.g. a
convolution), walks forward to some output value, and walks backwards to some
set of input values. In the rewrite mode all the operations between those values
are replaced with their TorqHL counterpart. In the marking mode we use the
function `markFuseGroupBackward` to mark those operations as belonging to a fuse
group.

In principal, an operation can belong to multiple patterns. Hence, the attribute
we use for marking the groups is an array attribute. Each group is identified by
a UID that is assigned to the principal operation (the one the pattern matching
starts from) at the beginning of the pass. This UID is recorded by the
`torq-fuse-group-id` attribute.

## Memory footprint check

Note that there's an inherent problem in approximating memory footprint at this
early stage of the pipeline: the graph is going to be optimized in later stages
of the pipeline, which entails a smaller footprint.

We approximate the memory footprint of an operation in the function
`getOperationDataSize`. For operations that do not belong to a fuse group we
generally sum the data size of all the tensor operands (ultimately using
`mlir::syna::getShapeTypeDataSize`). This is done in an `mlir::TypeSwitch` which
allows specialization for different operations.

For operations that belong to a fuse group we return a non-zero value only when
the operation has no consumers from the same fuse group (and 0 otherwise). This
ensures a fuse group operation is tiled only when it is the bottom most
operation of the group, and all the fuse group operations will be fused to its
tiling loop. The value we return for that operation is an approximation of the
TorqHL operation that will replace the whole group. To compute that we walk
backwards and look for the inputs to the fuse group, and we sum their data
sizes.

## Finding a good tiling factor

We currently always tile the second dimension of an operation, unless it only
has one dimension in which case we tile that dimension.

The initial tiling factor is calculated by dividing the approximated memory
footprint of the operation by the available memory. We assume operations that
belong to a fuse group will be executed on the NSS, so we use
`TorqHw::get().getAvailableMemoryForTiling()` to get the available memory for
them, and for other operations we use 10k.

After calling `scf::tileConsumerAndFuseProducersUsingSCF` we know which
operations fused into the tiling loop. We calculate a memory footprint for each
one of them, take the maximum of all footprints, divide it by the tiling factor,
and use that as the approximation of the tile footprint. If this approximation
divided by the factor is still smaller than the available memory, we are done
processing the operation. Otherwise we call
`scf::tileConsumerAndFuseProducersUsingSCF` again, this time with a factor based
on the new memory footprint.

Note that if we tried to approximate the memory footprint of the tiled operation
by inspecting the resulted loop, we would not be able to leverage the fuse group
information. Hence, we collect the footprints from the original operations and
divide it by the factor, or by the available memory to calculate the value that
we need.

:::{attention} TODO
Taking the maximum of all the operations (tiled + fused), assume they are in
applied in a sequence. This should be fixed to take into account the real
hierarchy.
:::
