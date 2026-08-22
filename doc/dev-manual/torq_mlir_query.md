# torq-mlir-query

`torq-mlir-query` is an interactive REPL for inspecting MLIR files: filtering
operations, walking use-def chains (backward/forward slices), and extracting
matched operations into a standalone function. It is the Torq build of
upstream [mlir-query](https://mlir.llvm.org/OpenMeetings/2023-07-13-MLIR-query.pdf),
with all IREE, Torq (`torq_hl`, `torq_hw`), TOSA, and Torch dialects
registered, so it can load IR dumped from any stage of the Torq pipeline.

Source: `compiler/tools/torq-mlir-query.cc`

## Building

```bash
$ cmake --build ../iree-build --target torq-mlir-query
```

The binary lands at `../iree-build/third_party/iree/tools/torq-mlir-query`.

## Usage

`torq-mlir-query [options] <input.mlir>` loads the IR, then runs queries
against it, interactively or from the command line. Pass
`--allow-unregistered-dialect` for IR containing unknown ops, and
`--no-implicit-module` if the input's top-level op is not a `module`.

### Interactive use

Running without `-c` opens a prompt:

```bash
$ torq-mlir-query mymodel.mlir
```

Commands:

| Command | Description |
|---|---|
| `match MATCHER` / `m MATCHER` | Print all operations matching `MATCHER` |
| `help` | List commands |
| `quit` / `q` | Exit |
| `# ...` | Comment (useful in scripted input) |

Example session:

```
mlir-query> m hasOpName("torq_hl.conv2d")

Match #1:
mymodel.mlir:42:10: note: "root" binds here
    %5 = "torq_hl.conv2d"(...) ...
...
3 matches.
```

### Non-interactive use (file / stdin)

Queries can be given on the command line with `-c` (repeatable, run in
order):

```bash
$ torq-mlir-query mymodel.mlir -c 'm isConstantOp()' -c 'm hasOpAttrName("layer_id")'
```

Queries can also be piped in on stdin:

```bash
$ echo 'm hasOpName("torq_hl.act")' | torq-mlir-query mymodel.mlir
```

The input file defaults to `-` (stdin), so the IR itself can be piped in
instead — in that case the queries must come via `-c`:

```bash
$ torq-compile --compile-to=input mymodel.mlir | torq-mlir-query -c 'm isConstantOp()'
```

## Registered matchers

The matchers below are what `torq-mlir-query` currently registers; the set
can be extended as needed (see "Registering a new matcher" below).

Predicates on a single op:

- `hasOpName("dialect.op")` — match by operation name
- `hasOpAttrName("attr")` — op carries the given attribute
- `isConstantOp()` — op is a constant
- `isZero()`, `isOne()`, `isNonZero()` — integer constant values
- `isZeroFloat()`, `isPosZeroFloat()`, `isNegZeroFloat()`, `isOneFloat()`,
  `isPosInfFloat()`, `isNegInfFloat()` — float constant values

Combinators:

- `allOf(M1, M2, ...)` — all inner matchers must match
- `anyOf(M1, M2, ...)` — at least one inner matcher must match

Slice matchers (match a root op, then report its whole slice):

- `getAllDefinitions(M, maxDepth)` — backward slice: all transitive
  definitions of ops matching `M`, up to `maxDepth` levels
- `getDefinitions(M, maxDepth, inclusive, omitBlockArguments, omitUsesFromAbove)`
  — backward slice; the flags follow `BackwardSliceOptions` semantics
- `getDefinitionsByPredicate(M, filterM, inclusive, omitBlockArguments, omitUsesFromAbove)`
  — backward slice that stops where `filterM` rejects
- `getUsersByPredicate(M, filterM, inclusive)` — forward slice: transitive
  users, stopping where `filterM` rejects

Matcher arguments are strings, signed integers, booleans, or nested
matchers. Type `m ` and hit tab (see below) or pass a wrong argument to get
each matcher's signature.

## Composing matchers

Matchers nest arbitrarily. Some examples:

```
# Constants that feed a torq_hl.conv2d within 2 use-def hops:
m getAllDefinitions(hasOpName("torq_hl.conv2d"), 2)

# Ops that are zero-valued constants (int or float):
m allOf(isConstantOp(), anyOf(isZero(), isZeroFloat()))

# Forward slice from allocations, pruned at deallocations:
m getUsersByPredicate(hasOpName("memref.alloc"), hasOpName("memref.dealloc"), true)
```

Refer the upstream lit tests for more examples:
`third_party/iree/third_party/llvm-project/mlir/test/mlir-query/`.

## Extracting matches into a function

Appending `.extract("name")` to a match query clones the matched ops into a
fresh `func.func @name` whose arguments are the values flowing into the
slice, and prints it:

```
m getAllDefinitions(hasOpName("torq_hl.act"), 3).extract("act_backward_slice")
```

This is handy for extracting a reproducer out of a large module. Extraction
does not yet handle regions that capture values from above yet.

## Tab completion

Matcher-name completion at the prompt requires LLVM to be built with
libedit, which the IREE superbuild disables by default. Enable it on an
existing build directory and rebuild:

```bash
$ cmake -DLLVM_ENABLE_LIBEDIT=ON ../iree-build
$ cmake --build ../iree-build --target torq-mlir-query
```

Command history works regardless.

## Registering a new matcher

Matchers are plain C++ exposed to the query language through the registry in
`compiler/tools/torq-mlir-query.cc`. A matcher is any class with a
`bool match(mlir::Operation *op)` method (single-op predicate) or
`bool match(Operation *op, SetVector<Operation *> &matchedOps)` (multi-op
matcher, like the slice matchers). What gets registered is a *factory*
function returning the matcher; its parameters become the query-language
arguments (allowed types: `StringRef`, `int64_t`, `bool`, and other
matchers).

```cpp
struct HasNumResults {
    int64_t n;
    bool match(mlir::Operation *op) { return op->getNumResults() == n; }
};
static HasNumResults hasNumResults(int64_t n) { return {n}; }

// In main():
matcherRegistry.registerMatcher("hasNumResults", hasNumResults);
```

Then in a query: `m allOf(hasOpName("torq_hl.split"), hasNumResults(4))`.
