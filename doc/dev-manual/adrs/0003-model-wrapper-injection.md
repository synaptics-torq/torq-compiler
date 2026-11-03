# ADR0003 - Inject MLIR model wrappers before dispatch creation

## Status

Proposed | Accepted | **Implemented**

## Context

Applications often need preprocessing and postprocessing that are not part of
the imported model. Examples include changing input representations, applying
normalization, combining inputs, and transforming model results into an
application-specific output format.

These transformations should be configurable without modifying the original
model or adding model-specific compiler passes. They may change the public
function signature, consume multiple inputs, produce multiple outputs, or pass
some values through unchanged. Per-argument options do not express these
relationships naturally.

The compiler must see the complete computation before creating dispatches and
generating its public ABI. Otherwise, preprocessing and postprocessing cannot
participate in normal dispatch grouping, and ABI generation may retain the
original model signature instead of the application-facing signature.

## Decision

Provide `--torq-wrapper=wrapper.mlir` to inject a separately authored MLIR
module using the `torq-apply-wrapper` module pass.

The wrapper declares the original model function with its exact signature and
defines one or more public entry points. Its function bodies express the
desired computation with lowered operations such as Linalg, Tensor, and Arith.
Ordinary SSA operands and results specify the input/output mapping, including
reordering, sharing, and multi-input/multi-output transformations.

For example, a wrapper may add one to each input element, call a model that
squares its input, and double each result. The application invokes the wrapper
entry point and receives `2 * (input + 1)^2` rather than the original model's
output.

Insert the wrapper after frontend lowering and Torq dtype conversion, but
before tied-operand annotation, dispatch region creation, and ABI generation.
The TOSA, Torch/ONNX, and Linalg input pipelines use this placement; StableHLO
pipelines reach it through the Linalg pipeline. Consequently, model declarations
must match the post-conversion model types. Wrapper operations are not passed
back through frontend import or the preceding Torq dtype conversion pipeline.
This means that they must be expressed using the lowered MLIR operations and 
types available at this stage of the compilation pipeline.

The pass makes the original model definitions private and adds the wrapper
definitions to the model module. Wrapper public definitions become the exported
entry points and determine their public input/output types. Wrapper definitions
must not collide with existing model symbols; a declaration matching a model
definition must have an identical function type.

Inline model calls and calls to defined wrapper helpers recursively, then
canonicalize the resulting module before normal dispatch grouping. Inlining
exposes the combined computation to the existing fusion, tiling, and kernel
selection passes instead of establishing a separate preprocessing execution
mechanism.

The wrapper module may contain only top-level `func.func` declarations and
definitions. It must contain a public definition and a declaration matching a
defined model function. Reject invalid signatures, symbol conflicts, unresolved
callees, and recursive call graphs. Verify the resulting module after injection.

## Alternatives

- **Per-argument and per-result command-line specifications.** Simple unary
  transformations are concise, but shared operands, multiple results, helper
  computations, and value reordering require an additional mapping language.
  MLIR already represents and verifies these relationships.
- **Modify each model in its original framework.** This makes preprocessing
  dependent on the source framework and requires maintaining modified model
  artifacts. A lowered MLIR wrapper works across supported import paths without
  changing the source model.
- **Perform preprocessing and postprocessing in application code.** This keeps
  them outside the compiler but prevents joint optimization and can introduce
  additional host execution and transfers.
- **Inject the wrapper after dispatch creation or ABI generation.** This avoids
  changing input pipeline construction but requires additional dispatch handling,
  loses joint grouping opportunities, and risks exposing the wrong public ABI.

## Consequences

### Positive

- Application-facing signatures and value mappings are explicit, typed MLIR.
- Original model artifacts remain unchanged.
- Linalg preprocessing, model computation, and postprocessing use the same
  dispatch and code generation infrastructure.
- Defined helpers allow reuse without imposing an execution boundary.

### Negative

- Wrapper authors must use the model's post-conversion types and already
  lowered operations, rather than frontend-specific operations.
- Inlining can increase IR size when a model or helper is called repeatedly.
- Recursive call graphs and additional top-level module constructs are not
  supported.
- Changing the exported signature requires callers to use the wrapper entry
  point and its types rather than the original model interface.
