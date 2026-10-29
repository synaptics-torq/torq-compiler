## Kernel Development Details

This page holds the details behind [Kernel Development](kernel_dev.md): practical advice for
writing and reviewing EasyKernel kernels, for developers and coding agents alike. Read Kernel
Development first. The API itself (what each method does and the shapes it accepts) is documented
in `compiler/torq/Utils/Kernel.h`.

Kernel Development is written by humans for humans, with the help of agents. This page may be
updated by agents, for example when the EasyKernel API changes or a review settles a new rule;
see [Updating these pages](#updating-these-pages).


### Writing a kernel

The aim of the EasyKernel is that a kernel reads like the computation it implements. Kernels in
this style are `ElementWiseBinaryPattern.cpp`, `GatherPattern.cpp` and `ScatterPattern.cpp` in
`compiler/torq/Conversions/TorqHLToTorqHW/`.

- **Start from the op's operands.** Create each tensor with `LData x(op.getX())` and shape the
  view with the `LData` methods (`fuse()`, `reshapeDim()`, `vectorize()`, `insertDim()`,
  `eraseDim()`, `moveDim()`, `bitCast()`, `subviewDim()`). Don't read shapes and strides from
  `MemRefType`, and don't rebuild the tensors from them.
- **Write for any rank.** An op is usually the same computation repeated over its outer
  dimensions. Instead of naming them (batch, channels) and handling each rank separately, iterate
  over all of them at once and index with the same iterator:

  ```{code} C++
  For(auto t = slice.iterate(input.dims(0, -2))) {   // all the outer dimensions
      IData idata = slice.iram.load(input[t][n]);
      ...
      slice.store(output[t][n], res);
  }
  ```

  Keep one special case only where an operand really has another layout, and say why in a comment.
- **No magic numbers.** Take sizes from the API (`alu.iWidth()`, `act.width()`, `iram.gather()`,
  the dims of the `IData` or `PData` an instruction returns) rather than writing 64 or 32.
- **Keep the element type.** Bit-cast only when an instruction needs it, and say why: for example
  `iram.loadIndexed()` needs integer tables, so a gather copies a bf16 table as int16.
- **Fuse only dense dimensions.** Fuse the dimensions that are dense in every tensor that uses
  them (`denseDims()`), and iterate over the others with their strides. Ask for the density the
  kernel needs in the op's `getKernelEncoding()` (`denseInnerDims`), not for a fully dense tensor.
- **Don't erase unit dimensions** to save loops: a loop of one iteration costs nearly nothing. To
  give two tensors the same dimensions, insert a unit dimension (`insertDim(0, {1})`) instead.
- **Keep slice task names stable** (`Slice slice("gather")`): they appear in profiles and
  dashboards.

If the API is missing something, first try to express it in the kernel. If the EasyKernel itself
must change, keep the change small, check which kernels reach the changed code, and explain in the
PR why it can't be done in the kernel. `Kernel.h` comments are for kernel writers: say what a
method does, the shapes it accepts and what it returns, without hardware details.


### Decisions that must agree

Whether an op runs on the NPU is decided in three places, which must agree:

1. **Frontend** (`compiler/torq/Conversions/LinalgToTorqHL/`): a pattern turns a linalg op into a
   `torq_hl` op. If it refuses, the op runs on the CSS or the host. **Once an op is `torq_hl`
   there is no fallback**: an op the kernel can't lower is a compile error, so the frontend must
   refuse what the kernel can't do.
2. **Encoding** (`getKernelEncoding()` in `compiler/torq/Dialect/TorqHL/EncodingRequirements.cpp`):
   the layout, alignment and padding the kernel needs for its operands.
3. **Kernel** (`<Op>Pattern.cpp`).

Put a decision shared by these passes in one helper (in `compiler/torq/Utils/ConversionUtils.cpp`)
and call it from each of them. When the frontend must decide before all the information is
available, make it conservative: refusing only costs speed, accepting what the kernel can't do
breaks the compile.


### Measuring

- The compile-time profile is deterministic and matches the simulator:

  ```
  torq-compile in.mlir -o out.vmfb --torq-hw=SL2610 --torq-enable-profiling \
      --torq-dump-profiling=out.csv
  ```

  Add `--torq-disable-host --torq-disable-css` to keep every op on the NPU. Sum the `Slice`
  events for the kernel time.
- Compare **slice time** for kernel changes. On small tests DMA transfers dominate the total.
- When refactoring a kernel, compile the tests that reach it before and after the change and
  compare the profile CSVs: identical files mean identical slice programs and timings.
- `--debug-only=torq-easy-kernel` prints the tensors each instruction receives, as given and as
  the instruction sees them. Use it to check a view, or which path of a kernel a test takes.


### Testing

- `tests/testdata/tosa_ops/*.mlir` run through `tests/test_tosa_ops.py` with
  `--torq-disable-host --torq-disable-css`, so they fail if an op leaves the NPU. Put cases meant
  to run on the CSS or the host in `tests/testdata/tosa_ops_host_css/`.
- Run the tests for every target with `--torq-chips` (for example
  `--torq-chips=sl2610-v1,sr200-v1`).
- Check that the tests reach every path of the kernel (with `--debug-only=torq-easy-kernel`). For
  a path no test reaches, add a test, or at least check it with a temporary one.
- Comparison settings can be set per test file with comment lines (`// TORQ_FP_MAX_TOL: 0.05`,
  `// TORQ_USE_ABS_TOL_GATE: 1`, `// TORQ_INT_TOL: 1`, ...), parsed by
  `comparison_config_from_mlir()` in `python/torq/testing/iree.py`.
- Test padding with sizes that pad several rows or columns, not just one.
- Before blaming a change for a failure, run the same test on main.


### Pitfalls

- Pads are `[left, right, top, bottom]` in `torq_hl` (`LRTBDim`); TOSA uses
  `[top, bottom, left, right]`.
- Reading past the end of a float buffer returns junk that may be NaN or Inf, and NaN × 0 = NaN.
  Zero-padding the other operand only protects integer kernels.
- Format with `./scripts/format-code` (clang-format-18, which CI checks). Newer clang-format
  versions rewrap code that version 18 accepts.


### Updating these pages

- Agents may update this page on their own (in a PR) when a source changes, for example
  `Kernel.h` or a kernel named here as an example: check the facts that depend on it, and fix the
  ones that changed.
- Changes to [Kernel Development](kernel_dev.md) are written or reviewed by a human. Keep it an
  introduction and tutorial; put the details here, and link to the section that has them.
