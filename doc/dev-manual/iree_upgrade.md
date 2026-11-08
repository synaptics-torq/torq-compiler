# Upgrading IREE

Torq is built on top of IREE, which pins its own LLVM (MLIR) and torch-mlir. Upgrading to a new
IREE release therefore moves three upstream projects at once, and each of them can break the
build, the test inputs or the generated code.

This page describes how the upgrade is done, using the IREE 3.13.1 upgrade as the worked example
(synaptics-torq/torq-compiler-dev#2944). Upgrades to 3.12 and 3.13 followed the same steps.

## Overview

1. **Prepare the fork branches** in the three synaptics-torq forks (iree, llvm-project,
   torch-mlir): the new upstream base plus the Torq patches.
2. **Make the compiler build** on a `wip/iree-upgrade-X-Y` branch, following the upstream API
   changes.
3. **Test step by step**, from the op suites to the full models, and fix each regression.
4. **Run CI and compare** with `main`: test results and compile-time estimates.
5. **Merge, then tell the team** how to update their checkouts.

Each build tree needs a full LLVM rebuild, so keep about 50 GB of disk free per build tree.

## 1. Fork branches

Torq uses forks of the three projects, each with a `torq-X.Y` branch:

| Fork | Base of `torq-X.Y` | 3.13 example |
|---|---|---|
| synaptics-torq/iree | the IREE release tag | `iree-3.13.1` + 12 patches |
| synaptics-torq/llvm-project | the LLVM commit pinned by that IREE tag | `32d702efa29d` + 6 patches |
| synaptics-torq/torch-mlir | the torch-mlir commit pinned by that IREE tag | `52f5c1c4b86c` + 6 patches |

To find the pins of a release:

```shell
$ git -C third_party/iree ls-tree iree-3.13.1 third_party/llvm-project third_party/torch-mlir
```

For each fork:

1. Create `torq-X.Y` from the new base. Don't rebase the old branch: cherry-pick the patches of
   the previous `torq-<old>` branch onto the new one.
2. **Before carrying a patch, check whether upstream already has it** (search for the upstream PR
   or look at `git log` of the file it touches). Drop the patches that are already in the new
   base. For 3.13, two LLVM patches and the torch-mlir LayerNorm-without-bias patch were dropped.
3. Resolve conflicts so the Torq behaviour stays the same. A good check is to compare the
   resulting file with the one on the old branch.
4. In the iree fork, add one commit that bumps `third_party/llvm-project` and
   `third_party/torch-mlir` to the two new fork branches.

The iree fork's `.gitmodules` still points to the iree-org URLs. This works because GitHub serves
the commits of a fork through the upstream repository too.

Fixes found later during testing (for 3.13: a transpose propagation crash on a broadcast with a
constant index) go into the fork as new patches. Each patch we carry has to be re-applied at the
next upgrade, so propose it upstream when it makes sense.

```{note}
Shallow clones (`git clone --shallow-since=<date>`) of the three forks are enough and quick to
get. Pick a date before the previous upgrade, so the old patches and the upstream history are
there for cherry-picks and conflict checks.
```

## 2. Compiler branch

Create `wip/iree-upgrade-X-Y` from `main`, with one commit per topic so the PR stays reviewable.
The first commit bumps `third_party/iree` to the fork tip, so that every later commit builds.

Then follow the upstream changes until `cmake --build <build-dir> --target torq` passes. For 3.13
these were:

- MLIR strict properties in assembly formats: TorqHW and TorqHL opt out for now
  (`useStrictPropertiesInAssemblyFormat = 0`), as IREE and torch-mlir do.
- Removed APIs: `LinalgOp::getResults()`, `checkParallelRegions` renamed
  `mayHaveParallelRegions`.
- Removed ops: the elementwise linalg named ops (`linalg.reciprocal`, ...) became
  `linalg.elementwise <kind>`.

Also look at the runtime side: new HAL vtable entries, changed flags or tooling APIs.

## 3. Testing step by step

Run the suites in this order, and commit the fixes after each step:

1. `tests/test_linalg_ops.py`, `tests/test_tosa_ops.py`, `tests/test_torch_ops.py` (`-m ci`)
2. `tests/test_keras_ops.py`
3. `tests/compiler`
4. runtime, framework, harness and lab tests
5. `tests/models`
6. `extras/tests`

Test both chips: `sim-sl2610` and SR200 (`--torq-chips=next.group`). In 3.13, every SR200 CSS
program failed to link: the new clang turned a byte loop of the CSS bootstrap code into a call to
`memset`, which only the SL2610 configuration links from libc. Only CI caught it, because local
runs were on SL2610 only.

**Before calling a failure a regression, compare with the old build.** Compile the same input
with the old and the new build trees, run both on the simulator
(`runtime/tools/torq-run-module --torq_hw_type=sim`) and compare the outputs. If they are the
same, the failure isn't caused by the upgrade.

### Typical breaks

| What breaks | 3.13 example | Fix |
|---|---|---|
| Text syntax of test inputs | TOSA properties moved after the operands: `{acc_type = i32, ...}` became `acc_type(i32)` | `scripts/migrate_tosa_syntax.py` on the repo and on extras |
| Tools that print MLIR text with their own LLVM | `tosa-converter-for-tflite --text` writes the old TOSA syntax | Convert through bytecode and print with the compiler's MLIR (`torq.utils.tflite_to_tosa`) |
| Cached test inputs | Keras tests reused old converter output | Add a versioned input to the fixture so the cache is regenerated |
| Importer output | The ONNX importer emits new Q/DQ ops that `FuseQuantizedOps` doesn't fuse | `torq-legacy-onnx-quantization` pass |
| Lowering details | Dequantize computes in f32 and truncates to bf16 last | Dequant matchers look through the `truncf` |
| CSS toolchain | The new clang emits `memset` calls | `memset` defined in `css_kernel.c`; the bootstrap objects rebuild when clang changes |

A syntax change in the extras repository needs a companion PR there. Once it is merged, point
`scripts/extras.gitlink` to its merge commit.

## 4. CI and comparison with main

A PR's CI run tests the merge of the PR with `main` as it was when the run started. Compare it
with the CI run of `main` at that same commit (the checkout step of the PR run shows which commit
it merged), not with another PR.

- **Test results:** the same number of passed, skipped and xfailed tests per chip group.
- **Compile-time estimates:** download the `cmodel_compile_profiling_logs_*` artifacts of both
  runs (`gh run download`) and compare per test. For 3.13, no test changed by more than 0.5%,
  apart from a few layers whose estimate also varies between runs on `main`.
- Before reporting an outlier, check its history over several `main` sessions on the dashboard:
  some estimates move between runs of the same commit.

FPGA results are too unstable to compare.

## 5. Merge and team update

Write the squash commit message with the fork branches and their commits, the patches dropped,
the upstream changes that hit Torq and the user-facing breaks (they go into the release notes).
For 3.13, the user-facing break was that old-syntax TOSA text no longer parses; TOSA bytecode
still works.

Then tell the team. `git submodule update` alone doesn't move the submodules nested inside
`third_party/iree`, and the build then fails. After `git pull`, run:

```shell
$ scripts/checkout_submodules.sh
```

or, by hand:

```shell
$ git submodule update --init third_party/iree
$ git -C third_party/iree submodule update --init third_party/llvm-project third_party/torch-mlir
```

then rebuild, which takes a full LLVM build. `scripts/configure_python.sh` only needs to be run
again if the upgrade changed it or the Python requirements. Point people to the migration script
if their own test files use a syntax that changed.

## Checklist

- Fork branches `torq-X.Y` created, patches already upstream dropped
- iree fork bumps llvm-project and torch-mlir to their `torq-X.Y` branches
- Compiler builds; first commit bumps `third_party/iree`
- Op suites, Keras, compiler, runtime, models and extras tests pass on SL2610 and SR200
- Failures compared with the old build
- Companion extras PR merged, `scripts/extras.gitlink` bumped
- CI compared with `main`: test results and compile-time estimates
- Commit message lists fork commits, dropped patches and user-facing breaks
- Team told how to update submodules; release notes updated
- New fork patches proposed upstream
