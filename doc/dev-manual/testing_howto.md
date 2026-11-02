# Testing How-to Guides

Short recipes for the usual testing tasks. They assume you have done the tutorial of
[Testing](testing.md). For why things work this way, see
[Testing Architecture](testing_architecture.md); for every setting, see
[Testing Reference](testing_reference.md).

## Run tests

Run ``pytest`` from the root of the repository, with the development virtual environment active.

| To run | Command |
|---|---|
| The tests of one folder (e.g. the op tests) | ``pytest tests/ops`` |
| One test plan (a test module) | ``pytest tests/ops/test_tosa_ops.py`` |
| The tests whose name contains a text | ``pytest tests/ops/test_keras_ops.py -k avgpool`` |
| One test | ``pytest "tests/ops/test_tosa_ops.py::test_mlir_files_torq[add-single-rescale.mlir-sim-sl2610-v1]"`` |
| Only the tests CI runs | add ``-m ci`` |
| For another chip | add ``--torq-chips sr200-v1`` (or several: ``--torq-chips sl2610-v1,sr200-v1``) |
| On several processes | add ``-n auto`` (or ``-n 8``) |
| Without the cached results | add ``--recompute-cache`` |
| What CI runs on the C model, for one chip | ``pytest -m ci -n auto --torq-chips sl2610-v1`` (long) |

To see the tests a command would run, add ``--collect-only -q``.

## Add a test case

Add the smallest test that shows the feature or the bug. Choose the kind of case from what you
have:

| You have | Add it to | Section |
|---|---|---|
| A Keras layer or small model | ``tests/testdata/keras_ops/<family>.py`` | [Keras model](#keras-model) |
| A PyTorch module or an ONNX graph | ``tests/testdata/torch_ops/<op>.py`` | [PyTorch module or ONNX graph](#pytorch-module-or-onnx-graph) |
| An MLIR file (TOSA, linalg, StableHLO) | ``tests/testdata/tosa_ops/`` (or ``linalg_ops/``, ``stablehlo_ops/``) | [MLIR file](#mlir-file) |
| A whole model, and its layers | the model test of its format, in ``tests/models/`` | [A model and its layers](#a-model-and-its-layers) |
| A compiler behavior to check, without running the model | a regression test in ``tests/compiler/`` | [Compiler regression test](#compiler-regression-test) |

Then run the new tests on both chips, ``--torq-chips sl2610-v1,sr200-v1``.

### Keras model

A family module of ``tests/testdata/keras_ops/`` defines ``CASES``: a dictionary from a case name
to a builder of the Keras model. ``torq.testing.keras_builder`` has the builders:

```python
from torq.testing.keras_builder import layers, sequential, conv2d

CASES = {
    "relu_8x8x4": sequential((8, 8, 4), layers.ReLU()),
    "conv2d_k3x3_s1x1_valid_16x16x2_oc3": conv2d((16, 16, 2), filters=3, kernel_size=(3, 3)),
}
```

- **Name** the case by what it tests: the op, its parameters and the input shape (H x W x C).
  Don't add a copy of an existing model under another name: it would test the same thing twice.
- **Quantization:** each case is tested in int8 and in int16, as ``<name>_int8`` and
  ``<name>_int16``. To test only one, set ``QUANTIZE = "int8"`` in the module, or
  ``"quantize": "int8"`` in the case's ``CONFIG`` entry: the case is then named ``<name>``, with
  no suffix.
- **Settings of one case** (compiler options, tolerances) go in the module's ``CONFIG`` table (see
  [Change the settings of a case](#change-the-settings-of-a-case)).
- **A new family module** also needs its marker, the module name, in ``pytest.ini``.

### PyTorch module or ONNX graph

A module of ``tests/testdata/torch_ops/`` sets ``KIND`` and lists its builders in ``CASES``. A
PyTorch case also gives its input shapes and types in ``IO``:

```python
import torch
import torch.nn as nn

KIND = "torch"

class SimpleModule(nn.Module):
    def forward(self, x):
        return torch.abs(x)

def make_abs():
    return SimpleModule().eval()

IO = {"abs": [[[2, 16], "f32"]]}

CASES = {"abs": make_abs}
```

For an ONNX graph, set ``KIND = "onnx"`` and make the builder return an ``onnx.ModelProto``.
``torq.testing.onnx_builder`` builds small graphs. The model is converted to MLIR when the test
runs, so there is no MLIR file to commit.

Cases that must run on the host or the CSS go in ``tests/testdata/torch_ops_host_css/``.

### MLIR file

Copy the file to ``tests/testdata/tosa_ops/`` (or ``linalg_ops/``, ``stablehlo_ops/``): the file
is the case, and its name is the case name. The function is compiled for Torq and for the CPU,
and the outputs are compared.

The cases of ``tosa_ops`` and ``linalg_ops`` must run on the NPU: they are compiled with
``--torq-disable-host --torq-disable-css``, so a change that moves them to the host fails. Files
that need the host or the CSS go in ``tosa_ops_host_css/`` (or ``linalg_ops_host_css/``).

### A model and its layers

The model tests turn a model into one test for the full model and one test per layer (or block).
Add the model to the model list of a test plan of ``tests/models/`` that has models of the same
format: TFLite (``test_tflite_*.py``), Keras applications (``test_keras_app.py``), ONNX
(``test_onnx_*.py``, ``test_moonshine_models.py``) or PyTorch (``test_torch_zoo_turbine.py``).

The same layer often appears several times, in one model or in several. The model test plans
remove these duplicates (see
[Testing Architecture](testing_architecture.md#cases-from-models)). Check how many new tests you
added with ``--collect-only -q``.

To try a model without adding a test plan, copy it to ``tests/testdata/onnx_models/`` (or
``tflite_models/``, ``torch_models/``) and run the matching test of ``tests/harness/``, e.g.
``pytest tests/harness/test_onnx_model.py``. These generic tests are not run in CI.

### Compiler regression test

A test of one compiler behavior compiles a model and checks the IR or the diagnostics, without
running it. These tests are in ``tests/compiler/``. The easiest is an MLIR file in
``tests/testdata/linalg_compilation/``: the ``test_linalg_compilation.py`` tests check that it
compiles for each chip. Give the file its
compiler options in a comment:

```
// TORQ_COMPILE_OPTIONS: --torq-flag1 --torq-flag2
```

For other checks, add a test module to ``tests/compiler/``, with a test function that uses the
``torq_compiler`` fixture (the path of ``torq-compile``) and checks the output. Keep these tests fast and give them the ``ci`` marker.

## Change the settings of a case

| Setting | Keras case | PyTorch / ONNX case | MLIR file |
|---|---|---|---|
| Compiler options | ``CONFIG = {"<name>": {"compiler_options": [...]}}`` | ``CONFIG``, ``compiler_options`` (replaces the default ones) | ``CONFIG`` table of ``test_tosa_ops.py``, ``torq_compiler_options`` |
| Tolerances | ``CONFIG``, ``comparison``: ``{"fp_avg_tol": 0.05}`` | ``CONFIG``, ``comparison`` | a comment: ``// TORQ_FP_AVG_TOL: 0.05`` |
| Range of the random inputs | | ``CONFIG``, ``input_range: [low, high]`` | ``CONFIG`` table, ``tweaked_input_data_range`` |
| Runtime timeout | | ``CONFIG``, ``sim_runtime_timeout`` (seconds) | |

The tolerance keys are listed in [Testing Reference](testing_reference.md#comparison-settings).
Settings for one quantization of a Keras case go in an ``int8`` or ``int16`` entry of its
``CONFIG``.

## Handle a known failure

**Declare it.** If a test fails because of a bug that your pull request doesn't fix, file an
issue, then add a line to the ``-xfails.txt`` file next to the test module, e.g.
``tests/ops/test_tosa_ops-xfails.txt`` for ``tests/ops/test_tosa_ops.py``:

```
test_mlir_files_torq[add-single-rescale.mlir-sim-sr200-v1] synaptics-torq/torq-compiler-dev#1234
```

A line covers one test: one runtime and one chip. The test is reported ``xfailed`` and not run.
Entries for internal chips or models go in the file of the same name in ``extras/xfail``.

**Skip only a crash.** Skip a case only when running it crashes the test process, or when it has
no reference to compare with. A skipped test never runs, so nobody sees when it passes again.

**Check whether known failures pass again:**

```
$ pytest tests/ops/test_tosa_ops.py --try-xfails --recompute-cache
```

A test that passes is reported ``XPASS``: remove its line. To run only the tests of one issue,
use ``--issue synaptics-torq/torq-compiler-dev#1234``.

**Remove the lines of renamed or removed cases.** The C model CI job fails when a line matches no
test. To check before pushing:

```
$ pytest tests/ops/test_tosa_ops.py -m ci --check-xfails --collect-only
```

### Update the xfail files from a test run

To update many lines at once, record the outcome of every test in a report, then let
``torq.testing.issues`` update the files:

```
$ pytest --try-xfails --save-test-status-report-to=report.json --recompute-cache
$ python3 -m torq.testing.issues update --dry-run --report-path report.json --parent-issue ${PARENT_ISSUE}
```

It removes the lines of the tests that pass, and adds lines for the tests that fail, each with
an issue: an existing child issue of ``PARENT_ISSUE`` with the same error, or a new one. Without
``--dry-run``, it changes the files and creates the issues.

To get the reports from CI, add the ``generate-test-status-report`` label to a pull request. The
CI jobs then run the known failures too, and attach a report to the workflow run. Use
``--run-id ${RUN_ID}`` (the number in the URL of the run) instead of ``--report-path``.

## Choose the CI jobs of a test

A marker on the test function, or on a case, selects the CI jobs that run it:

| Marker | Run by |
|---|---|
| ``ci`` | The C model job, on every pull request |
| ``fpga_ci`` | The AWS FPGA job, in the merge queue and on ``main``; on a pull request, only with the ``test-aws`` label |
| ``astra_machina_sl_ci`` | The Astra Machina board job, in the merge queue and on ``main``; on a pull request, only with the ``test-astra-machina-sl`` label |

A test without these markers runs only when someone selects it.

**A slow test.** If a test takes minutes in CI, add it to ``tests/slow-tests.txt``, so that CI
starts it first. If it is too slow for every pull request, also give it the ``release_only``
marker: the C model job then runs it in the merge queue and on ``main``, and on a pull request
only with the ``run-release-only`` label.

**A case that doesn't need the FPGA.** To keep one case of an ``fpga_ci`` test function off the
FPGA job, give the case the ``not_fpga_ci`` marker. For Keras cases, add its name to
``tests/ops/keras_ops-not-fpga-ci.txt``.

## Debug a failing test

- ``--debug-ir`` saves the IR after each compiler pass, in the ``ir`` directory of the compiled
  model (``.pytest_cache/d/versioned_fixtures/torq_compiled_model_dir/...``).
- ``--no-output-capture`` prints the whole output of the compiler and the runtime, not only the
  end of it.
- ``scripts/diff-tensor.py`` shows where the outputs differ: the failing test prints the command.
- ``--recompute-cache`` reruns every step, if you think a cached result is wrong.

## Profile a test

| To get | Add |
|---|---|
| A runtime profile of each test | ``--torq-runtime-profiling-output-dir DIR`` |
| The time of each compiler pass | ``--torq-compile-time-profiling-output-dir DIR`` |
| Where the time of each test goes (compile, reference, run) | ``--torq-step-timings FILE`` |
