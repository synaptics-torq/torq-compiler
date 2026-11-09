# Testing Architecture

This page explains how the test framework is built, starting from plain pytest and adding one
piece at a time: fixtures, cases, the Torq fixtures that compile and run a model, the runtimes
and chips, the cache, the known failures and CI. It ends with a map of the source code.

Read it to change the framework, to write a new kind of test plan, or to understand a test
before debugging it. To add a test case, the [Testing](testing.md) tutorial and the
[How-to Guides](testing_howto.md) are enough.

## The goal

Every Torq test answers one question: does the model compiled for Torq give the same outputs as
a reference? For one test the framework must:

1. get the model as MLIR: read a file, or convert a Keras, TFLite, ONNX or PyTorch model;
2. make input data for it;
3. compile it with ``torq-compile`` for one chip, and run it with ``torq-run-module`` on one
   runtime (the C model simulator, an FPGA or a board);
4. compute the expected outputs with a reference (TFLite, ONNX Runtime, PyTorch, or IREE on
   the CPU);
5. compare the two, with tolerances that suit the data type.

Thousands of tests share these steps, and only a few details differ between them: where the
model comes from, which reference, which tolerances. The framework is built so that a test plan
only states those details. The rest of this page shows how.

## Pytest basics

### Collection and execution

A pytest run has two phases. In the **collection** phase, pytest imports the modules of
``tests/`` and its folders whose names start with ``test_``, and finds their functions whose names
start with ``test_``. In the **execution** phase, it calls each function. A function that raises an
exception fails; the others pass.

```python
# tests/test_foo.py
from mycode import bar

def test_bar():
    assert bar(2) == 1
```

This module has one test, ``tests/test_foo.py::test_bar``. That string is the test's **node
ID**; this documentation calls it the test name. You can run one test by giving its name:
``pytest tests/test_foo.py::test_bar``.

### Fixtures

When several tests need the same input, the code that makes it goes in a **fixture**: a function
with the ``pytest.fixture`` decorator. A test asks for a fixture by naming it as a parameter, and
pytest calls the fixture and passes its result:

```python
@pytest.fixture
def complex_input():
    return make_complex_input()

def test_bar(complex_input):
    assert bar(complex_input) == 1

def test_buz(complex_input):
    assert buz(complex_input) == 1
```

A fixture can ask for other fixtures in the same way. The steps of a Torq test are such a chain
of fixtures: the results need the compiled model, which needs the MLIR, and so on.

A fixture can also get any fixture by its name while it runs, with
``request.getfixturevalue(name)``. The Torq fixtures use this a lot, as shown below.

### Parametrization

A fixture can have several values. Pytest then makes one test per value:

```python
@pytest.fixture(params=[1, 2, 3])
def bar_input(request):
    return 2 ** request.param

def test_bar(bar_input):
    assert bar(bar_input) == bar_input * 2
```

This module has three tests, ``test_bar[1]``, ``test_bar[2]`` and ``test_bar[3]``. The part in
brackets is the **test parameters**.

The values are computed at collection time, which happens on every pytest run, even to run one
test. So they must be cheap: read a few files, build small objects. The expensive work goes in
the fixtures, which run only for the selected tests.

### Cases

When a test uses two parametrized fixtures, pytest makes a test for each combination: two
fixtures of three values give nine tests. Usually only some combinations make sense. The
framework solves this with **cases**: a test plan parametrizes a single fixture, ``case_config``,
with a list of ``Case`` objects, and the other fixtures read their values from it.

```python
from torq.testing.cases import Case

@pytest.fixture(params=[Case("first", {"a": 0, "b": 1}),
                        Case("second", {"a": 3, "b": 2})])
def case_config(request):
    return request.param.data

@pytest.fixture
def a(case_config):
    return case_config["a"]

@pytest.fixture
def b(case_config):
    return case_config["b"]

def test_bar(a, b):
    assert bar(a, b)
```

This gives two tests, ``test_bar[first]`` and ``test_bar[second]``: the name of a ``Case`` is
its part of the test parameters. A case can also carry markers, with
``pytest.param(case, marks=[...])``; a marker on the test function applies to all its cases.

## The torq.testing plugin

``pytest.ini`` loads the ``torq.testing`` plugin for every run (``addopts = -p torq.testing``).
The plugin adds the fixtures that do the five steps of [The goal](#the-goal), the
parametrization over runtimes and chips, the cache, the known-failure files, and the command
line options. Its modules are listed in [Source map](#source-map).

A test plan is a test module that uses the plugin. It defines its cases, a ``case_config``
fixture, and a test function that compares two results.

### Where the test plans are

The folders of ``tests/`` group the test plans by what their tests check:

- ``ops/``: one op or a short op pattern, compiled, run and compared with a reference. This is
  where most compiler changes are tested. Its cases are small models or MLIR files in
  ``tests/testdata/``, read by the test plans and by the collectors next to them.
- ``models/``: full models, and the cases made from their layers or blocks (see
  [Cases from models](#cases-from-models)).
- ``compiler/``: one compiler behavior, checked on the IR or the diagnostics, without running
  the model.
- ``runtime/``: the ways a compiled model runs: on the host, on the CSS, split between devices,
  through the runtime API.
- ``harness/``, ``sweeps/``: tests that run only when selected, to investigate a model file or to
  sweep a parameter space.
- ``lab/`` and ``framework/``: unit tests of ``torq-lab`` and of the ``torq.testing`` plugin,
  without the compiler.

A test plan's known failures and helper modules are next to it. ``tests/testdata/`` holds the
cases and models that several test plans share. The tests of internal models are in
``extras/tests/``.

## Following one test

This section follows one test of the MLIR op tests from collection to result:

```
tests/ops/test_tosa_ops.py::test_mlir_files_torq[add-single-rescale.mlir-sim-sl2610-v1]
```

### The test plan

``tests/ops/test_tosa_ops.py`` makes one case per MLIR file of ``tests/testdata/tosa_ops`` (the
case name is the file name), and returns this from ``case_config`` (simplified):

```python
@pytest.fixture(params=CASES)
def case_config(request, runtime_hw_type, chip_config):
    return {
        "mlir_model_file": "static_mlir_model_file",
        "static_mlir_model_file": request.param.data,      # the path of the MLIR file
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir",
        "torq_compiler_options": ["--torq-disable-host", "--torq-disable-css"],
    }

@pytest.mark.ci
@pytest.mark.fpga_ci
def test_mlir_files_torq(request, torq_results, llvmcpu_reference_results, case_config):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
```

The test asks for two results, ``torq_results`` and ``llvmcpu_reference_results``, and compares
them.

### Fixture names as settings

Three values of this ``case_config`` are names of fixtures: ``"static_mlir_model_file"``,
``"tweaked_random_input_data"``, ``"comparison_config_from_mlir"``. The plugin's generic fixtures
read them and call that fixture. For example:

```python
@pytest.fixture
def mlir_model_file(request, case_config):
    return request.getfixturevalue(case_config["mlir_model_file"])
```

This is how a test plan chooses where the model, the inputs and the tolerances come from,
without writing any compile, run or compare code. The Keras op tests reuse the same chain and
only change the names: their MLIR comes from ``tflite_mlir_model_file`` (a TFLite model
converted to MLIR, itself made from the Keras model) and their reference is
``tflite_reference_results``.

### The chain

When pytest runs the test, it resolves the fixtures it needs. For this test:

```
test_mlir_files_torq
├── torq_results                         run outputs, read back as arrays
│   └── torq_results_dir                 runs torq-run-module on the runtime of the test
│       ├── torq_compiled_model          the compiled model
│       │   └── torq_compiled_model_dir  runs torq-compile for the chip of the test
│       │       ├── mlir_model_file  ->  static_mlir_model_file (add-single-rescale.mlir)
│       │       ├── torq_compiler_options  (case options + command line options)
│       │       └── chip_config            (sl2610-v1)
│       └── torq_input_data_args
│           └── input_data  ->  tweaked_random_input_data (random, from the MLIR signature)
├── llvmcpu_reference_results            reference outputs
│   └── llvmcpu_reference_results_dir    runs the model compiled by IREE for the CPU
│       └── llvmcpu_compiled_model       compiles the same MLIR for the CPU
└── compare_test_results(...)            compares, with the tolerances of
                                         comparison_config  ->  comparison_config_from_mlir
```

``compare_test_results`` starts from the default tolerances (``DEFAULT_COMPARISON_CONFIG``) and
applies the settings that the ``comparison_config`` fixture returns. Here,
``comparison_config_from_mlir`` reads them from comments in the MLIR file, such as
``// TORQ_FP_MAX_TOL: 0.05``.

## Runtimes and chips

The test name ends with ``-sim-sl2610-v1``, but ``test_tosa_ops.py`` never mentions runtimes or
chips. The plugin adds them: every test that uses the ``runtime_hw_type`` fixture, which all the
compile-and-run fixtures do, is parametrized over

- the runtimes of ``--torq-runtime-hw-type``: ``sim`` (the default, the C model simulator),
  ``aws_fpga`` or ``astra_machina``;
- the chips of ``--torq-chips``.

Each chip is described by a JSON file in ``tests/testdata/chips`` (and in ``extras/chips`` for
internal chips), e.g. ``sl2610-v1.json`` and ``sr200-v1.json``. To test one chip, name it:
``--torq-chips sr200-v1``.

The default of ``--torq-chips`` is ``sl2610-v1``. The CI jobs test one chip each: the
``TEST_CHIPS``, ``FPGA_TEST_CHIPS`` and ``BOARD_TEST_CHIPS`` repository variables list the chips
of the C model, FPGA and board jobs. They are set per repository, not in the workflow files, so
the chips of ``extras`` are never named in the public repository.

A ``case_config`` can depend on the runtime and the chip. ``test_tosa_ops.py`` does that to add
compiler options on the FPGA only.

## The cache

Compiling and running a model takes seconds to minutes, and most of the time nothing that the
result depends on has changed since the last run. So the expensive fixtures store their results
on disk, in ``.pytest_cache/d/versioned_fixtures/<fixture name>/``, and reuse them.

### Versions

A cached fixture is a **versioned fixture**: it returns an object with the value (``data``,
``file_path`` or a directory) and a ``version``. Its version is a hash of the versions of the
fixtures it uses. Pytest calls the fixture only if no result with that version is stored, so a
change anywhere upstream reaches every step that depends on it:

- a changed MLIR file has a new version (the hash of its content), so its compilation, its run
  and its reference are redone;
- a rebuilt ``torq-compile`` has a new version (its modification time), so every compilation is
  redone, but the references are reused;
- a changed case setting, e.g. a compiler option, changes the version of the fixtures that use
  it.

Several versions of a fixture are stored side by side, so going back to an earlier branch can
reuse its results.

### The decorators

``torq.testing.versioned_fixtures`` has one decorator per kind of value:

| Decorator | The fixture returns | Its version is |
|---|---|---|
| ``versioned_cached_data_fixture`` | data, stored in the cache | from its inputs |
| ``versioned_generated_file_fixture`` | a file it writes (``versioned_file``) | from its inputs |
| ``versioned_generated_directory_fixture`` | a directory it fills (``versioned_dir``) | from its inputs |
| ``versioned_static_file_fixture`` | the path of an existing file | the hash of the file |
| ``versioned_hashable_object_fixture`` | an object that converts to JSON | the hash of the object |
| ``versioned_unhashable_object_fixture`` | any object | from its inputs |

A fixture that takes a plain value, such as a ``request.param``, needs a version for it: wrap it
in a ``versioned_hashable_object_fixture`` first.

```python
@versioned_hashable_object_fixture
def versioned_myparameter(myparameter):
    return myparameter

@versioned_cached_data_fixture
def complex_input(versioned_myparameter):
    ...  # computed once per value of myparameter, then read from the cache

def test_bar(complex_input):
    assert bar(complex_input.data) == 1
```

``--recompute-cache`` ignores the stored results. ``--debug-cache`` prints why each fixture is
recomputed or reused. The version of the compiler and of the runtime is the hash of the binary
they run and of the libraries it loads, so a rebuild to the same bytes, or a fresh install in CI,
keeps the cache.

Every version also includes a salt for what the results depend on beyond the fixtures' inputs:
the installed Python packages (but the ``torq-*`` wheels), the iree and iree-turbine commits,
the code of ``python/torq``, and the code of the test module (its file, the other modules of
its folder that aren't tests, and the ``conftest.py`` files above it). A change in any of them
recomputes the results of the tests it can affect.

The results can also be shared across sessions: ``--remote-cache-write STORE`` writes each result a
session computes to a store, a directory or ``s3://bucket/prefix``, and ``--remote-cache-read
STORES`` reads results from stores before computing them (both read the environment variables
``TORQ_REMOTE_CACHE_WRITE`` and ``TORQ_REMOTE_CACHE_READ`` by default). A store keeps one archive
per result, under its version. Only results computed successfully are stored; not the runs on an
FPGA or a board, nor the compiler's IR dumps, nor the inputs, which are cheap to make again
(fixtures marked ``@local_only``). The tests still compare the outputs with the reference, so a
change of tolerance takes effect. A result fetched from a store doesn't report the compile and
run times measured when it was computed, and the dashboard bundle lists it in
``fetched_results``. ``--test-cache-summary FILE`` appends to ``FILE`` how many results each test
file and each fixture reused, fetched and computed (the C-model CI jobs write it to their job
summary).

The C-model CI jobs use the store when the ``TEST_CACHE`` repository variable is ``true``. Pushes
to ``main`` write their results to the artifact bucket (``ARTIFACT_BUCKET``). Pull requests and
the merge queue write theirs to the bucket of work in progress (``WIP_ARTIFACT_BUCKET``), which
only they can read, and read both. The buckets keep their objects for 7 days. Runs on tags
(releases) and pull requests with the ``no-test-cache`` label compute everything.

A session only gets the results it would compute itself, so a developer gets the compiles of
the CI store only with the same options (CI passes ``--no-phases-dump``), on the same OS (the
Python version and packages are in the salt) and, for the code compiled for the host CPU (sim
compiles and LLVM-CPU references), on the same CPU.

## Cases from models

The model test plans don't list their cases: they make them from models at collection time.
A model gives one case for the full model, and one case per layer (or per block, for PyTorch).
Each layer case is a model of that layer alone, with the inputs it gets in the full model.

The same layer often appears several times: a block repeated in a model, or the same layer in
two models. Testing it twice costs time and finds nothing new. So these plans call
``drop_duplicate_cases(cases, key)`` before parametrizing, which keeps the first case of each
key. The key compares what the layer computes, not its name:

| Key | Used by | Two cases are the same when they have the same |
|---|---|---|
| ``keras_layer_key`` (``cases.py``) | ``test_keras_app.py`` | one-layer Keras model configuration |
| ``traced_layer_key`` (``torch.py``) | ``test_torch_zoo_turbine.py`` | module and call signature |
| ``onnx_layer_key`` (``onnx.py``) | ``test_moonshine_models.py``, ``test_onnx_gemma3.py`` | ONNX node, attributes and small constants |

Making the layer cases of a big model takes time, so it is cached too. In CI, the TFLite model
tests make only the full-model cases (``--tflite-layer-test-generate-full-model-only``).

## Known failures

A test that fails because of a known bug must not fail CI, and must not be forgotten. The
framework declares known failures in files, not in the code: ``test_x-xfails.txt`` next to
``test_x.py``, one test name per line with its issue. A plugin marks those tests ``xfail``
at collection.

Keeping them in files, with the full test name, has three uses:

- a line covers exactly one test: one case, one runtime, one chip;
- each failure is linked to an issue, so ``--issue`` can run the tests of an issue;
- the files can be rewritten by a tool, ``torq.testing.issues``, from a report of a test run:
  it removes the tests that pass again and adds the new failures.

By default a known failure is not run, which saves time. ``--try-xfails`` runs them and reports
the ones that pass.

A line whose test no longer exists hides nothing, but makes the files wrong. ``--check-xfails``
(used by the C model CI job) fails when a line matches no test. It compares the lines without
their runtime and chip: a run on the simulator for one chip must not report the lines of the
FPGA or of another chip.

## CI

### Jobs and markers

The CI test jobs are defined in ``.github/workflows/test-build.yml`` and the workflows it calls.
Each job runs pytest with one runtime and selects its tests with a marker:

| Job | Runtime | Marker | When |
|---|---|---|---|
| C model | ``sim`` | ``ci`` | Every pull request, the merge queue, ``main`` |
| AWS FPGA | ``aws_fpga`` | ``fpga_ci`` | The merge queue, ``main``; pull requests with ``test-aws`` |
| Astra Machina SL | ``astra_machina`` | ``astra_machina_sl_ci`` | The merge queue, ``main``; pull requests with ``test-astra-machina-sl`` |

The C model and FPGA jobs run once per chip group, in parallel. Each job runs its tests on many
pytest-xdist workers, and uploads the results to the test dashboard. The details (workers, time
limits, labels) are in [Testing Reference](testing_reference.md#ci).

### Fitting in the time

A job must end within its time limit, and a pull request waits for the C model jobs, so their
duration matters. The framework has three tools for it:

- **Start the slow tests first.** A job with many workers ends when its last test ends. The
  xdist ``worksteal`` scheduler moves waiting tests from busy workers to idle ones, but it can't
  split a test: a test of 5 minutes that starts 2 minutes before the end makes the job 3 minutes
  longer. With ``--slow-tests-first``, the tests listed in ``tests/slow-tests.txt`` start first,
  spread over the workers.
- **Run the slowest tests less often.** A test with the ``release_only`` marker runs in the C
  model job in the merge queue and on ``main``, but on a pull request only with the
  ``run-release-only`` label.
- **Run a case on fewer jobs.** Markers apply to a test function, and all the cases of a test
  plan usually share one function: every Keras op case is ``fpga_ci`` because
  ``test_keras_model_torq`` is. The ``not_fpga_ci`` marker removes one case from the FPGA job
  without changing the others.

## Source map

The plugin is in ``python/torq/testing/``; the modules are loaded by
``python/torq/testing/__init__.py``.

| Module | What it does | Start reading at |
|---|---|---|
| ``iree.py`` | The compile and run fixtures, runtimes and chips, most options | ``pytest_generate_tests``, ``torq_compiled_model_dir``, ``torq_results_dir`` |
| ``versioned_fixtures.py`` | The cache | the module docstring, ``versioned_cached_data_fixture`` |
| ``cases.py`` | ``Case`` (from ``torq.lab``), ``drop_duplicate_cases`` | the module docstring |
| ``comparison.py`` | ``compare_test_results`` | ``compare_test_results`` |
| ``tensorflow.py``, ``tflite_layer_tests.py`` | Keras and TFLite models, TFLite layer extraction, TFLite references | ``tflite_mlir_model_file``, ``tflite_reference_results`` |
| ``onnx.py`` | ONNX models, ONNX Runtime references; the layer extraction is in ``torq.lab`` (``torq/lab/model_tools/extraction/onnx/layers.py``) | ``onnx_mlir_model_file``, ``generate_onnx_layers_from_hf`` |
| ``torch.py``, ``torch_turbine.py`` | PyTorch export, traced layers, PyTorch references | ``generate_torch_layers_from_model`` |
| ``keras_builder.py``, ``onnx_builder.py`` | Small models for the op tests | the module docstrings |
| ``xfail.py`` | The ``-xfails.txt`` files, ``--try-xfails``, ``--check-xfails``, ``--issue`` | ``pytest_collection_modifyitems`` |
| ``issues.py`` | Updates the xfail files and issues from a report (``python3 -m torq.testing.issues update``) | ``main`` |
| ``reporting.py`` | Status reports, upload to the dashboard | |
| ``scheduling.py`` | ``--slow-tests-first`` | the module docstring |
| ``step_timings.py`` | ``--torq-step-timings`` | the module docstring |
| ``aws_fpga.py``, ``remote_testing.py`` | Runs on the FPGA and on the boards | |

The test plans are the ``test_*.py`` modules of the folders of ``tests/`` (see
[Where the test plans are](#where-the-test-plans-are)). The op test plans of ``tests/ops/`` read
their cases from ``tests/testdata``, with collectors next to them:

| Test plan | Cases | Collector |
|---|---|---|
| ``test_keras_ops.py`` | ``tests/testdata/keras_ops/*.py`` | ``tests/ops/keras_ops_models.py`` |
| ``test_torch_ops.py`` | ``tests/testdata/torch_ops/*.py`` (directories of ``torch_ops.group``) | ``tests/ops/torch_ops_models.py`` |
| ``test_tosa_ops.py``, ``test_linalg_ops.py``, ``test_stablehlo_ops.py`` | MLIR files of ``tests/testdata/<plan>/`` | ``get_test_cases_from_files`` |

To see what a test plan or a folder collects:

```
$ pytest tests/ops/test_tosa_ops.py --collect-only -q
$ pytest tests/compiler --collect-only -q
```
