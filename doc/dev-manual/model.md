# Model testing and debugging workflow

```{note}
This page is under construction
```

The torq-compiler should by default produce an optimized model for hardware, however
sometimes the results may not be what we want. This chapter explains how to leverage
the fact torq-compiler is open source and that its behaviour can be changed.

In the following we imagine we have a ONNX model and we want to make it run as fast
as possible on the torq hardware.

After installing the torq development environment you can start by adding the model
in the testing framework. This framework automatically generates useful test cases
to improve the compilation of a model.

The guide assumes you installed the development environment for the compiler and 
compiled it. It also assumes you activated the compiler development environment


## Running test suite for a model

### Initial setup

To run a test suite for a model add the ONNX file in the ``tests/testdata/onnx_models``.

In the rest of this guide we will use the test ``example-matmul.onnx`` already present
in the directory.

The pytest framework will automatically detect the model and create a set of test
cases for it for any new file found in this directory.

## Collect all the tests for a model

To view the test cases run the following command:

```
$ pytest tests/test_onnx_model.py -k example-matmul --collect-only
```

The result will look as follows:

```
<Package tests>
  <Module test_onnx_model.py>
    <Function test_onnx_model_llvmcpu_torq[example-matmul_layer_MatMul_0-sim-default]>
    <Function test_onnx_model_llvmcpu_torq[example-matmul_full_model-sim-default]>
```

The framework created a test case for the full model (denoted with the ``_full_model`` suffix) 
and a set of test cases for each of the layers of the model (denoted with the corresponding 
layer name). 

## What the tests do

The tests is defined in the function ``test_onnx_model_llvmcpu_torq`` in the ``tests/test_onnx_model.py``.

The test compiles the model for Torq simulation (using torq-compile) and for the 
host CPU (using iree-compile and the LLVMCPU backend), runs them against some random inputs
and compares the results.

The test returns ERROR if the any of the steps required to compute the two inference 
results fail and return FAIL if the two inputs are not sufficiently similar.



## Executing the tests

To run the tests use the following command line:

```
$ pytest tests/test_onnx_model.py -k example-matmul
```

Pytest will provide a report with all the successes, errors and failures and potentially related error logs.

In the ideal situation all the tests will pass, in some cases, especially if models contain layers that are not yet supported
a subset of the test will succeed while some will fail.

Once the tests have run it is possible to interpret the results as follows:

1) If the test with name ``_full_model-sim-default`` it means that the full model was successfully executed and the results
   of the model are similar to IREE default LLVMCPU backend

2) If that's not the case typically a subset of the other tests have passed, indicating the layers that were successfully
   compiled and their output validated.

To further understand failures it is possible to inspect the output of pytest. This will contain the exact point in the 
testing procedure that created a problem.

Issues may be problems during compilation or simulation (these issues are marked as ERROR) or mismatch of outputs with the
reference results (these are marked as FAIL).

The two categories of problems need to be investigated differently.

## Investigating ERORRs

The first step to investigate an error is pinpoint which step in the testing procedure failed. This can be identified by
looking the stack trace provided by PyTest. In the line that starts with ``kwargs`` it is possible to see the sub-request
that failed, in particular the name of the fixture that failed.

For instance:

```
kwargs = {'llvmcpu_compiler': VersionedUncachedData(data='...<SubRequest 'torq_compiled_model_dir' for <Function test_onnx_model_llvmcpu_torq[mnist-1_full_model-sim-default]>>, ...}
```

Shows that the fixture ``torq_compiled_model_dir`` failed to execute. This means that the test framework was not able to compile
the model with  ``torq-compile``. When an external program failed it is possible to find the stdout and stderr of the invocation
in the same pytest output log.

Errors in the full model are often associated with errors in some of the layer tests. It is often better to make sure all 
per-layer tests pass before diving into the full model errors.

Typical errors are:

- Errors while converting from ONNX to MLIR (fixture **onnx_mlir_model_file**). These errors are typically due to an unsupported
  version of ONNX being used. Converting the model to a recent version may help.

- Errors while compiling with IREE (fixture **llvmcpu_compiled_model**). These errors may be due to an unsupported version of 
  ONNX or operator

- Errors while compiling with ``torq-compile`` (fixture **torq_compiled_model_dir**)

## Investigating FAILUREs

Debugging failures is more complex as the compiler fully executed the model but found a mismatch in the outputs.

The pytest result provides a command line that can be run to execute an output comparison tool. This helps to visualize
the difference between expected and actual results of the computation.

Accuracy failure in the full model are often associated with failures in some of the layer tests. It is often better to make
sure all per-layer tests pass before diving into the full model errors.

Accuracy tests failures, especially with floating point models, may sometimes be false negatives. Floating point operations
are executed differently on CPU and TORQ and the exact numerical results are often different. In some cases the default
tolerance of the testing framework is too strict and the tests fails even if this is not actually a problem.

## Debugging a test

In order to execute a single test you can invoke pytest as follows:

```
$ pytest tests/test_onnx_model.py -s -k example-matmul_layer_MatMul_0-sim-default
```

The ``-s`` option allows to see the output of the tools called by the testing framework while the test is running.

When debugging an error while compiling with ``torq-compile`` it is often useful to inspect the IR being
produced by the compiler. This allows to narrow down to the pass that actually is failing (or by inspecting
the IR the pass that generated the problematic IR).

This can be done by using the following command line:

```
$ pytest tests/test_onnx_model.py -s -k example-matmul_layer_MatMul_0-sim-default --debug-ir ir_dump
```

The IR produced by the compiler will be stored in the directory ``ir_dump``. Make sure you clean this directory
between runs.


## Performance debugging

Debugging the performance of the model to detect bottlnecks can be done with the same testing framework
using two different approaches:

- Compile time profiling: the compiler estimates the time operation take on Torq hardware and creates a report
- Runtime profiling: the runtime records the timestamp for the operations and creates a report

The first approach provides a very rough estimate while the second approach provides a more accurate measurement. In order
to provide realistic results however runtime profiling must executed using real hardware as the simulation is not time 
accurate.

To perform compile time profiling use the following command line option:

```
pytest tests/test_onnx_model.py -k example-matmul_layer --torq-compile-time-profiling-output-dir=profile  --recompute-cache
```

The compile time logs are going to be available in the directory ``profile``.

To perform runtime profiling use the following command line option:

```
pytest tests/test_onnx_model.py -k example-matmul_layer --torq-runtime-profiling-output-dir=profile --recompute-cache
```

The compile time logs are going to be available in the directory ``profile``.

Both runtime and compile time performance tracing produce both ``.csv`` files and ``.pb``  files that can be viewed with
the [Perfetto tool](https://ui.perfetto.dev/).

## Executing tests using real hardware

The test by default run on an host based simulator (as denoted by the "-sim" suffix). It is possible to enable tests on
the hardware with the following command line:

```
pytest tests/test_onnx_model.py -k example-matmul_layer --torq-runtime-hw-type=astra_machina --torq-addr ${board_address}
```

where ``${board_address}`` denotes the address of an astra board (e.g., ``root@10.3.120.55``).

### Auto-deploying the runtime to the board

When iterating on the runtime binary (``torq-run-module``) or the NPU kernel module, use
``--update-astra-runtime`` to automatically deploy your locally built artifacts to the board
before running tests:

```
pytest tests/test_onnx_model.py -k example-matmul_layer \
  --torq-runtime-hw-type=astra_machina \
  --torq-addr ${board_address} \
  --update-astra-runtime
```

This flag enables:

- **Auto-deploy torq-run-module**: the locally cross-compiled binary is compared (MD5) against
  the one in your user-specific board path and copied only if it differs.
- **Per-user runner paths**: each developer gets an isolated binary on the board under
  ``/home/root/iree-build-soc/<username>/torq-run-module``. Runtime execution uses this
  copied binary, so tests run with the developer's locally built runtime rather than a
  shared system binary.
- **Board hostname validation**: the board's hostname must match the pattern
  ``sl2619-dev-board-NNN`` (e.g. ``sl2619-dev-board-002``).  If it doesn't, pytest
  fails immediately with guidance on how to set the hostname using
  ``scripts/set_board_hostname.py``.
- **Exclusive board access**: a session-level lock is acquired at the start of the
  pytest session and released when it ends.  The lock records the owner's username,
  hostname, and timestamp so others can see who is using the board.  If a previous
  session crashed without releasing the lock you can clear it with:

  ```
  python3 scripts/reset_board_lock.py root@<board-ip>
  ```

  Even if the session is interrupted with Ctrl+C or killed with SIGTERM, the lock
  is released automatically via signal handlers and ``atexit``.
- **Wall-clock timing**: the remote execution time is measured **on the board itself**
  using the shell ``time`` built-in, so the reported value excludes SSH transport
  overhead.  The ``real`` time printed by ``time`` is parsed from the command output
  and recorded in the profiling summary at the end of the session.

### Deploying a custom NPU kernel module

To deploy a custom ``.ko`` file to the board, pass ``--torq-ko-path``:

```
pytest tests/test_onnx_model.py -k example-matmul_layer \
  --torq-runtime-hw-type=astra_machina \
  --torq-addr ${board_address} \
  --update-astra-runtime \
  --torq-ko-path /path/to/syna_npu.ko
```

The framework compares the local ``.ko`` hash with the on-board module at
``/usr/lib/modules/6.12.11/updates/syna_npu.ko``. If they differ, the on-board
copy is replaced and the board is **rebooted** — the pytest session exits with an
error asking you to re-run after the board finishes booting.

``--torq-ko-path`` requires ``--update-astra-runtime`` to be set.

### Finding boards on the network

Three utility scripts help manage boards:

**Scan for reachable boards** (only boards with hostname ``sl2619-dev-board-NNN`` are shown):

```
python3 scripts/scan_boards.py
```

Output:

```
IP                   HOSTNAME
----------------------------------------
10.46.130.17         sl2619-dev-board-001
10.46.130.42         sl2619-dev-board-002
----------------------------------------
2 board(s) reachable.
```

**Search for a specific board by hostname:**

```
python3 scripts/scan_boards.py --find sl2619-dev-board-001
```

**Set a hostname on a board** (persists across reboots):

```
python3 scripts/set_board_hostname.py 10.46.130.17 sl2619-dev-board-001
```

**Reset a stale board lock** (if a previous session crashed without releasing it):

```
python3 scripts/reset_board_lock.py root@10.46.130.17
```

## Using the results dashboard

In order to analyze large number of tests e.g. when the model contains many layers. It may be beneficial to use the
performance dashboard.

The dashboard is a web application following the instructions in ``webapps/dashboard``. Once started you can point
the testing framework to it by setting the following environment variable:

``
export TORQ_PERF_SERVER=http://localhost:8080
``

After each test you will see a link to the results in the dashboard. You can use the dashboard to compare the performance
across different tests and to inspect detailed traces.

## Advanced: Test framework internals and PyTest fixtures

The test leverages a set of [PyTest fixtures](https://docs.pytest.org/en/latest/how-to/fixtures.html) to obtain the different artifacts required
to perform the comparison. Fixtures can in turn depend on other fixtures. The most
important are:

- **onnx_mlir_model_file** : this fixture returns the input model converted to MLIR
  using ``iree.compiler.tools.import_onnx``

- **torq_compiled_model_dir**: the model compiled with ``torq-compile``

- **llvmcpu_compiled_model**: the model compiled with ``iree-compile`` with the LLVMCPU backend

- **tweaked_random_input_data**: random inputs suitable for the model

- **llvmcpu_reference_results** : results of the inference using **tweaked_random_input_data** and the model **llvmcpu_compiled_model**

- **torq_results** : results of the inference using **tweaked_random_input_data** and the model in **torq_compiled_model_dir**

- **chip_config**: the target TORQ-enabled chip for which the model is compiled and simulated (different chips can be enabled with the ``--torq-chips`` command line option)

- **runtime_hw_type**: the target hardware emulation used to run the torq model (can be changed with the ``--torq-runtime-hw-type``)

The source code of the fixtures and their dependecy relatioship can be found by inspecting the files in ``python/torq/testing``.



## TFLite model testing

The testing framework also supports TFLite models. Unlike ONNX models which are imported via ``iree.compiler.tools.import_onnx``, TFLite models go through a different conversion pipeline: ``iree-import-tflite`` (TFLite → TOSA bytecode) followed by ``iree-opt`` (TOSA → text MLIR).

A key difference from the ONNX workflow is how layers are extracted. TFLite layers are extracted by directly manipulating the TFLite flatbuffer, which **preserves quantization parameters exactly** from the original model. This ensures that each extracted layer remains correctly quantized with its original scale and zero-point values.

The test file is ``tests/test_tflite_model.py``. Place your ``.tflite`` model files in
``tests/testdata/tflite_models/`` and the framework will automatically detect them and
create test cases for each layer and the full model.

### Initial setup

Place the ``.tflite`` file in ``tests/testdata/tflite_models/``.
The framework will automatically detect it and create test cases.

### Collecting tests

To view the test cases:

```
$ pytest tests/test_tflite_model.py --collect-only
```

The output will show individual layer tests and a full model test:

```
<Function test_tflite_model_llvmcpu_torq[my_model_layer_CONV_2D_0-sim-default]>
<Function test_tflite_model_llvmcpu_torq[my_model_layer_ADD_3-sim-default]>
...
<Function test_tflite_model_llvmcpu_torq[my_model_full_model-sim-default]>
```

### What the tests do

Each TFLite test case:

1. **Extracts layers** from the TFLite flatbuffer (at collection time). Each layer becomes a standalone ``.tflite`` model preserving the original quantization.
2. **Converts to MLIR** via ``iree-import-tflite`` → ``iree-opt`` (at test runtime, cached by versioned fixtures).
3. **Compiles** the MLIR for Torq simulation (``torq-compile``) and LLVM-CPU (``iree-compile``).
4. **Runs inference** with random inputs on both backends.
5. **Compares results** element-wise between the two backends.

### Environment variables

The following environment variables control test behaviour:

- ``MAX_LAYERS``: limit the number of layers extracted (default: ``0`` = no limit).
  Example: ``MAX_LAYERS=5 pytest tests/test_tflite_model.py -v --collect-only``

- ``FORCE_EXTRACT``: set to ``1`` to force re-extraction of layers even if cached.
  Example: ``FORCE_EXTRACT=1 pytest tests/test_tflite_model.py -v --collect-only``

### Running specific subsets

```bash
# Full model only:
pytest tests/test_tflite_model.py -v -k "full_model"

# Specific layer type:
pytest tests/test_tflite_model.py -v -s -k "layer_CONV_2D"

# A single specific layer:
pytest tests/test_tflite_model.py -v -s -k "my_model_layer_CONV_2D_1"

# Parallel execution:
pytest tests/test_tflite_model.py -v -n 8
```

### How layer extraction works

The layer extraction is performed by ``torq.testing.tflite_layer_extractor.extract_all_layers``.
It works directly on the TFLite flatbuffer:

1. Parses the flatbuffer to enumerate all operators.
2. For each operator, builds a new single-operator TFLite model that includes only the relevant
   tensors, buffers, and quantization parameters.
3. Saves each layer as a standalone ``.tflite`` file in
   ``tests/testdata/tflite_models/.mlir_cache/<model_stem>_layers/``.
4. A ``_cases_cache.json`` is saved alongside so that subsequent runs skip extraction entirely
   and load the test case list from the JSON cache.

The extraction happens at **test collection time** (inside ``pytest_generate_tests``), since pytest needs the list of layers to parametrize the tests. The TOSA/MLIR conversion happens later at **test runtime** via the ``tflite_mlir_model_file`` versioned fixture.

### Caching

TFLite tests use two caching layers:

1. **Layer extraction cache** (``tests/testdata/tflite_models/.mlir_cache/``):
   stores extracted layer ``.tflite`` files and ``_cases_cache.json``. Cleared with
   ``FORCE_EXTRACT=1``.

2. **Versioned fixtures cache** (``.pytest_cache/d/versioned_fixtures/``):
   stores TOSA/MLIR files, compiled models, and inference results. Each artifact is
   keyed by a hash of its inputs, so changes propagate automatically. Cleared with
   ``--recompute-cache``.

### Investigating errors and failures

The same debugging approaches described in the ONNX sections above apply. The key fixture
names differ for TFLite:

- **tflite_mlir_model_file**: converts TFLite → TOSA (via ``iree-import-tflite``) → text MLIR (via ``iree-opt``). Errors here indicate unsupported TFLite operators or conversion issues.

- **torq_compiled_model_dir**: compiles the MLIR with ``torq-compile``. Same as ONNX.

Debugging options (``--debug-ir``, ``-s``, ``--recompute-cache``), performance profiling
(``--torq-compile-time-profiling-output-dir``, ``--torq-runtime-profiling-output-dir``), and hardware execution (``--torq-runtime-hw-type``, ``--torq-addr`` , ``--update-astra-runtime``)
all work identically to the ONNX workflow described above.

### Key TFLite fixtures

- **tflite_layer_model**: provides the test case data (layer path, op name, quantization info)

- **tflite_model_file**: versioned static file fixture wrapping the ``.tflite`` path

- **tflite_mlir_model_file**: converts TFLite → TOSA → MLIR (cached)

- **tweaked_random_input_data**: random inputs used for layer and full model tests

- **llvmcpu_reference_results**: results of inference using LLVM-CPU backend

- **torq_results**: results of inference using Torq backend