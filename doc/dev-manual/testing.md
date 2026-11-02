# Testing

The compiler tests use ``pytest``. A test compiles a model for Torq, runs it on the simulator (or
on hardware), and compares its outputs with a reference: TFLite, ONNX Runtime, PyTorch, or IREE
on the CPU.

The testing documentation has four pages:

| You want to | Read |
|---|---|
| Write your first test case | This page, the tutorial below |
| Do a specific task: run some tests, add a case of another kind, mark a known failure, choose the CI jobs | [Testing How-to Guides](testing_howto.md) |
| Understand how the framework works, from pytest fixtures to CI | [Testing Architecture](testing_architecture.md) |
| Look up a marker, an option, a file format or a CI label | [Testing Reference](testing_reference.md) |

Opt-in suites that are not run by default are in [More testing](more_testing.md).

## Tutorial: your first test case

In this tutorial you add a test of an average pool to the Keras op tests, run it on the
simulator, and see what happens when it fails. It takes about 10 minutes.

### Before you start

You need a built compiler and runtime (see [Getting started](getting_started.md)) and the
development virtual environment. It is active in the development docker. In a native build,
activate it:

```
$ source ../venv/bin/activate
```

Run the commands from the root of the repository.

### 1. Add the case

Each test case of the Keras op tests is a small Keras model. The models are grouped by op family
in ``tests/testdata/keras_ops/``. The pools are in ``pooling.py``, in a list of layer settings.
Add a 2x2 average pool with ``same`` padding on a 9x9x4 input to the list:

```python
_LAYERS = [
    ...
    ("avgpool", (2, 2), (2, 2), "same", (9, 9, 4)),
]
```

That's all the code: ``pooling.py`` turns each line into a model, and names it after its
settings, ``avgpool_k2x2_s2x2_same_9x9x4``.

### 2. List its tests

```
$ pytest tests/ops/test_keras_ops.py -k avgpool_k2x2_s2x2_same_9x9x4 --collect-only -q
tests/ops/test_keras_ops.py::test_keras_model_llvm[avgpool_k2x2_s2x2_same_9x9x4_int8-sim-sl2610-v1]
tests/ops/test_keras_ops.py::test_keras_model_llvm[avgpool_k2x2_s2x2_same_9x9x4_int16-sim-sl2610-v1]
tests/ops/test_keras_ops.py::test_keras_model_torq[avgpool_k2x2_s2x2_same_9x9x4_int8-sim-sl2610-v1]
tests/ops/test_keras_ops.py::test_keras_model_torq[avgpool_k2x2_s2x2_same_9x9x4_int16-sim-sl2610-v1]
```

Your case gives four tests. Each test name says what runs:

- ``test_keras_model_torq`` compiles the model for Torq and compares the results with TFLite.
  ``test_keras_model_llvm`` checks the reference path (IREE on the CPU against TFLite) and is
  not run in CI.
- ``_int8`` and ``_int16``: each Keras model is quantized to both.
- ``sim-sl2610-v1``: the runtime (the C model simulator) and the chip.

### 3. Run it

```
$ pytest tests/ops/test_keras_ops.py -k avgpool_k2x2_s2x2_same_9x9x4 -m ci --torq-chips sl2610-v1
...
2 passed, 1334 deselected in 9.59s
```

``-m ci`` selects the tests that CI runs, here the two ``test_keras_model_torq`` tests.
``--torq-chips`` names the chip to compile for.

Run the same command again: it ends in a fraction of a second. The framework keeps the results of
each step (the quantized model, the compiled model, the reference, the outputs) in
``.pytest_cache``, and redoes a step only when one of its inputs changes. For example, it redoes
the compilation when you rebuild the compiler.

Now run it for the other chip, SR200:

```
$ pytest tests/ops/test_keras_ops.py -k avgpool_k2x2_s2x2_same_9x9x4 -m ci --torq-chips sr200-v1
```

### 4. When a test fails

When the outputs differ from the reference, the test prints how much they differ and the
command to look at the two tensors:

```
To display the difference between expected and observed tensor run:
scripts/diff-tensor.py .../output_observed.npy .../output_expected.npy
Max relative difference: 1.0
Max absolute difference: 12.0
Number of differences: 25 out of 100 [25.00%]
```

If the compiler fails instead, the error is in the test output. Add ``--debug-ir`` to the command
to save the IR after each compiler pass (see
[Testing How-to Guides](testing_howto.md#debug-a-failing-test)).

If the failure is a bug that your pull request doesn't fix, file an issue and declare the known
failure in ``tests/ops/test_keras_ops-xfails.txt``, one line per test, with the issue:

```
test_keras_model_torq[avgpool_k2x2_s2x2_same_9x9x4_int8-sim-sl2610-v1] synaptics-torq/torq-compiler-dev#1234
```

The test is now reported as ``xfailed`` and not run.

### 5. Open the pull request

``test_keras_model_torq`` has the ``ci`` marker, so the C model CI job runs your case on every
pull request, for SL2610 and SR200. You don't need to change anything else.

### Next steps

- To add a PyTorch, ONNX or MLIR case, or the layers of a model, see
  [Add a test case](testing_howto.md#add-a-test-case).
- To understand what happened between the case and the result, read
  [Testing Architecture](testing_architecture.md).
