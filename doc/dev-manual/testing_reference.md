# Testing Reference

The settings, files and CI facts of the test framework, for looking things up. Each section
names its source of truth, the file or command that defines it: when this page and the source
disagree, the source is right (and this page needs a fix). For how these pieces fit together, see
[Testing Architecture](testing_architecture.md).

## Test folders

Source: the folders of ``tests/``. Each folder holds one test plan, with its known failures and
helper modules.

| Folder | What its tests check |
|---|---|
| ``tests/ops/`` | One op or a short op pattern, compiled, run and compared with a reference |
| ``tests/models/`` | A full model, and its layers or blocks |
| ``tests/compiler/`` | One compiler behavior, without running the model |
| ``tests/runtime/`` | The execution paths: host, CSS, heterogeneous runs, the runtime API |
| ``tests/harness/`` | Any model file, layer by layer; for investigations, not in CI |
| ``tests/lab/`` | ``torq-lab`` and its tools, without the compiler |
| ``tests/framework/`` | The ``torq.testing`` plugin |
| ``tests/sweeps/`` | Opt-in sweeps, not run by default |

``tests/testdata/`` holds the cases and models shared by the test plans, and ``extras/tests/`` the
tests of internal models.

## Test names

Source: pytest node IDs.

```
tests/ops/test_tosa_ops.py::test_mlir_files_torq[add-single-rescale.mlir-sim-sl2610-v1]
<test module>             ::<test function>     [<case>-<runtime>-<chip>]
```

- **Case:** the name of a ``Case``, e.g. a file name or a Keras case name with its quantization
  (``avgpool_k3x3_s3x3_valid_10x10x4_int8``).
- **Runtime:** ``sim``, ``aws_fpga`` or ``astra_machina``.
- **Chip:** a chip name, e.g. ``sl2610-v1``.

The ``-xfails.txt`` files and the ``-k`` option use these names.

## Markers

Source: ``pytest.ini`` (``pytest --markers`` lists them).

| Marker | Meaning |
|---|---|
| ``ci`` | Run by the C model CI job |
| ``fpga_ci`` | Run by the AWS FPGA CI job |
| ``astra_machina_sl_ci`` | Run by the Astra Machina SL board CI job |
| ``release_only`` | Run by the C model CI job in the merge queue and on ``main``; on a pull request only with the ``run-release-only`` label |
| ``not_fpga_ci`` | Not run by the FPGA CI job, even when its test function is ``fpga_ci`` |
| ``int8``, ``int16`` | Keras op cases of one quantization |
| ``pooling``, ``conv``, ... | Keras op cases of one family module (the module name) |
| ``layer``, ``full_model`` | Layer and full-model cases of the TFLite model tests |
| ``conv_sweep``, ``gen_config_standalone`` | Opt-in suites, deselected by default (see [More testing](more_testing.md)) |

An explicit ``-m`` replaces the default selection of ``pytest.ini``, which deselects the opt-in
suites.

## Options

Source: ``pytest --help``, section "Custom options". The most used ones:

| Option | What for |
|---|---|
| ``--torq-chips CHIPS`` | The chips to compile for: chip names, comma-separated, or ``all``. Default ``sl2610-v1`` |
| ``--torq-runtime-hw-type TYPES`` | The runtimes: ``sim`` (default), ``aws_fpga``, ``astra_machina`` |
| ``--recompute-cache`` | Redo every step, ignoring the cache |
| ``--debug-cache`` | Print why each cached step is reused or redone |
| ``--remote-cache-read STORES``, ``--remote-cache-write STORE`` | Read cached results from stores (directories or ``s3://bucket/prefix``, comma-separated), and write the computed ones to a store. Default ``$TORQ_REMOTE_CACHE_READ``, ``$TORQ_REMOTE_CACHE_WRITE`` |
| ``--try-xfails`` | Run the known failures, and report those that pass (``XPASS``) |
| ``--issue ISSUE`` | Run only the tests of the known failures of an issue |
| ``--check-xfails`` | Fail when a line of an ``-xfails.txt`` file matches no test |
| ``--save-test-status-report-to FILE`` | Write the outcome of each test, for ``torq.testing.issues`` |
| ``--slow-tests-first`` | Start the tests of ``slow-tests.txt`` first, spread over the xdist workers |
| ``--debug-ir [DIR]`` | Save the IR after each compiler pass |
| ``--no-output-capture`` | Print the whole compiler and runtime output |
| ``--extra-torq-compiler-options OPTS`` | Add compiler options to every test |
| ``--torq-compiler-timeout S``, ``--torq-runtime-timeout S`` | Time limits of one compilation and one run (default 300 s and 240 s) |
| ``--torq-runtime-profiling-output-dir DIR`` | Save a runtime profile of each test |
| ``--torq-compile-time-profiling-output-dir DIR`` | Save the time of each compiler pass of each test |
| ``--torq-step-timings FILE`` | Save where the time of each test goes, and print the slowest tests |
| ``--tflite-layer-test-generate-full-model-only`` | TFLite model tests: only the full models, not their layers |

## Files

| File | Content |
|---|---|
| ``tests/<folder>/test_x-xfails.txt`` | The known failures of ``tests/<folder>/test_x.py``, next to it: ``<test function>[<test parameters>] <issue>`` per line, ``#`` for comments |
| ``extras/xfail/test_x-xfails.txt`` | The same, for internal chips and models |
| ``tests/slow-tests.txt``, ``extras/tests/slow-tests.txt`` | The slow tests, slowest first: ``<test file name>::<test name>`` per line (the file name without its folder), ``*`` as the only wildcard. A test matches the first line it fits |
| ``tests/ops/keras_ops-not-fpga-ci.txt`` | Keras op case names that get the ``not_fpga_ci`` marker |
| ``tests/ops/keras_known_failures.py`` | ``SKIP_TESTS``: Keras op cases that are skipped (they crash the test process) |
| ``tests/testdata/chips/<chip>.json`` | A chip description; ``extras/chips`` for internal chips |
| ``tests/testdata/keras_ops/*.py`` | Keras op cases, see [Keras module](#keras-module) |
| ``tests/testdata/torch_ops/*.py`` | PyTorch and ONNX op cases, see [PyTorch and ONNX module](#pytorch-and-onnx-module) |
| ``tests/testdata/{tosa,linalg,stablehlo}_ops/*.mlir`` | MLIR op cases, see [MLIR file](#mlir-file) |
| ``tests/testdata/*_host_css/`` | Cases that may run on the host or the CSS |
| ``tests/testdata/linalg_compilation/*.mlir`` | Compile-only cases of ``tests/compiler/test_linalg_compilation.py`` |
| ``.pytest_cache/d/versioned_fixtures/`` | The cache; can be deleted at any time |

## Case settings

### Keras module

Source: the docstring of ``tests/ops/keras_ops_models.py``.

| Name | Content |
|---|---|
| ``CASES`` | ``{case name: builder}``; ``builder()`` returns the Keras model (``torq.testing.keras_builder``) |
| ``QUANTIZE`` | The quantizations of every case, default ``("int8", "int16")``: case names ``<name>_int8``, ``<name>_int16``. A single string (``"int8"``) gives one case named ``<name>`` |
| ``DEFAULT_CONFIG`` | Settings of every case of the module |
| ``CONFIG`` | ``{case name: settings}``, with the keys ``quantize`` (as ``QUANTIZE``), ``compiler_options`` (added torq-compile options), ``comparison`` ([comparison settings](#comparison-settings)), ``int8``, ``int16`` (settings of one quantization only) |

### PyTorch and ONNX module

Source: the docstring of ``tests/ops/torch_ops_models.py``. The module is read without being
imported: these names must be plain literals.

| Name | Content |
|---|---|
| ``KIND`` | ``"torch"`` (builders return an ``nn.Module``) or ``"onnx"`` (builders return an ``onnx.ModelProto``) |
| ``CASES`` | ``{case name: builder}``: a function of the module or a ``functools.partial`` of one. Names starting with ``disable`` are skipped |
| ``IO`` | PyTorch only: ``{case name: [[shape, dtype], ...]}``, the example inputs of ``torch.export`` |
| ``CONFIG`` | ``{case name: settings}``, with the keys ``compiler_options`` (replace the default ones), ``comparison``, ``reference`` (the name of the reference fixture), ``input_range`` (``[low, high]`` of the random inputs), ``sim_runtime_timeout`` (seconds) |

### MLIR file

Sources: ``comparison_config_from_mlir`` in ``python/torq/testing/iree.py``, the ``CONFIG``
table of ``tests/ops/test_tosa_ops.py``.

- **Tolerances:** comments ``// TORQ_<KEY>: <number>``, with a [comparison setting](#comparison-settings)
  in upper case, e.g. ``// TORQ_FP_MAX_TOL: 0.05``.
- **Other settings of a tosa_ops file:** the ``CONFIG`` table of ``test_tosa_ops.py``,
  ``{file stem or pattern: settings}``. A pattern uses ``*`` and ``?`` (``"identity-*"``).
  The keys are ``torq_compiler_options`` (added), ``torq_compiler_timeout`` and
  ``tweaked_input_data_range`` (``(low, high)`` of the random inputs). Settings under an
  ``"aws_fpga"`` key apply on the FPGA only. Every entry must match a file.
- **linalg_ops** sets its per-file settings in the ``case_config`` of ``test_linalg_ops.py``.
- **Compile-only cases** (``linalg_compilation``): ``// TORQ_COMPILE_OPTIONS: --flag1 --flag2``.

## Comparison settings

Source: ``DEFAULT_COMPARISON_CONFIG`` and ``compare_outputs`` in
``python/torq/lab/verification/compare.py``.

Integer outputs pass when every element differs from the reference by at most ``int_tol``. For
float outputs, the relative difference of an element is
``|observed - expected| / (|observed| + |expected| + epsilon)``; they pass when the fraction of
elements whose relative difference is over ``fp_max_tol`` is at most ``allowed_wrong``.

| Key | Default | Meaning |
|---|---|---|
| ``int_tol`` | 1 | Integers: largest allowed difference |
| ``int_thld`` | 1 | Integers: has no effect, since an element over ``int_tol`` already fails the test |
| ``fp_max_tol`` | 0.01 | Floats: an element over this relative difference is wrong |
| ``allowed_wrong`` | 0 | Floats: largest allowed fraction of wrong elements |
| ``use_abs_tol_gate`` | false | Floats: an element is wrong only if its absolute difference is also over ``fp_abs_tol_frac`` times the largest expected value |
| ``fp_abs_tol_frac`` | 0 | See ``use_abs_tol_gate`` |
| ``fp_avg_tol`` | 0.01 | Floats: the relative difference over which an element is counted in the "number of differences" printed on failure; it doesn't decide the result |
| ``epsilon`` | 1e-6 | Floats: added to the denominator of the relative difference |
| ``allow_all_zero`` | false | Accept an output that is all zeros when the reference is not |
| ``skip_nan_check`` | false | Floats: don't check that NaNs are at the same places |

## CI

Sources: ``.github/workflows/test.yml`` (pull requests and merge queue), ``release.yml``
(pushes), ``test-build.yml`` and the job workflows it calls.

### Jobs

| Job (workflow) | Selection | Runtime, chips | Workers |
|---|---|---|---|
| C model (``test-cmodel.yml``) | ``-m ci``; on pull requests ``ci and not release_only``, unless ``run-release-only`` | ``sim``; one job per chip of the ``TEST_CHIPS`` variable | 18, ``--dist worksteal --slow-tests-first`` |
| AWS FPGA (``test-aws-fpga.yml``) | ``-m "fpga_ci and not not_fpga_ci"`` | ``aws_fpga``; one job per chip of the ``FPGA_TEST_CHIPS`` variable (SL2610 runs on the boards instead) | ``auto``, ``--dist worksteal --slow-tests-first`` |
| Astra Machina SL (``test-astra-machina-sl.yml``) | ``-m astra_machina_sl_ci`` | ``astra_machina``; one job per chip of the ``BOARD_TEST_CHIPS`` variable | 4 |
| torq-lab (``test-torq-lab.yml``) | ``tests/lab/test_lab_*.py``, without the ``torq.testing`` plugin | none | 1 |

- **When:** the C model and torq-lab jobs run on every pull request. The FPGA and board jobs run
  in the merge queue and on pushes, and on a pull request only with their label. They start after
  the C model jobs pass (or are skipped).
- **C model options:** ``--check-xfails``, ``--durations=30``, ``--torq-step-timings``, and
  compile-time profiles. The step timings and profiles are in the artifact
  ``cmodel_compile_profiling_logs_<chip>``.
- **All jobs** pass ``--tflite-layer-test-generate-full-model-only``. They check out ``extras``
  at the commit of ``scripts/extras.gitlink`` (update it with ``scripts/pin_extras.sh``), and
  upload their results to the test dashboard.

### Time limits

| Limit | Value | Set in |
|---|---|---|
| pytest | 40 min (``timeout 2400``); the tests still running are lost | ``.github/actions/tests/action.yml`` |
| C model, FPGA and board jobs | 50 min (``timeout-minutes``) | the job workflows |
| CodeBuild runner | 60 min | the runner |
| torq-lab job | 20 min | ``test-torq-lab.yml`` |

### Labels

| Label | Effect on a pull request |
|---|---|
| ``test-aws`` | Run the FPGA jobs |
| ``test-astra-machina-sl`` | Run the Astra Machina board job |
| ``skip-test-cmodel`` | Don't run the C model jobs; the FPGA and board jobs still run if labeled |
| ``skip-test-torq-lab`` | Don't run the torq-lab job |
| ``run-release-only`` | Run the ``release_only`` tests in the C model jobs |
| ``generate-test-status-report`` | Run the known failures too (``--try-xfails``), attach a status report per job, don't fail on test failures |
| ``test-stage-dashboard`` | Upload the results to the staging dashboard instead of production |

Adding a label starts a new run; removing one doesn't. Re-running a workflow run reuses the
labels it had when it started.
