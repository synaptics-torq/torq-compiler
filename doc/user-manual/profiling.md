# Profiling

Profiling helps you understand the performance characteristics of your models on the Synaptics backend. This includes:
- **Compile-time profiling**: Estimates of approximate clock cycles per operation.
- **Runtime profiling**: Actual execution times for each block during inference.

---
## Compile-Time Profiling

- Use profiling to identify performance bottlenecks, optimize memory usage, and better understand hardware behavior.

- Torq profiling adds memory footprints and approximate clock cycles information in the MLIR output.
    By default it is disabled, to enable it add option ``--torq-enable-profiling``
    to the ``torq-compile`` command. By default profiling is written to `timeline.csv`. The profiling details can be written to a different file using ``--torq-dump-profiling=path/to/trace.csv``

- Compile the model using torq-compile with the profiling flag:
    ```shell
    $ torq-compile tests/testdata/tosa_ops/add.mlir -o model.vmfb --torq-enable-profiling --torq-dump-profiling=./trace.csv --dump-compilation-phases-to=./compilation_phases
    ```
> **Note:** The `--dump-compilation-phases-to` flag dumps the debug information into a specified directory. These debug files are later used to annotate the runtime profiling results.

- Understanding trace.csv Output

  The `trace.csv` file contains a timeline of estimated execution steps, memory operations, and kernel invocations emitted during compile-time profiling.

  **Column Breakdown:**

  | Column   | Description                                                        |
  |----------|--------------------------------------------------------------------|
  | ID       | Unique identifier. Convention: DI# (DMA_IN), DO# (DMA_OUT), S# (compute op). |
  | Start    | Start time in cycles of this operation.  |
  | End      | End time in cycles.                                                |
  | Location | MLIR source location that generated this operation. Helpful for tracing. |
  | Op Type  | Type of operation — e.g., DMA_IN, DMA_OUT, fully_connected, etc.   |


## Runtime Profiling

```{warning}
**This release only supports host-based simulation which is not cycle accurate**. The reported timing will not be representative of the real performance; the documentation in this section shall be considered as a preview of the profiling mode available on the upcoming release.
```

- Runtime profiling records the actual execution time for each code block when the model is run on simulation. This will print the time each individual block of code takes to execute.

- Run the model using torq-run-module with the profiling flag:
    ```shell
    $ torq-run-module --module=model.vmfb --input="1x56x56x24xi8=1" --torq_profile_host=./runtime.csv
    ```
- Understanding `runtime.csv` Output

  The `runtime.csv` file contains detailed timing information per execution block. This helps analyze actual performance, identify slow paths, and correlate them with model structure.

  **Column Breakdown:**

  | Column           | Description                                                              |
  |------------------|--------------------------------------------------------------------------|
  | actionIndex      | Runtime action index (starting from 0).                                  |
  | elapsed_time(us) | Time elapsed (μs) since the previous operation.                          |
  | timestamp(us)    | Timestamp (μs) of the event                                              |
  | event            | Type of action                                                           |
  | location         | MLIR source location that generated this operation. Helpful for tracing. |


- To annotate the runtime profiling ``runtime.csv``, use annotate_profiling.py by passing the runtime.csv file along with the executable-targets phase dump file, which you can obtain using the ``--dump-compilation-phases-to`` flag during compilation.

  ```shell
  $ annotate_profiling.py ./compilation_phases/add.9.executable-targets.mlir ./runtime.csv ./annotated_runtime.csv
  ```

- This enriches the trace with hardware-level details such as actual DMA operations, kernel launch times, and usage of slices.

  **Annotated CSV Columns:**

  | Column | Description |
  |--------|-------------|
  | action_id | Index of the runtime action |
  | job_id | Index of the NSS job associated with the action (if applicable) |
  | operation | Action being performed |
  | invocation_names | Name of the slice programs invoked during the action (if applicable) |
  | original_operator | Original operator from input model |
  | total_time | Total duration of the operation in μs |
  | slice_used_0_in_program | Flag indicating if slice 0 was used by this action |
  | slice_used_1_in_program | Flag indicating if slice 1 was used by this action |
  | dma_in_used_in_program | Flag indicating if DMA input was used by this action |
  | dma_out_used_in_program | Flag indicating if DMA output was used by this action |
  | cdma_used_in_program | Flag indicating if CDMA was used by this action |
  | css_used_in_program | Flag indicating if CSS was used by this action |
  | timestamp_start | Start timestamp in μs |
  | timestamp_end | End timestamp in μs |
  | location | MLIR source location that generated this operation |


## Profiling with the Test Framework

When running tests on real hardware via pytest, you can combine runtime profiling with
``--update-astra-runtime`` to get a profiling summary printed at the end of the session:

```bash
pytest tests/test_onnx_model.py -k example-matmul_layer \
  --torq-runtime-hw-type=astra_machina \
  --torq-addr root@10.46.130.17 \
  --torq-runtime-profiling-output-dir=./profile \
  --update-astra-runtime \
  --recompute-cache
```

At the end of the session, a **Host Profile Overview** section is printed with a
summary extracted from the Perfetto ``.pb`` trace files:

```
====================== Host Profile Overview =======================
  Model: conv2d_f4_s4_64x64x16_i16
  ────────────────────────────────────────────────────────
  WALL_TIME                       207.000ms
  OVERALL                          12.845ms
  Dma                             453.000µs  (3.53%)
  Dma Only                         46.000µs  (0.36%)
  Dma Total                       453.000µs  (3.53%)
  Cdma                                  0ns  (0.00%)
  Compute                         407.000µs  (3.17%)
  Compute Only                          0ns  (0.00%)
  Slice                           407.000µs  (3.17%)
  Slice 0                         407.000µs  (3.17%)
  Slice 1                         407.000µs  (3.17%)
  Css                                   0ns  (0.00%)
  Overlap                         407.000µs  (3.17%)
  Idle                              6.000µs  (0.05%)
  Host                                  0ns  (0.00%)
  Host Copy                        12.386ms  (96.43%)
```

The summary dynamically displays all available metrics from the trace — no hard-coded
list of keys. ``WALL_TIME`` is the end-to-end time measured on the host (including SSH
overhead), while ``OVERALL`` and the breakdown rows come from the on-device trace.

### Wall-time measurement

When ``--update-astra-runtime`` is active, the framework first syncs your locally built
``torq-run-module`` into your user-specific board path
(``/home/root/iree-build-soc/<username>/torq-run-module``) and then measures wall-clock time
around that remote invocation using Python's ``time.monotonic()``. The result is:

- Logged during the test (``Wall time: 1.234s``)
- Saved as ``wall_time.txt`` in the test results directory
- Attached to the test report via ``record_property("wall_time", ...)``
- Printed at the top of the Host Profile Overview summary
