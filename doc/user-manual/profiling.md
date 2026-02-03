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
    $ torq-run-module --module=model.vmfb --input="1x56x56x24xi8=1" --torq_profile=./runtime.csv 
    ```
- Understanding `runtime.csv` Output

  The `runtime.csv` file contains detailed timing information per execution block. This helps analyze actual performance, identify slow paths, and correlate them with model structure.

  **Column Breakdown:**

  | Column           | Description                                                        |
  |------------------|--------------------------------------------------------------------|
  | ID               | Operation index (starting from 0).                                 |
  | time_since_open  | Cumulative time (μs) since the task was loaded.                    |
  | time_since_start | Time elapsed (μs) since the previous operation.                    |
  | Location         | MLIR source location that generated this operation. Helpful for tracing. |

- To annotate the runtime profiling ``runtime.csv``, use annotate_profiling.py by passing the runtime.csv file along with the executable-targets phase dump file, which you can obtain using the ``--dump-compilation-phases-to`` flag during compilation.

  ```shell
  $ annotate_profiling.py ./compilation_phases/add.9.executable-targets.mlir ./runtime.csv ./annotated_runtime.csv
  ```

- This enriches the trace with hardware-level details such as actual DMA operations, kernel launch times, and usage of slices.

  **Annotated CSV Column:**

  | Column                          | Description                                                                 |
  |----------------------------------|-----------------------------------------------------------------------------|
  | NSS Program Index               | Index of the low-level program sequence executed by the Synaptics NPU.      |
  | torq-hw Operation in program   | Hardware action being profiled                                              |
  | Slice ID (if applicable)        | Logical slice involved.                                                     |
  | Invocation Name                 | Name of the task or function associated with the slice.                     |
  | NSS Program Start Timestamp [us]| Start time in microseconds.                                                 |
  | NSS Program End Timestamp [us]  | End time in microseconds.                                                   |
  | NSS Program Total Time [us]     | Duration of that program block.                                             |
  | Slice 0/1 Used in NSS Program   | Flags indicating if that slice was active during this block.                |