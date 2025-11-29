Model Profiler
==============

### TLDR;

The script converts timeline.csv from profiling pass to Perfetto log. The Perfetto log is saved as proto buffer.

### **Purpose**
This script converts timeline profiling logs (from model compile and runtime stages) stored in CSV format into a **Perfetto protobuf trace file (`.pb`)**. The resulting trace can be visualized using Perfetto's trace viewer for performance analysis.

---

### **Key Features**
- Supports both **compile-time** and **runtime** profile formats.
- Organizes logs into Perfetto’s process/thread view hierarchy.
- Automatically serializes the logs into a `.pb` binary trace file.


Dependency:
-----------
* `protobuf 5.29.3` is used
* `libprotoc 28.0` for `protoc` 

### Steps:

1. Generate compile time profiling using flag `--torq-enable-profiling` and `--torq-dump-profiling=./compile_profile.csv`
```
/buildsrc/iree-build/third_party/iree/tools/torq-compile /buildsrc/iree-synaptics-synpu/tests/testdata/tosa_ops/conv-stride1-notile.mlir -o ./output_torq.vmfb --dump-compilation-phases-to=output_torq_phases --mlir-print-ir-after-all --mlir-print-ir-tree-dir=output_torq_passes  --torq-enable-profiling --torq-dump-profiling=./compile_profile.csv
```

2. Generate runtime profiling using flag `--torq_profile=./runtime_profile.csv`. Create sample input `in_rnd_0.npy`.
```
/buildsrc/iree-build/third_party/iree/tools/iree-run-module --device=torq --module=./output_torq.vmfb  --function=main --output=@./output_torq-cmodel.npy --input=@./in_rnd_0.npy  --torq_profile=./runtime_profile.csv
```

Sample code to generate `in_rnd_0.npy`:
```
import numpy as np
np.random.seed(0)
dtype_info = np.iinfo(np.int8)
shape = [1, 224, 224, 3]
random_array = np.random.randint(dtype_info.min, dtype_info.max + 1, size=shape, dtype=np.int8)
np.save('in_rnd_0.npy', random_array)
```

3. Run `perfetto_logger.py` with inputs profile `compile_timeline.csv` and, runtime profile to get output. The output will be at `perfetto_log.pb` for the below command
```
python3 perfetto_logger.py --compile_profile ./compile_profile.csv --runtime_profile ./runtime_profile.csv --pb ./perfetto_log.pb
```

4. You can also specify multiple compile profiles or multiple runtime profiles or both.
```
python3 perfetto_logger.py --compile_profile ./compile_profile.csv ./compile_profile.csv --pb ./perfetto_log.pb
```
```
python3 perfetto_logger.py --runtime_profile ./runtime_profile.csv ./runtime_profile.csv --pb ./perfetto_log.pb
```
```
python3 perfetto_logger.py --compile_profile ./compile_profile.csv ./compile_profile.csv --runtime_profile ./runtime_profile.csv ./runtime_profile.csv --pb ./perfetto_log.pb
```

#### Generation of perfetto_api.py
`perfetto_api.py` is generated using `protoc` from Perfetto protobuf definition.
`perfetto_api.py` is used to create log protobuf.

1. Install `protoc`.
```
https://google.github.io/proto-lens/installing-protoc.html
```

2. Clone Perfetto repo.
```
git clone https://github.com/google/perfetto.git
```

3. Generate `perfetto_trace_pb2.py` using `protoc`. After running below commands `perfetto_trace_pb2.py` will be created in `temp` dir.
```
cd ./perfetto/protos/perfetto/trace
mkdir temp
protoc -I=. --python_out=./temp perfetto_trace.proto
```

4. Rename `temp/perfetto_trace_pb2.py` to `perfetto_api.py`.

API DOC:
-------

### `class TimeChartLog`
Represents a single log entry to be added to the Perfetto trace.

| Attribute   | Description                     |
|-------------|---------------------------------|
| `process`   | Process name                    |
| `thread`    | Thread name (usually event type)|
| `function`  | Event label / description       |
| `start_time`| Start timestamp (in ticks)      |
| `end_time`  | End timestamp (in ticks)        |

---


### `class TimeChartView`

Manages Perfetto trace generation.

#### Key Methods:

- **`__init__(filename)`**: Initializes trace file and internal UUID counters.
- **`render(log)`**: Renders a `TimeChartLog` by generating `TYPE_SLICE_BEGIN` and `TYPE_SLICE_END` events.
- **`add_process_descriptor(name)`**: Registers a process in the Perfetto trace.
- **`add_thread_descriptor(process, thread)`**: Registers a thread under a process.
- **`flush()`**: Writes serialized protobuf trace data to disk.
- **`set_track_event(time, type, log)`**: Adds a single Perfetto track event.

---

### `csv_to_view_compile_profile(view_name, view, csv_path, num_slices)`
Parses **compile-time profiling CSV** and populates the `TimeChartView`.

**CSV format**:
```
event_name, start_time, end_time, mlir_src_line, event_type
```

Handles:
- `S<n>` for slices
- `DI`, `DO` for DMA ops

---

### `csv_to_view_runtime_profile(view_name, view, csv_path)`
Parses **runtime profiling CSV** with `;` as delimiter.

**CSV format**:
```
dispatch_id;time_since_open;time_since_start;mlir_loc
```

---

### `convert_to_perfetto(compile_profile_logs, runtime_profile_logs, pb_path, num_slices=2)`
Main function that converts input logs into a Perfetto trace file.

| Argument             | Description                                      |
|----------------------|--------------------------------------------------|
| `compile_profile_logs` | Dict of named compile CSV logs                |
| `runtime_profile_logs` | Dict of named runtime CSV logs                |
| `pb_path`              | Output path for `.pb` trace file              |
| `num_slices`           | Number of ML slices to modulo-label slice IDs |

---


References:

1. https://perfetto.dev/docs/reference/synthetic-track-event
2. https://github.com/google/perfetto
