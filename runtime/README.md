# Synaptics Torq Runtime

The `torq-runtime` Python package provides bindings for loading and running compiled `.vmfb` models on a Torq device directly from Python. It bundles the Torq runtime alongside the IREE runtime Python bindings into a single wheel with both namespaces, together with the `torq-run-module` and IREE CLI tools.

## Installation

```bash
pip install torq-runtime
```

## Quick Start

#### Inference from Python:

```python
import numpy as np
from torq.runtime import VMFBInferenceRunner

# Load the compiled model
runner = VMFBInferenceRunner("mobilenetv2.vmfb", device_uri="torq")

# Prepare input data
input_data = np.random.randint(0, 255, size=(1, 224, 224, 3), dtype=np.int8)

# Run inference
outputs = runner.infer([input_data])
print(f"Inference took {runner.infer_time_ms:.2f} ms")
```

#### Inference via CLI

```bash
torq-run-module --module="mobilenetv2.vmfb" --device="torq" --input=1x224x224x3xsi8
```

## Documentation

The full API reference and usage examples are available in the [Torq Runtime user guide](https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/torq_runtime.html).
