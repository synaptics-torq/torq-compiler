# Torch Model Test Fixtures

This folder contains PyTorch models that are exercised by `tests/test_torch_model.py`.

`test_torch_model.py` will automatically find every `.pt`/`.pth` and `.py` file here and create two test cases for each:

- a **full-model** test (`<stem>_full_model`)
- a **per-layer** test for every named submodule (`<stem>_<layer>_<LayerClass>`)

## Quick start: add a new model

The easiest way is to save a **bundle dict**:

```python
import torch

model = MyModel().eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

torch.save(
    {"model": model, "example_inputs": example_inputs},
    "tests/testdata/torch_models/my_model.pt",
)
```

That’s it. Run the tests with:

```bash
pytest tests/test_torch_model.py -v -k my_model
```

## File formats

### 1. Bundle dict `.pt` / `.pth` (recommended)

Save a Python dict with two keys:

| Key | Value |
|---|---|
| `model` | A `torch.nn.Module` (usually in `eval()` mode) |
| `example_inputs` | A tuple or list of `torch.Tensor` |

Example:

```python
torch.save({
    "model": model,
    "example_inputs": (torch.randn(1, 3, 224, 224),),
}, "my_model.pt")
```

This is the preferred format because everything lives in a single file.

### 2. Python module `.py` (alternative)

Define a `torch.nn.Module` subclass and optionally expose `get_example_inputs()`:

```python
# my_model.py
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

def get_example_inputs():
    return (torch.randn(1, 10),)
```

The test will instantiate one `torch.nn.Module` subclass from the file and call `get_example_inputs()`. If the file is named `my_model.py`, the loader prefers a class named `MyModel`; otherwise it falls back to the first `nn.Module` subclass it finds. So naming the class to match the file stem is the safest convention.

## I have a downloaded model — how do I use it?

### If the download is a full model object

Wrap it into the bundle format:

```python
import sys
import torch
from pathlib import Path

# Add the directory that contains the model source if it is not on PYTHONPATH
sys.path.insert(0, str(Path("/path/to/model/code")))

from downloaded_model import MyModel

model = MyModel().eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

out_path = Path("tests/testdata/torch_models/my_downloaded_model.pt")
torch.save({"model": model, "example_inputs": example_inputs}, out_path)
print(f"Saved fixture to {out_path}")
```

### If the download is only a state dict

Instantiate the model class, load the weights, then save the bundle:

```python
import torch
from my_model import MyModel

model = MyModel()
state_dict = torch.load("downloaded_weights.pt", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

example_inputs = (torch.randn(1, 3, 224, 224),)
torch.save(
    {"model": model, "example_inputs": example_inputs},
    "tests/testdata/torch_models/my_downloaded_model.pt",
)
```

## Tips

- Always call `model.eval()` before saving. Dropout and BatchNorm behave differently in training mode and can cause wrong reference results.
- Pick a clear file name that matches the model (for example, `resnet18.pt`).
- If a `.py` file with the same stem exists next to the `.pt` file, it is imported first so pickle can resolve custom model classes.
- Deeply nested models generate one test case per named submodule, so expect many per-layer tests for large architectures.

## Running tests

```bash
# List all torch model test cases
pytest tests/test_torch_model.py --collect-only

# Run the full-model test for one model
pytest tests/test_torch_model.py -v -s my_model_full_model

# Run a single layer
pytest tests/test_torch_model.py -v -s my_model_fc_Linear
```
