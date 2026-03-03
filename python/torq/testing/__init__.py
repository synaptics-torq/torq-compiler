"""
This pytest plugin provides helper functions and fixtures to facilitate testing
of the torq compiler and runtime.
"""

# this registers the hooks and fixtures defined in the modules listed below

pytest_plugins = [
    "torq.testing.versioned_fixtures",
    "torq.testing.performance",
    "torq.testing.cases",
    "torq.testing.iree",
    "torq.testing.onnx",
    "torq.testing.comparison",
    "torq.testing.reporting"
]

# Add tensorflow plugin only if the right tensorflow version is available
from importlib import metadata
from packaging.version import Version
try:
    if Version(metadata.version("tensorflow")) >= Version("2.18.1"):
        pytest_plugins.append("torq.testing.tensorflow")
    else:
        print("Warning: obsolete tensorflow version")
except metadata.PackageNotFoundError:
    print("Warning: tensorflow not available")

