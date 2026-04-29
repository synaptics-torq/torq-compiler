"""
This pytest plugin provides helper functions and fixtures to facilitate testing
of the torq compiler and runtime.
"""

# this registers the hooks and fixtures defined in the modules listed below

pytest_plugins = [
    "torq.testing.versioned_fixtures",
    "torq.testing.cases",
    "torq.testing.numpy",
    "torq.testing.onnx",
    "torq.testing.torch",
    "torq.testing.comparison",
    "torq.testing.reporting",
    "torq.testing.issues",
    "torq.testing.xfail",
    "torq.testing.engines",
]

# Add tensorflow plugin only if the right tensorflow version is available
from importlib import metadata
import os
from packaging.version import Version
try:
    if Version(metadata.version("tensorflow")) >= Version("2.18.1"):
        pytest_plugins.append("torq.testing.tensorflow")
        pytest_plugins.append("torq.testing.tflite_layer_tests")
    else:
        print("Warning: obsolete tensorflow version")
except metadata.PackageNotFoundError:
    print("Warning: tensorflow not available")


# check if the iree package is available, if so add plugins that depend on it
try:
    import iree
    pytest_plugins.append("torq.testing.performance")
    pytest_plugins.append("torq.testing.iree")
    pytest_plugins.append("torq.executor_discovery.pytest_plugin")

except ImportError:
    print("Warning: iree not available, skipping iree test support")