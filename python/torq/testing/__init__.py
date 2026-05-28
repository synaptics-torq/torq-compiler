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

# IMPORTANT: IREE must be imported BEFORE tensorflow to avoid LLVM symbol conflicts.
# TensorFlow loads its bundled LLVM with RTLD_GLOBAL, which makes its LLVM command-line
# options globally visible. If IREE's libIREECompiler.so (which also bundles LLVM) is
# loaded afterwards, duplicate option registration causes a fatal abort.
# By loading IREE first (with default RTLD_LOCAL), its LLVM symbols stay process-local
# and don't conflict with TensorFlow's subsequent RTLD_GLOBAL load.
from importlib import metadata
import os
from packaging.version import Version

# check if the iree package is available, if so add plugins that depend on it
try:
    import iree
    # Eagerly load the IREE compiler C extension (which links libIREECompiler.so containing
    # LLVM). This MUST happen before importing tensorflow, which loads its own LLVM with
    # RTLD_GLOBAL. Loading IREE first with the default RTLD_LOCAL keeps its LLVM symbols
    # process-local and avoids duplicate LLVM cl::Option registration aborts.
    from iree.compiler.ir import Context as _IREEContext  # noqa: F401
    pytest_plugins.append("torq.testing.iree")
    pytest_plugins.append("torq.executor_discovery.pytest_plugin")

except ImportError:
    print("Warning: iree not available, skipping iree test support")

# Add tensorflow plugin only if the right tensorflow version is available
try:
    import tensorflow
    pytest_plugins.append("torq.testing.tensorflow")
    pytest_plugins.append("torq.testing.tflite_layer_tests")
except ImportError:
    print("Warning: tensorflow not available")


try:
    import jax
    pytest_plugins.append("torq.testing.jax")    
except ImportError:
    print("Warning: jax not available, skipping jax test support")
