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
    "torq.testing.tensorflow",        
    "torq.testing.comparison"
]
