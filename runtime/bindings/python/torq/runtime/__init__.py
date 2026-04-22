# Copyright 2024 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Torq runtime Python bindings.

This module provides Torq-specific runtime functionality that extends the IREE runtime.
"""

__all__ = [
    "InferenceRunner",
    "run_vmfb",
    "profile_vmfb_inference_time",
    "profile_vmfb_resources",
    "VMFBInferenceRunner",
]

from .base import InferenceRunner
from .vmfb import (
    run_vmfb,
    profile_vmfb_inference_time,
    VMFBInferenceRunner,
)
from .profiling import profile_vmfb_resources
