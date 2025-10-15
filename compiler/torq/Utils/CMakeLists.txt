# Copyright 2024 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2`.0 WITH LLVM-exception

iree_cc_library(
  NAME
    Utils
  SRCS
    MemoryUtils.cpp
    TorqUtils.cpp
    CodeSizeUtils.cpp
    ConversionUtils.cpp
    Kernel.cpp
    ComputeConstants.cpp
    InvocationUtils.cpp
    TorqHw.cpp
    ExecutorAssignment.cpp
    EncodingUtils.cpp
  HDRS
    MemoryUtils.h
    TorqUtils.h
    CodeSizeUtils.h
    ConversionUtils.h
    ComputeConstants.h
    Kernel.h
    InvocationUtils.h
    TorqHw.h
    ExecutorAssignment.h
    EncodingUtils.h
  DEPS
    MLIRIR
    iree::target::torq::Dialect::TorqHL::Dialect
    iree::target::torq::Dialect::TorqHW::Dialect
  PUBLIC
)
