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
    PassProgressUtils.cpp
    TorqUtils.cpp
    CodeSizeUtils.cpp
    ConversionUtils.cpp
    Kernel.cpp
    ComputeConstants.cpp
    InvocationUtils.cpp
    LayoutTransformUtils.cpp
    TorqHw.cpp
    EncodingUtils.cpp
    ShapeUtils.cpp
    ExecutorAssignment.cpp
    StorageUtils.cpp
  HDRS
    MemoryUtils.h
    PassProgressUtils.h
    TorqUtils.h
    CodeSizeUtils.h
    ConversionUtils.h
    ComputeConstants.h
    Kernel.h
    InvocationUtils.h
    LayoutTransformUtils.h
    TorqHw.h
    ExecutorAssignment.h
    EncodingUtils.h
    ShapeUtils.h
    TorqMatcherBase.h
  DEPS
    MLIRIR
    IREEDialectsTransforms
    iree::target::torq::Conversions::TorqHLToLinalg::TorqHLToLinalg
    iree::target::torq::Dialect::TorqHL::Dialect
    iree::target::torq::Dialect::TorqHW::Dialect
  PUBLIC
)
