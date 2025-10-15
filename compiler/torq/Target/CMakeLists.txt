# Copyright 2024 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_library(
  NAME
    Target
  SRCS
    "TorqTarget.cpp"
  DEPS
    LLVMSupport
    MLIRAsmParser
    MLIRIR
    MLIRSupport
    iree::compiler::Dialect::HAL::Target
    iree::target::torq::Codegen
    iree::base::internal::flatcc::building
    iree::base::internal::flatcc::parsing
    iree::target::torq::Dialect::TorqHL::Dialect
    iree::target::torq::Dialect::TorqHW::Dialect
    iree::target::torq::Serialization::Serialization
  PUBLIC
)
