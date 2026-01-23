# Copyright 2024 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::target::torq")

add_subdirectory(Dialect/TorqHL)
add_subdirectory(Dialect/TorqHW)
add_subdirectory(Utils)
add_subdirectory(Transforms/Linalg)
add_subdirectory(Transforms/TorqHL)
add_subdirectory(Conversions/LinalgToTorqHL)
add_subdirectory(Conversions/TorqHLToTorqHW)
add_subdirectory(Conversions/TosaToTorqHL)
add_subdirectory(Pipelines)
add_subdirectory(Codegen)
add_subdirectory(Serialization)
add_subdirectory(Target)
add_subdirectory(Dialect/Tensor)

iree_cc_library(
  NAME
    registration
  SRCS
    "PluginRegistration.cpp"
  DEPS
    iree::compiler::Dialect::HAL::Target
    iree::compiler::PluginAPI
    iree::base::core_headers
    iree::base::internal::flatcc::building
    schemas::torq_executable_def_c_fbs
    iree::target::torq::Target
    iree::target::torq::Serialization::Serialization
    iree::target::torq::Dialect::TorqHL::Dialect
    iree::target::torq::Dialect::TorqHW::Dialect
    iree::target::torq::Transforms::TorqHL::Transforms
    iree::target::torq::Transforms::Linalg::Transforms
    iree::target::torq::Codegen
    iree::target::torq::Pipelines
    iree::target::torq::Dialect::Tensor::IR
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    torq
  TARGET
    ::registration
)

if(IREE_BUILD_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()
