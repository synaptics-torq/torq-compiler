# Copyright 2024 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


include(AddLLVM)
include(AddMLIR)
include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=iree.compiler.")

set(_PYTHON_BUILD_PREFIX "${IREE_BINARY_DIR}/compiler/torq/python")
set(_PYTHON_INSTALL_PREFIX "python_packages/iree_compiler")

include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  "${IREE_SOURCE_DIR}/compiler/src"
  "${IREE_SOURCE_DIR}/compiler/bindings/c"
  "${IREE_SOURCE_DIR}/llvm-external-projects/iree-dialects/include"
  "${IREE_SOURCE_DIR}/third_party/llvm-project/mlir/include"
  "${IREE_SOURCE_DIR}/third_party/stablehlo/include"
  "${IREE_SOURCE_DIR}/third_party/stablehlo"
)


declare_mlir_python_sources(TorqPythonSources)
declare_mlir_python_sources(TorqPythonSources.Dialects
  ADD_TO_PARENT TorqPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TorqPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/TorqHLOps.td
  GEN_ENUM_BINDINGS
  SOURCES dialects/torq_hl.py
  DIALECT_NAME torq_hl
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TorqPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/TorqHWOps.td
  GEN_ENUM_BINDINGS
  SOURCES dialects/torq_hw.py
  DIALECT_NAME torq_hw
)

add_mlir_python_modules(TorqCompilerPythonModules
  ROOT_PREFIX "${_PYTHON_BUILD_PREFIX}/torq"
  INSTALL_PREFIX "${_PYTHON_INSTALL_PREFIX}/iree/compiler"
  DECLARED_SOURCES
    TorqPythonSources
)
