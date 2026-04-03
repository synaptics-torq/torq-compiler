# Copyright 2024 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Torq Runtime Python Bindings

# Output directory is the same as IREE's runtime bindings to share PYTHONPATH
set(_TORQ_RUNTIME_PYTHON_OUTPUT_DIR "${IREE_BINARY_DIR}/runtime/bindings/python")

set(_TORQ_RUNTIME_PYTHON_SRCS
  "torq/runtime/__init__.py"
  "torq/runtime/base.py"
  "torq/runtime/utils.py"
  "torq/runtime/vmfb.py"
  "torq/runtime/scripts/__init__.py"
  "torq/runtime/scripts/torq_run_module/__init__.py"
  "torq/runtime/scripts/torq_run_module/__main__.py"
)

# Create a target that symlinks Torq Python files to the build directory
add_custom_target(torq_runtime_python_bindings ALL
  COMMENT "Setting up Torq runtime Python bindings"
)
add_dependencies(torq_runtime_python_bindings iree_runtime_bindings_python_runtime)

# Symlink each source file to the output directory
foreach(_SRC_FILE ${_TORQ_RUNTIME_PYTHON_SRCS})
  set(_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${_SRC_FILE}")
  set(_DST_PATH "${_TORQ_RUNTIME_PYTHON_OUTPUT_DIR}/${_SRC_FILE}")
  get_filename_component(_DST_DIR "${_DST_PATH}" DIRECTORY)

  add_custom_command(
    TARGET torq_runtime_python_bindings
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_DST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${_SRC_PATH}" "${_DST_PATH}"
    BYPRODUCTS "${_DST_PATH}"
    COMMENT "Symlinking ${_SRC_FILE}"
  )
endforeach()

set(_INSTALL_DIR "python_packages/iree_runtime")
set(_INSTALL_COMPONENT "TorqPythonPackage-runtime")
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/torq/"
  COMPONENT "${_INSTALL_COMPONENT}"
  DESTINATION "${_INSTALL_DIR}/torq/"
  FILES_MATCHING PATTERN "*.py"
)
