# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Torq Compiler Python Bindings
#
# This target symlinks Torq compiler Python sources into the IREE compiler
# build tree so they share a PYTHONPATH and can be packaged together.

set(_TORQ_COMPILER_PYTHON_OUTPUT_DIR "${IREE_BINARY_DIR}/compiler/bindings/python")

set(_TORQ_COMPILER_PYTHON_SRCS
  "torq/compiler/__init__.py"
  "torq/compiler/tools/__init__.py"
  "torq/compiler/tools/binaries.py"
  "torq/compiler/tools/core.py"
  "torq/_compiler_libs/__init__.py"
)

# Create a target that symlinks Torq compiler Python files to the build directory
add_custom_target(torq_compiler_python_bindings ALL
  COMMENT "Setting up Torq compiler Python bindings"
)

# Symlink each source file to the output directory
foreach(_SRC_FILE ${_TORQ_COMPILER_PYTHON_SRCS})
  set(_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${_SRC_FILE}")
  set(_DST_PATH "${_TORQ_COMPILER_PYTHON_OUTPUT_DIR}/${_SRC_FILE}")
  get_filename_component(_DST_DIR "${_DST_PATH}" DIRECTORY)

  add_custom_command(
    TARGET torq_compiler_python_bindings
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_DST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${_SRC_PATH}" "${_DST_PATH}"
    BYPRODUCTS "${_DST_PATH}"
    COMMENT "Symlinking ${_SRC_FILE}"
  )
endforeach()

set(_INSTALL_DIR "python_packages/iree_compiler")
set(_INSTALL_COMPONENT "TorqPythonPackage-compiler")
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/torq/"
  COMPONENT "${_INSTALL_COMPONENT}"
  DESTINATION "${_INSTALL_DIR}/torq/"
  FILES_MATCHING PATTERN "*.py"
)

# Build torq-compile for the Python package with $ORIGIN RPATH so it finds
# libIREECompiler.so in the same _mlir_libs/ directory at runtime.
# NOTE: We avoid OUTPUT_NAME "torq-compile" to prevent shadowing the standalone
# torq-compile target's ninja phony alias. A post-build copy produces the
# correctly-named binary in the build tree, and install(TARGETS) + rename
# handles the installed copy with proper RPATH fixup.
set(_MLIR_LIBS_OUTPUT_DIR "${_TORQ_COMPILER_PYTHON_OUTPUT_DIR}/iree/compiler/_mlir_libs")
add_executable(torq_compile_python_tool
  "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/torq-compile-main.cc"
)
target_link_libraries(torq_compile_python_tool
  iree_compiler_bindings_c_headers
  iree_compiler_API_Impl
)
set_target_properties(torq_compile_python_tool PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${_MLIR_LIBS_OUTPUT_DIR}"
)
mlir_python_setup_extension_rpath(torq_compile_python_tool)
add_custom_command(TARGET torq_compile_python_tool POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    "$<TARGET_FILE:torq_compile_python_tool>"
    "${_MLIR_LIBS_OUTPUT_DIR}/torq-compile"
  COMMENT "Copying torq_compile_python_tool -> torq-compile"
)
install(TARGETS torq_compile_python_tool
  COMPONENT "${_INSTALL_COMPONENT}"
  DESTINATION "${_INSTALL_DIR}/iree/compiler/_mlir_libs"
)
install(CODE "
  file(RENAME
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${_INSTALL_DIR}/iree/compiler/_mlir_libs/torq_compile_python_tool\"
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${_INSTALL_DIR}/iree/compiler/_mlir_libs/torq-compile\"
  )"
  COMPONENT "${_INSTALL_COMPONENT}"
)
add_dependencies(torq_compiler_python_bindings torq_compile_python_tool)
