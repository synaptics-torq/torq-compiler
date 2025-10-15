# Copyright 2024 Synaptics
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_setup_c_src_root(
  DEFAULT_EXPORT_SET Runtime
  DEFAULT_INSTALL_COMPONENT IREEDevLibraries-Runtime
  IMPLICIT_DEFS_TARGET iree_TORQ_runtime_defs
)

add_subdirectory(torq_hw)

# Flatbuffer schema generation does not require XRT. Moreover the generated
# flatbuffer header files are used by the compiler to create artefacts
# (.vmfb file), and so the schema sub-directory is required even when not
# building the XRT driver code.
add_subdirectory(schemas)

if(IREE_TORQ_ENABLE_TORQ_DRIVER)
    add_subdirectory(driver)
endif()
