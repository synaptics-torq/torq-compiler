# Copyright 2024 Synaptics
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# TORQ HAL driver
# Enable with: -DIREE_EXTERNAL_HAL_DRIVERS=torq
#-------------------------------------------------------------------------------

iree_add_all_subdirs()

iree_register_external_hal_driver(
  NAME
    torq
  DRIVER_TARGET
    driver::registration
  REGISTER_FN
    iree_hal_torq_driver_module_register
)

set(torq_hdrs
    native_executable.h
    nop_executable_cache.h
    torq_command_buffer.h
    torq_device.h
    torq_driver.h
    torq_event.h
    torq_semaphore.h
)
set(torq_srcs
    native_executable.cc
    nop_executable_cache.cc
    torq_command_buffer.c
    torq_device.c
    torq_driver.c
    torq_event.c
    torq_semaphore.c
)
set(torq_deps
    iree::base
    iree::base::internal
    iree::base::internal::arena
    iree::base::internal::cpu
    iree::base::internal::synchronization
    iree::base::internal::flatcc::parsing
    iree::hal
    iree::hal::local
    iree::hal::local::executable_environment
    iree::hal::utils::deferred_command_buffer
    iree::hal::utils::file_transfer
    iree::hal::utils::semaphore_base
    iree::tooling::numpy_io
    iree::io::stream
    iree::io::stdio_stream
    schemas::torq_executable_def_c_fbs
    torq_hw::TorqHw
)
set(torq_defs)

iree_cc_library(
  NAME
    torq
  HDRS
    ${torq_hdrs}
  INCLUDES
    ${TORQ_HW_DIR}
    ${TORQ_SIMULATOR_INC_DIR}
    ${AWS_FPGA_INC_DIR}
  SRCS
    ${torq_srcs}
  DEPS
    ${torq_deps}
  DEFINES
    ${torq_defs}
  PUBLIC
)
