# Copyright 2024 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_TORQ_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

set(IREE_TORQ_ENABLE_TORQ_DRIVER OFF)

if("torq" IN_LIST IREE_EXTERNAL_HAL_DRIVERS)
  message(STATUS "Enabling torq build because it is an enabled HAL driver")
  set(IREE_TORQ_ENABLE_TORQ_DRIVER ON)
endif()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/runtime TORQ)

if(TORQ_ENABLE_AWS_FPGA)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party/aws_fpga_mgmt aws_fpga_mgmt)
endif()
