# Copyright 2024 Synaptics
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# TORQ HAL driver
# Enable with: -DIREE_EXTERNAL_HAL_DRIVERS=torq
#-------------------------------------------------------------------------------

option(TORQ_HW_DEBUG "Enable verbose debugging logs of HW" OFF)

option(TORQ_ENABLE_SOC_FPGA "Enable SoC FPGA support" OFF)
option(TORQ_ENABLE_ASTRA_MACHINA "Enable Astra Machina platform support" OFF)
option(TORQ_ENABLE_AWS_FPGA "Enable AWS FPGA support" ON)
option(TORQ_ENABLE_SIMULATOR "Enable Simulator support" ON)
option(TORQ_ENABLE_KELVIN_SIMULATOR "Enable Kelvin CSS Simulator support" OFF)

set(torq_deps)
set(torq_srcs src/TorqUtils.cpp src/TorqHw.cpp src/TorqEventLog.cpp)
set(torq_defs)
set(include_dirs
    ${TORQ_HW_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/inc
    ${TORQ_HW_DIR}/reg
    ${TORQ_HW_DIR}/ct
    ${TORQ_HW_DIR}/css_sw/common
)

if (TORQ_ENABLE_SIMULATOR)
  list(APPEND torq_defs "ENABLE_SIMULATOR=1")
  list(APPEND torq_srcs
      src/TorqSimulator.h
      src/TorqSimulator.cpp
      src/TorqHw.cpp
      src/QemuCSSSimulation.cpp
  )
  list(APPEND torq_deps iree::hal::local::executable_library)

  # Note: that doesn´t work for some obscure EXPORT problem
  # target_link_libraries(torq_hw_TorqHw PRIVATE torq_cm)
  # so we need to add the include directories and the library manually instead
  list(APPEND include_dirs ${TORQ_SIMULATOR_INC_DIR})
  list(APPEND torq_deps ${TORQ_SIMULATOR_LIB})

  set(TORQ_KELVIN_SIMULATOR_LIB
    ${IREE_TORQ_SOURCE_DIR}/third_party/kelvin-simulator/${CMAKE_HOST_SYSTEM_PROCESSOR}-${CMAKE_HOST_SYSTEM_NAME}/libkelvin_simulator_mpact.so)
  if(TORQ_ENABLE_KELVIN_SIMULATOR AND EXISTS ${TORQ_KELVIN_SIMULATOR_LIB})
    list(APPEND torq_srcs src/KelvinCSSSimulation.cpp)
    list(APPEND torq_deps ${TORQ_KELVIN_SIMULATOR_LIB})
    list(APPEND torq_defs "TORQ_KELVIN_SIMULATOR=1")
  endif()
endif()

if(TORQ_HW_DEBUG)
  list(APPEND torq_defs "TORQ_HW_DEBUG=1")
endif()

if(TORQ_ENABLE_AWS_FPGA)
    list(APPEND torq_srcs src/TorqAwsFpga.cpp src/TorqAwsFpga.h)
    list(APPEND torq_deps ${IREE_TORQ_SOURCE_DIR}/third_party/fpga_mgmt/${CMAKE_HOST_SYSTEM_PROCESSOR}-${CMAKE_HOST_SYSTEM_NAME}/libfpga_mgmt.so.1.0.0)
    list(APPEND torq_defs "ENABLE_AWS_FPGA=1")
    set(AWS_FPGA_INC_DIR ${IREE_TORQ_SOURCE_DIR}/third_party/fpga_mgmt/include)
endif()

if(TORQ_ENABLE_SOC_FPGA)
    list(APPEND torq_srcs src/TorqSoCFpga.cpp src/TorqSoCFpga.h)
    list(APPEND torq_defs "ENABLE_SOC_FPGA=1")
endif()

if(TORQ_ENABLE_ASTRA_MACHINA)
    list(APPEND torq_srcs src/TorqAstraMachina.cpp src/TorqAstraMachina.h)
    list(APPEND torq_defs "ENABLE_ASTRA_MACHINA=1")
    add_subdirectory(hal/SL261x)
endif()

if(TORQ_ENABLE_AWS_FPGA)
    list(APPEND include_dirs ${AWS_FPGA_INC_DIR})
endif()

iree_cc_library(
  NAME
    TorqHw
  HDRS
    inc/TorqHw.h
    inc/TorqUtils.h
  INCLUDES
    ${include_dirs}
  SRCS
    ${torq_srcs}
  DEPS
    ${torq_deps}
  DEFINES
    ${torq_defs}
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/inc
)
