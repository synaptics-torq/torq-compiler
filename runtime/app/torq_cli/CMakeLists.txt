# Copyright 2026 Synaptics
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT TARGET torq-runtime)
  add_library(torq-runtime INTERFACE)
  target_link_libraries(torq-runtime INTERFACE iree::runtime)
endif()

if(NOT TARGET torq_engine)
  set(TORQ_COMPILER_SRC_DIR "${PROJECT_SOURCE_DIR}")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../fw/torq_engine"
      "${CMAKE_CURRENT_BINARY_DIR}/torq_engine")
endif()

add_executable(torq_cli main.cpp)
target_compile_features(torq_cli PRIVATE cxx_std_17)
target_link_libraries(torq_cli PRIVATE torq::engine torq::preprocess torq::postprocess)
install(TARGETS torq_cli RUNTIME DESTINATION bin)