# Copyright 2026 Synaptics
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(TARGET torq_postprocess)
    return()
endif()
add_library(torq_postprocess STATIC src/classifier.cpp src/detector.cpp)
add_library(torq::postprocess ALIAS torq_postprocess)
target_compile_features(torq_postprocess PUBLIC cxx_std_17)
target_include_directories(torq_postprocess PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/inc>
)
target_link_libraries(torq_postprocess PUBLIC torq::engine)
if(TORQ_OS_RTOS)
    target_link_libraries(torq_postprocess PRIVATE zephyr_interface)
endif()