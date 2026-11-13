# Copyright 2026 Synaptics
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(TARGET torq_preprocess)
    return()
endif()
add_library(torq_preprocess STATIC src/preprocessor.cpp)
add_library(torq::preprocess ALIAS torq_preprocess)
target_compile_features(torq_preprocess PUBLIC cxx_std_17)
target_include_directories(torq_preprocess PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/inc>
)
target_link_libraries(torq_preprocess PUBLIC torq::engine)
if(TORQ_OS_RTOS)
    target_link_libraries(torq_preprocess PRIVATE zephyr_interface)
endif()