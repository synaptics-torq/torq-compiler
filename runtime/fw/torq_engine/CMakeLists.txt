# Copyright 2026 Synaptics
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(torq_engine LANGUAGES CXX)
endif()

add_library(torq_engine STATIC src/network.cpp src/vmfb_metadata.cpp)
add_library(torq::engine ALIAS torq_engine)
target_include_directories(torq_engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/inc>
)
target_compile_features(torq_engine PUBLIC cxx_std_17)
target_compile_definitions(torq_engine PUBLIC
    TORQ_OS_RTOS=$<BOOL:${TORQ_OS_RTOS}>
    TORQ_OS_LINUX=$<BOOL:${TORQ_OS_LINUX}>
)
if(TORQ_OS_RTOS)
    if(NOT TARGET torq_runtime_lib OR NOT DEFINED TORQ_COMPILER_SRC_DIR)
        message(FATAL_ERROR "Initialize the RTOS TORQ runtime before adding torq_engine.")
    endif()
    target_sources(torq_engine PRIVATE src/allocator_rtos.cpp src/predictor_torq_rtos.cpp)
    zephyr_sources(src/torq_iree_shim_rtos.cpp)
    target_link_libraries(torq_engine PRIVATE zephyr_interface torq_runtime_lib)
    add_dependencies(torq_engine zephyr_generated_headers)
    # target_compile_definitions(torq_engine PRIVATE
    #     IREE_PLATFORM_GENERIC
    #     IREE_SYNCHRONIZATION_DISABLE_UNSAFE=1
    #     IREE_FILE_IO_ENABLE=0
    #     IREE_STATUS_FEATURES=0
    # )
    target_include_directories(torq_engine PRIVATE
        ${TORQ_COMPILER_SRC_DIR}
        ${TORQ_COMPILER_SRC_DIR}/third_party/iree/runtime/src
    )
elseif(TORQ_OS_LINUX)
    set(TORQ_IREE_INCLUDE_DIR "" CACHE PATH "Linux IREE runtime include directory")
    set(TORQ_LINUX_RUNTIME_TARGET "torq-runtime" CACHE STRING "Linux TORQ runtime target or library")
    set(TORQ_LINUX_DEVICE "torq" CACHE STRING "Linux inference device: torq or local-task")
    set_property(CACHE TORQ_LINUX_DEVICE PROPERTY STRINGS torq local-task)
    if(NOT TORQ_LINUX_DEVICE STREQUAL "torq" AND NOT TORQ_LINUX_DEVICE STREQUAL "local-task")
        message(FATAL_ERROR "TORQ_LINUX_DEVICE must be torq or local-task.")
    endif()
    if(NOT TORQ_IREE_INCLUDE_DIR AND DEFINED TORQ_COMPILER_SRC_DIR)
        set(TORQ_IREE_INCLUDE_DIR "${TORQ_COMPILER_SRC_DIR}/third_party/iree/runtime/src")
    endif()
    find_package(Threads REQUIRED)
    target_compile_definitions(torq_engine PRIVATE TORQ_LINUX_DEVICE="${TORQ_LINUX_DEVICE}")
    target_sources(torq_engine PRIVATE
        src/allocator_linux.cpp
        src/predictor_torq_linux.cpp
    )
    target_include_directories(torq_engine PRIVATE
        ${TORQ_IREE_INCLUDE_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/../..
        ${CMAKE_CURRENT_LIST_DIR}/../../torq_hw/inc
    )
    target_link_libraries(torq_engine PRIVATE
        ${TORQ_LINUX_RUNTIME_TARGET} Threads::Threads
    )
    if(NOT TARGET ${TORQ_LINUX_RUNTIME_TARGET})
        set(TORQ_LINUX_ALLOCATOR_CTL "iree_allocator_libc_ctl" CACHE STRING "Linux IREE system allocator")
        target_compile_definitions(torq_engine PRIVATE IREE_ALLOCATOR_SYSTEM_CTL=${TORQ_LINUX_ALLOCATOR_CTL})
    endif()
else()
    message(FATAL_ERROR "TORQ supports Zephyr RTOS and Linux only.")
endif()

if(NOT TARGET torq_preprocess)
    add_subdirectory(../torq_preprocess ${CMAKE_CURRENT_BINARY_DIR}/torq_preprocess)
endif()
if(NOT TARGET torq_postprocess)
    add_subdirectory(../torq_postprocess ${CMAKE_CURRENT_BINARY_DIR}/torq_postprocess)
endif()
