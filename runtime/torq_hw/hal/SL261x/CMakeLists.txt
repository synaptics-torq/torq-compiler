# Copyright 2025 Synaptics
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# TORQ Kernel driver for SoC
#-------------------------------------------------------------------------------

project(torq_soc_kernel_module)

set(MODULE_NAME "syna_npu.ko")
set(KERNEL_ARCH "arm64")
set(KERNEL_CROSS_COMPILE "aarch64-linux-gnu-")
set(KERNEL_EXTRACT_PATH "${CMAKE_CURRENT_BINARY_DIR}/syna_kernel/")
set(SYNA_KERNEL_PREBUILTS "${IREE_TORQ_SOURCE_DIR}/third_party/runtimes-prebuilt/soc-kernel/syna-kernel-artifacts.tgz")

add_custom_command(
    OUTPUT ${MODULE_NAME}
    BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_NAME}
    COMMAND ${CMAKE_COMMAND} -E env bash ${IREE_TORQ_SOURCE_DIR}/scripts/setup_soc_ko_build_deps.sh
    COMMAND ${CMAKE_COMMAND} -E make_directory ${KERNEL_EXTRACT_PATH}
    COMMAND tar -xzf ${SYNA_KERNEL_PREBUILTS} -C ${KERNEL_EXTRACT_PATH}/
    COMMAND KBUILD_MODPOST_WARN=1 make -C ${KERNEL_EXTRACT_PATH}
                                  ARCH=${KERNEL_ARCH} CROSS_COMPILE=${KERNEL_CROSS_COMPILE}
                                  M=${CMAKE_CURRENT_SOURCE_DIR} modules
    COMMENT "Building ${MODULE_NAME} linking to prebuilts ${SYNA_KERNEL_PREBUILTS}"
)

add_custom_target(torq_kernel_module
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_NAME} ${CMAKE_CURRENT_BINARY_DIR}/
    COMMAND make -C "${KERNEL_EXTRACT_PATH}" ARCH=${KERNEL_ARCH} CROSS_COMPILE=${KERNEL_CROSS_COMPILE} M="${CMAKE_CURRENT_SOURCE_DIR}" clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${KERNEL_EXTRACT_PATH}
    DEPENDS ${MODULE_NAME}
    COMMENT "copying  ${MODULE_NAME} to ${CMAKE_CURRENT_BINARY_DIR}"
)

add_custom_target(
    torq_kernel_module_clean
    COMMAND make -C "${KERNEL_EXTRACT_PATH}" ARCH=${KERNEL_ARCH} CROSS_COMPILE=${KERNEL_CROSS_COMPILE} M="${CMAKE_CURRENT_SOURCE_DIR}" clean
)
