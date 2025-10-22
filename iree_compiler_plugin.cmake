# Copyright 2024 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/compiler target/TORQ)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tools torq-tools)