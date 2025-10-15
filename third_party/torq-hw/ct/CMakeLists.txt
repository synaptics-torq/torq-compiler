# Copyright 2024 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2`.0 WITH LLVM-exception

set(SRC
  torq_api.h
  torq_api.c
  torq_desc_dump.c
)
add_library(torq_ct ${SRC})
target_include_directories(torq_ct
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../reg
)

#target_compile_options(torq_ct PRIVATE -w)
target_compile_options(torq_ct PRIVATE -Wno-multichar -UNDEBUG)

# Workaround for IREE build system usage
set(TORQ_CT_INC_DIR ${CMAKE_CURRENT_LIST_DIR} PARENT_SCOPE)
set(TORQ_CT_LIB ${CMAKE_CURRENT_BINARY_DIR}/libtorq_ct.a PARENT_SCOPE)
