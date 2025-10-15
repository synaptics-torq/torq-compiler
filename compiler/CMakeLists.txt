# Copyright 2024 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


iree_setup_c_src_root(
  PACKAGE_ROOT_PREFIX iree::target
  IMPLICIT_DEFS_TARGET iree_target_torq_compiler_defs
)

add_subdirectory(torq)
