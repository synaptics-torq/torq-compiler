// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/IR/PatternMatch.h"

namespace mlir::syna::torq {

// Patterns giving the torq_hl conv2d, depthwise_conv2d and max_pool2d ops the weight layout and the
// input segmentation their kernels read. They run on the final padding of the ops
// (PrepareConvPoolKernelsPass).
void populateConvPoolKernelLayoutPatterns(RewritePatternSet &patterns);

} // namespace mlir::syna::torq
