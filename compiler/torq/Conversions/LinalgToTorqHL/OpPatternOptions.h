// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/Support/CommandLine.h"

namespace mlir::syna::torq {

inline llvm::cl::opt<bool> clACTBasedAdd(
    "torq-act-based-add",
    llvm::cl::desc(
        "use ACT based torq_hl::ElementWiseBinaryOp::ADD instead of ALU based torq_hl::AddOp"
    ),
    llvm::cl::init(false)
);

inline llvm::cl::opt<bool> clACTBasedSub(
    "torq-act-based-sub",
    llvm::cl::desc(
        "use ACT based torq_hl::ElementWiseBinaryOp::SUB instead of ALU based torq_hl::AddOp"
    ),
    llvm::cl::init(false)
);

inline llvm::cl::opt<bool> clMulCasti32Toi16(
    "torq-mul-cast-i32-to-i16",
    llvm::cl::desc("Automatically cast input from i32 to i16 for MUL operation"),
    llvm::cl::init(false)
);

inline llvm::cl::opt<bool> clTableAsGather(
    "torq-convert-table-to-gather", llvm::cl::desc("use GatherOp instead of TosaOp for TOSA Table"),
    llvm::cl::init(false)
);

inline llvm::cl::opt<bool> clConv1dAsMatmul(
    "torq-convert-conv1d-to-matmul", llvm::cl::desc("Convert conv1d to imToCol + matmul"),
    llvm::cl::init(false)
);

} // namespace mlir::syna::torq