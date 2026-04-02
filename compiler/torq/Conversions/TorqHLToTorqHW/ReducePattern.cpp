// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl"

static llvm::cl::opt<bool> clTorqFakeReduce(
    "torq-fake-reduce",
    llvm::cl::desc("Replace reduce kernel with a single-iteration no-op kernel "
                   "(useful to isolate DMA transfer time for throughput measurement)"),
    llvm::cl::init(false)
);

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

template <>
LogicalResult ReducePattern::transform(torq_hl::ReduceOp op, PatternRewriter &rewriter) const {
    auto ctx = op.getContext();

    // Get operation details
    std::string opName = op.getName().str();
    int axis = op.getAxis();

    // Determine ALU operation mode based on reduction type
    torq_hw::ALUOp1Mode hwOp1Mode;
    if (opName == "reduce_sum") {
        hwOp1Mode = torq_hw::ALUOp1Mode::ACC;
    }
    else if (opName == "reduce_max") {
        hwOp1Mode = torq_hw::ALUOp1Mode::MAX;
    }
    else if (opName == "reduce_min") {
        hwOp1Mode = torq_hw::ALUOp1Mode::MIN;
    }
    else if (opName == "reduce_and") {
        hwOp1Mode = torq_hw::ALUOp1Mode::BAND;
    }
    else if (opName == "reduce_or") {
        hwOp1Mode = torq_hw::ALUOp1Mode::BOR;
    }
    else if (opName == "reduce_xor") {
        hwOp1Mode = torq_hw::ALUOp1Mode::BXOR;
    }
    else if (opName == "reduce_mul") {
        hwOp1Mode = torq_hw::ALUOp1Mode::MUL;
    }
    else {
        return op.emitError() << "Unsupported reduce operation: " << opName;
    }

    Slice slice(opName);

    // Setup input and output tensors
    LData input(op.getInput());
    LData output(op.getInit());

    auto inputShape = input.shape();
    int rank = inputShape.size();
    int denseDims = std::min(input.denseDims(), output.denseDims());
    denseDims = std::min(denseDims, rank - axis - 1);

    DType inputDType = input.elementType();
    int vectorSize = slice.alu.iWidth(inputDType);

    // Vectorize only the last dimension (retain)
    input.fuse(denseDims).vectorize(vectorSize);

    struct In : Vectorized {
        enum {
            Batch = 0,
        };
    };

    if (clTorqFakeReduce) {
        // Single-iteration kernel: makes compute negligible so DMA transfer
        // time dominates. Used for DMA throughput measurement.
        IData idata = slice.iram.load(input[Indexes(input.shape().size(), 0)]);
        PData pdata = slice.alu.accumulate(idata, hwOp1Mode);
        QData res = slice.act.load(pdata);
        slice.append(output, res);
    }
    else {
        For(auto b = slice.iterate(input.dims(In::Batch, axis))) {
            For(auto i = slice.iterate(input.dims(axis + 1, In::Vectors))) {
                For(auto rv = slice.iterate(input.dim(In::Vectors))) {
                    PData pdata;
                    For(auto r = slice.iterate(input.dim(axis))) {
                        // Accumulate across the reduce dimension
                        IData idata = slice.iram.load(input[b][r][i][rv]);
                        pdata = slice.alu.accumulate(idata, hwOp1Mode);
                    }

                    // Store the accumulated result
                    For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                        QData res = slice.act.load(pdata[av]);
                        slice.append(output[b][i], res);
                    }
                }
            }
        }
    }

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,                            // Operation to replace
        opName,                        // Task name
        ValueRange{op.getInput()},     // Input tensor
        ValueRange{op.getWeights()},   // Weights
        ValueRange{op.getScaleBias()}, // BiasScale tensor
        ValueRange{op.getInit()},      // Output tensor initializer
        ValueRange{},                  // Symbols
        slice.getCfgAttr(ctx),         // Slice configuration
        slice.getNdls()                // NDLs
    );

    LLVM_DEBUG(llvm::dbgs() << "Successfully lowered " << opName << " to ReducePattern\n");
    return success();
}
} // namespace mlir::syna::torq
