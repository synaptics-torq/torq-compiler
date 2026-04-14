// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"

#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"
#include <numeric>

#define DEBUG_TYPE "torq-lower-torqhl"

namespace mlir::syna::torq {

// This kernel supports matmul, dot product and matrix-vector multiplication
template <>
LogicalResult MatMulPattern::transform(torq_hl::MatMulOp op, PatternRewriter &rewriter) const {
    struct MatA { // Loaded as weights
        enum { Batch, M, K };
    };
    struct MatB : Vectorized { // Loaded as input data
        enum { Batch, K, N };
    };
    struct MatC {
        enum { Batch, M, N };
    };

    LData matA(op.getInput1());
    LData matB(op.getInput2());
    LData output(op.getInit());
    LData biasScale(op.getScaleBias());

    auto rankA = matA.shape().size();
    auto rankB = matB.shape().size();
    if (rankA < 1 || rankA >= 4 || rankB < 1 || rankB >= 4) {
        op.emitError() << "Input rank must be 1, 2, 3, got " << rankA << ", " << rankB << "\n";
        return failure();
    }

    if (rankB == 1) {
        // Convert dot product and mat-vect-product to mat-product by making B Kx1 and C Mx1.
        // TODO in some cases dot-prod could be implemented more efficiently with a dedicated kernel
        matB.insertDim(rankB, {1});
        output.insertDim(output.shape().size(), {1});
    }

    // Make input and output matrices 3D
    for (auto *mat : SmallVector<LData *>{&matA, &matB, &output}) {
        while (mat->shape().size() < 3) {
            mat->insertDim(0, {1});
        }
    }
    assert(matA.dim(MatA::K) == matB.dim(MatB::K));

    // Broadcast batch dimension in matA or matB if needed
    if (rankA == 3 && rankB == 2) {
        matB.getShape()[MatB::Batch].stride = 0;
    }
    else if (rankA == 2 && rankB == 3) {
        matA.getShape()[MatA::Batch].stride = 0;
    }

    Slice slice("matmul");
    matB.vectorize(slice.alu.iWidth(matB.elementType(), matA.elementType()));

    BData bdata = slice.bram.load(biasScale);
    For(auto batch = slice.iterate(matA.dim(MatA::Batch))) {
        For(auto im = slice.iterate(matA.dim(MatA::M))) {     // rows in matA
            For(auto in = slice.iterate(matB.dim(MatB::N))) { // col vectors in matB (N/vectSize)
                PData pdata;
                For(auto ik = slice.iterate(matA.dim(MatA::K))) { // cols in matA == rows in matB
                    WData a = slice.wram.load(matA[batch][im][ik]);
                    IData b = slice.iram.load(matB[batch][ik][in]);
                    pdata = slice.alu.scalarProductAccumulate(b, a);
                }

                For(auto a = slice.iterate(pdata.dim(PData::Vectors))) {
                    QData res = slice.act.rescaleClamp(
                        pdata[a], bdata, op.getShift(), op.getOutputZp(), op.getOutputMin(),
                        op.getOutputMax()
                    );
                    slice.append(output[batch][im], res);
                }
            }
        }
    }

    auto newOp = torq_hw::SliceTaskOp::create(
        rewriter, op->getLoc(), slice.name(),
        op.getInput2(),                          // Input tensor
        op.getInput1(),                          // Weights
        op.getScaleBias(),                       // BiasScale tensor
        op.getInit(),                            // Output tensor initializer
        slice.getCfgAttr(rewriter.getContext()), // Slice configuration
        slice.getNdls()                          // NDLs
    );
    rewriter.replaceOp(op, newOp.getOperation());

    return success();
}

} // namespace mlir::syna::torq
