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

static torq_hw::SliceTaskOp lowerToMatmul(
    torq_hl::MatMulOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset, int chCount
) {
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
        return nullptr;
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

    // Broadcast batch dimension in matA or matB to match output.
    matA.broadcastAs(output, 1);
    matB.broadcastAs(output, 1);

    Slice slice("matmul");

    output.subviewDim(MatC::M, chOffset, chCount);
    matA.subviewDim(MatA::M, chOffset, chCount);

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

    LLVM_DEBUG(llvm::dbgs() << "Lowered to matmul kernel: " << "\n");
    return torq_hw::SliceTaskOp::create(
        rewriter, op->getLoc(), slice.name(),
        op.getInput2(),                          // Input tensor
        op.getInput1(),                          // Weights
        op.getScaleBias(),                       // BiasScale tensor
        op.getInit(),                            // Output tensor initializer
        slice.getCfgAttr(rewriter.getContext()), // Slice configuration
        slice.getNdls()                          // NDLs
    );
}

static torq_hw::SliceTaskOp lowerToFastMatmul(
    torq_hl::MatMulOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset, int chCount
) {
    struct MatA { // Loaded as weights
        enum { Batch, M, K };
        enum { RowGroups = 1, ColGroups, RowBlock, ColBlock }; // After reshaping
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
        return {};
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

    // Broadcast batch dimension in matA or matB to match output.
    Slice slice("matmul-fast");
    matA.broadcastAs(output, 1);
    matB.broadcastAs(output, 1);
    // Get subview of output, weight and biasScale tensors for the given output channel offset and
    // count
    output.subviewDim(MatC::M, chOffset, chCount);
    matA.subviewDim(MatA::M, chOffset, chCount);

    int aluWidth = slice.alu.iWidth(matB.elementType(), matA.elementType());
    int rowChunks = aluWidth / slice.wram.transposeWidth();
    int outRowVectSize = slice.wram.transposeHeight();

    if (rowChunks == matA.dim(MatA::K)) {
        // Corner case: this would make us loading a dense tensor to WRAM instead of using S:4
        // and the data will not be transposed. For now just load smaller chunks so we are sure
        // they will be loaded in multiple steps.
        // The case where matA.dim(MatA::K) == 1 works because the transpose of [4,1] is == [4]
        rowChunks -= 1;
    }

    // Find the largest divisor of K that is less than or equal to rowChunks
    if (matA.dim(MatA::K) < rowChunks)
        rowChunks = 1;
    else {
        while (matA.dim(MatA::K) % rowChunks != 0 && rowChunks > 1) {
            rowChunks--;
        }
    }

    output.reshapeDim(MatC::M, {-1, outRowVectSize});

    matA.reshapeDim(MatA::K, {-1, rowChunks});      // Split each row in rowChunks
    matA.reshapeDim(MatA::M, {-1, outRowVectSize}); // Split rows in groups of outRowVectSize
    // Shape: [Batch, M_outer, M_inner, K_outer, K_inner]

    matB.vectorize(aluWidth);
    matB.reshapeDim(MatB::K, {-1, rowChunks}); // Split rows in rowChunks

    BData bdata = slice.bram.load(biasScale);
    For(auto batch = slice.iterate(matA.dim(MatA::Batch))) {
        For(auto im = slice.iterate(output.dim(MatA::RowGroups))) { // row groups in matA
            For(auto in =
                    slice.iterate(matB.dim(MatB::Vectors))) { // col vectors in matB (N/vectSize)
                PData pdata;
                For(auto ik = slice.iterate(matA.dim(MatA::RowBlock))
                ) { // col groups in matA == row groups in matB

                    WData wdata = slice.wram.load(matA[batch][im][":"][ik]);
                    // Reshape wdata (we should actually transpose!)
                    wdata.setShape({wdata.dim(1), wdata.dim(0)});
                    For(auto ikk = slice.iterate(wdata.dim(0))
                    ) { // cols in group in matA == rows in group in matB
                        IData idata = slice.iram.load(matB[batch][ik][ikk][in]);
                        pdata = slice.alu.outerProductAccumulate(idata, wdata[ikk]);
                    }
                }
                For(auto o = slice.iterate(outRowVectSize)) { // Not necessarily all the pdata
                    For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                        QData res = slice.act.rescaleClamp(
                            pdata[o][av], bdata, op.getShift(), op.getOutputZp(), op.getOutputMin(),
                            op.getOutputMax()
                        );
                        slice.append(output[batch][im][o], res);
                    }
                }
            }
        }
    }

    LLVM_DEBUG(llvm::dbgs() << "Lowered to fast matmul kernel: " << "\n");
    return torq_hw::SliceTaskOp::create(
        rewriter, op->getLoc(), slice.name(),
        op.getInput2(),                          // Input tensor
        op.getInput1(),                          // Weights
        op.getScaleBias(),                       // BiasScale tensor
        taskInitTensor,                          // Output tensor initializer
        slice.getCfgAttr(rewriter.getContext()), // Slice configuration
        slice.getNdls()                          // NDLs
    );
}

// This kernel supports matmul, dot product and matrix-vector multiplication
template <>
LogicalResult MatMulPattern::transform(torq_hl::MatMulOp op, PatternRewriter &rewriter) const {
    Value initValue = op.getInit();
    auto wDims = LData(op.getInput1()).dims();
    int outRowVectSize = 4;
    int outRowCount = wDims.size() == 1 ? 1 : wDims.size() == 2 ? wDims[0] : wDims[1];
    torq_hw::SliceTaskOp hwOp;

    // Force to normal matmul

    if (int peeledOutRow = outRowCount % outRowVectSize) {
        // Compute 64x1 at a time
        if (!(hwOp =
                  lowerToMatmul(op, rewriter, initValue, outRowCount - peeledOutRow, peeledOutRow)
            )) {
            return failure();
        }
        initValue = hwOp.getQ()[0];
        outRowCount -= peeledOutRow;
    }
    if (outRowCount > 0) {
        // Compute 64x4 at a time
        if (!(hwOp = lowerToFastMatmul(op, rewriter, initValue, 0, outRowCount))) {
            return failure();
        }
    }
    rewriter.replaceOp(op, hwOp.getOperation()->getResults());
    return success();
}

} // namespace mlir::syna::torq
