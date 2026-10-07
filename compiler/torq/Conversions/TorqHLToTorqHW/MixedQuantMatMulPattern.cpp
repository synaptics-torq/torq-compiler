// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for the language governing use and
// distribution.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Y[1,N] = x[1,K] . dequant(wq[K,N]): int weights stream LRAM and are cast to
// bf16 on the WRAM load (never materialized). zp mode: wscale is the combined
// [2nB+1, N] scale|nzs|ones table; scratch is partf | xsumrows | stage8.

#include "Patterns.h"

#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "torq/Dialect/TorqHW/TorqHWInfo.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

namespace {

// View dims: Blocked = x/weights per block, Rows = scale table / partials.
struct Blocked {
    enum { Block, Row, Vectors, Lanes };
};
struct Rows {
    enum { Row, Vectors, Lanes };
};

// x[k] x 8 weight columns; PRAM pair layout the ACT drain requires.
void buildMacPartialsTask(
    PatternRewriter &rewriter, Location loc, Slice &mac, Value x, Value wq, int wqRank, LData partf,
    int nB, int B, int wW, DType et, int32_t cMin, int32_t cMax
) {
    Slice &slice = mac;
    LData xD(x); // [K, wW]: x replicated across lanes by the raise
    xD.reshapeDim(0, {nB, B});
    LData wqD(wq); // [K, N] or already [nB, B, N]
    if (wqRank == 2)
        wqD.reshapeDim(0, {nB, B});
    wqD.vectorize(wW); // [nB, B, vectors, wW]
    For(auto g = slice.iterate(wqD.dim(Blocked::Block))) {
        For(auto jt = slice.iterate(wqD.dim(Blocked::Vectors))) {
            PData pdata;
            For(auto kk = slice.iterate(wqD.dim(Blocked::Row))) {
                IData xk = slice.iram.load(xD[g][kk]);         // 8 lanes
                WData w = slice.wram.load(wqD[g][kk][jt], et); // cast on load
                pdata = slice.alu.elementwiseProductAccumulate(xk, w);
            }
            For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                QData res = slice.act.clamp(pdata[av], cMin, cMax);
                slice.append(partf[g], res);
            }
        }
    }
    slice.createSliceTaskOp(rewriter, loc);
}

// The zp correction, in two tasks: xsum[g] = sum_k x[g*B+k] (product against
// 8 ones) drained 8-wide into stage8, then broadcast into the N-wide xsumrows.
void buildXsumCorrectionTasks(
    PatternRewriter &rewriter, Location loc, Value x, Value wscale, Value scratch, int nB, int B,
    int wW, int Np, int32_t cMin, int32_t cMax
) {
    LData partfxr(scratch);
    partfxr.setShape({2, nB, Np});
    LData stage8(scratch);
    stage8.setShape({nB, wW});
    stage8.setOffset(2 * nB * Np);
    // The wscale ones row sits at the same offset as stage8.
    LData onesV(wscale);
    onesV.setShape({wW});
    onesV.setOffset(2 * nB * Np);

    Slice xsumSlice("mqmm_xsum");
    LData xD(x);
    xD.reshapeDim(0, {nB, B});
    WData ones = xsumSlice.wram.load(onesV);
    For(auto g = xsumSlice.iterate(xD.dim(Blocked::Block))) {
        PData pdata;
        For(auto kk = xsumSlice.iterate(xD.dim(Blocked::Row))) {
            IData xk = xsumSlice.iram.load(xD[g][kk]);
            pdata = xsumSlice.alu.elementwiseProductAccumulate(xk, ones);
        }
        For(auto av = xsumSlice.iterate(pdata.dim(PData::Vectors))) {
            QData res = xsumSlice.act.clamp(pdata[av], cMin, cMax);
            xsumSlice.append(stage8[g], res);
        }
    }
    xsumSlice.createSliceTaskOp(rewriter, loc);

    Slice bcastSlice("mqmm_bcast");
    WData bcastOnes = bcastSlice.wram.load(onesV);
    LData xrows = partfxr[1]; // appended to a row at a time
    // Trip counts come from the vectorized parent: [2, nB, vectors, wW].
    LData partfxrV = partfxr;
    partfxrV.vectorize(wW);
    For(auto g = bcastSlice.iterate(partfxrV.dim(Blocked::Row))) {
        For(auto jt = bcastSlice.iterate(partfxrV.dim(Blocked::Vectors))) {
            IData s = bcastSlice.iram.load(stage8[g]);
            PData pdata = bcastSlice.alu.elementwiseProductAccumulate(s, bcastOnes);
            For(auto av = bcastSlice.iterate(pdata.dim(PData::Vectors))) {
                QData res = bcastSlice.act.clamp(pdata[av], cMin, cMax);
                bcastSlice.append(xrows[g], res);
            }
        }
    }
    bcastSlice.createSliceTaskOp(rewriter, loc);
}

// Y[j] = sum_g D[g,j]*W[g,j] over nR rows (2*nB in zp mode: partf|xsum against
// scale|nzs -- the correction is just extra blocks in the same stream).
void buildFinalAccumulateTask(
    PatternRewriter &rewriter, Location loc, torq_hl::MixedQuantMatMulOp op, Value wscale,
    Value scratch, Value y, int nR, int wW, int Np, int32_t cMin, int32_t cMax
) {
    Slice slice("mqmm_final");
    LData partf(scratch); // partf(|xsumrows) as one [nR, Np] stream
    partf.setShape({nR, Np});
    partf.vectorize(wW);
    LData ws(wscale);
    ws.setShape({nR, Np});
    ws.vectorize(wW);
    // The op's result is [1, 1, N]; drain it as one run of N values.
    LData yD(y);
    while (yD.shape().size() > 1)
        yD.eraseDim(0);
    For(auto jt = slice.iterate(partf.dim(Rows::Vectors))) {
        PData pdata;
        For(auto g = slice.iterate(partf.dim(Rows::Row))) {
            IData i = slice.iram.load(partf[g][jt]);
            WData w = slice.wram.load(ws[g][jt]);
            pdata = slice.alu.elementwiseProductAccumulate(i, w);
        }
        For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
            QData res = slice.act.clamp(pdata[av], cMin, cMax);
            slice.append(yD, res);
        }
    }
    auto taskOp = slice.createSliceTaskOp(rewriter, loc);
    rewriter.replaceOp(op, taskOp);
}

} // namespace

template <>
LogicalResult MixedQuantMatMulPattern::transform(
    torq_hl::MixedQuantMatMulOp op, PatternRewriter &rewriter
) const {
    Location loc = op.getLoc();
    constexpr DType kElemType = DType::bf16; // bf16 end to end
    const int32_t cMin = minVal(kElemType), cMax = maxVal(kElemType);

    Value y = op.getInit();
    Value x = op.getX();
    Value wq = op.getWq();
    Value wscale = op.getWscale();
    Value scratch = op.getScratch();

    LData wqL(wq);
    const int wqRank = wqL.shape().size();
    if (wqRank != 2 && wqRank != 3)
        return op.emitError() << "wq must be rank 2 [K,N] or rank 3 [nB,B,N]";
    const int N = wqL.dim(wqRank - 1);
    const int K = wqRank == 3 ? wqL.dim(0) * wqL.dim(1) : wqL.dim(0);
    const int B = static_cast<int>(op.getBlockSize());
    if (B <= 0 || K % B != 0)
        return op.emitError() << "K=" << K << " must be a multiple of block_size=" << B;
    const int nB = K / B;

    LData wsL(wscale);
    const bool hasZp = wsL.dim(0) == 2 * nB + 1;
    if (!hasZp && wsL.dim(0) != nB)
        return op.emitError() << "wscale rows must be nB or 2*nB+1";

    // Drain width (8 on SL2610); must match the raise's kMqmmDrain.
    Slice macSlice("mqmm_mac");
    const int wW = macSlice.act.width(kElemType, kElemType);
    assert(
        wW == HwInfo::act_width / 2 && "MQMM kernel drain width must match the raise's kMqmmDrain"
    );
    // Ragged tiles: partial final group, caller sizes scratch/output to groups.
    const int Np = llvm::alignTo(N, wW);

    // scratch = partf[nB,Np] | (zp) xsumrows[nB,Np] | stage8[nB,wW]
    LData partfxr(scratch);
    partfxr.setShape({hasZp ? 2 : 1, nB, Np});

    buildMacPartialsTask(
        rewriter, loc, macSlice, x, wq, wqRank, partfxr[0], nB, B, wW, kElemType, cMin, cMax
    );
    if (hasZp)
        buildXsumCorrectionTasks(rewriter, loc, x, wscale, scratch, nB, B, wW, Np, cMin, cMax);
    buildFinalAccumulateTask(
        rewriter, loc, op, wscale, scratch, y, hasZp ? 2 * nB : nB, wW, Np, cMin, cMax
    );
    return success();
}

} // namespace mlir::syna::torq
