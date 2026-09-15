// FCToHw.cpp - lower torq_hl FullyConnected to torq_hw::SliceTaskOp
#include "Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"
#include "llvm/Support/Debug.h"

#include "llvm/Support/Debug.h"

using namespace mlir::syna::torq_hw;

#define DEBUG_TYPE "torq-lower-torqhl"

namespace mlir::syna::torq {

// Layout of the in/out/weight tensors for processing
struct In : Vectorized {
    enum { N, IC };
};

struct Out {
    enum { N, OC };
};

struct Weight {
    enum { IC, OCVect, OCElement };
};

struct BiasScale {
    enum { OCVect, ActVect, ActItems, Items };
};

// Per-output-channel bias/scale kernel.
//
// Expected tensor layouts:
//   input    : [N, IC]
//   weight   : [IC, OCVect, OCElement]   where OCElement = alu.iWidth(wType, inType)
//   biasScale: [OCVect, ceil(OCElement/actVectSize), actVectSize, biasScaleWidth(inType)]
//   output   : [N, OC]
//
// Each output-channel vector (OCVect) has its own bias/scale entry, applied after
// the IC reduction.
//
// `rowOffset` / `rowCount` select a range of input rows (N). They are the whole tensor
// unless the fast kernel below peeled a tail of rows off for this kernel to pick up.
static torq_hw::SliceTaskOp lowerToHwPerChannel(
    torq_hl::FullyConnectedOp op, PatternRewriter &rewriter, int rowOffset, int rowCount
) {

    LData input(op.getInput());
    LData output(op.getInit());
    LData weight(op.getWeights());
    Slice slice("fc");
    weight.forceReshapeDim(1, {-1, slice.alu.iWidth(weight.elementType(), input.elementType())});
    LData biasScale(op.getScaleBias());

    const auto inType = input.elementType();
    const auto wType = weight.elementType();
    const int weightVectSize = weight.dim(Weight::OCElement);
    const int actVectSize = std::min(weightVectSize, slice.act.width(wType, inType, true));
    biasScale.forceReshapeDim(
        0, {-1, (int)div_ceil(weightVectSize, actVectSize), actVectSize, biasScaleWidth(inType)}
    );

    input.subviewDim(In::N, rowOffset, rowCount);
    output.subviewDim(Out::N, rowOffset, rowCount);

    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto ocv = slice.iterate(weight.dim(Weight::OCVect))) {
            PData pdata;
            For(auto icv = slice.iterate(input.dim(In::IC))) {
                IData fcWeights = slice.iram.load(weight[icv][ocv]);
                WData fcInput = slice.wram.load(input[batch][icv]);
                pdata = slice.alu.scalarProductAccumulate(fcWeights, fcInput);
            }
            For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                BData bdata = slice.bram.load(biasScale[ocv][av]);
                QData res = slice.act.rescaleClamp(
                    pdata[av], bdata, op.getShiftFactor(), op.getOutputZp(), op.getOutputMin(),
                    op.getOutputMax()
                );
                slice.append(output[batch], res);
            }
        }
    }

    return slice.createSliceTaskOp(rewriter, op.getLoc());
}

// Batch-scaled bias/scale kernel.
//
// Expected tensor layouts:
//   input    : [N, IC]
//   weight   : [IC, OCVect, OCElement]   where OCElement = alu.iWidth(wType, inType)
//   biasScale: [N, biasScaleWidth(inType)]  (one entry per batch element, shared across
//              all output channels). When isSingleBias is true, biasScale has shape
//              [1, biasScaleWidth(inType)] and is broadcast to match N.
//   output   : [N, OC]
//
// Used when op.getIsBatchScaled() is set, or when a single bias (biasScale.dim(0)==1)
// is broadcast over the whole batch.
static torq_hw::SliceTaskOp
lowerToHwBatchScaled(torq_hl::FullyConnectedOp op, PatternRewriter &rewriter, bool isSingleBias) {

    LData input(op.getInput());
    LData output(op.getInit());
    LData weight(op.getWeights());
    Slice slice("fc");
    weight.forceReshapeDim(1, {-1, slice.alu.iWidth(weight.elementType(), input.elementType())});
    LData biasScale(op.getScaleBias());

    biasScale.forceReshapeDim(0, {-1, biasScaleWidth(input.elementType())});
    if (isSingleBias)
        biasScale.broadcastAs(output, 1);

    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto ocv = slice.iterate(weight.dim(Weight::OCVect))) {
            PData pdata;
            For(auto icv = slice.iterate(input.dim(In::IC))) {
                IData fcWeights = slice.iram.load(weight[icv][ocv]);
                WData fcInput = slice.wram.load(input[batch][icv]);
                pdata = slice.alu.scalarProductAccumulate(fcWeights, fcInput);
            }
            BData bdata = slice.bram.load(biasScale[batch]);
            For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                QData res = slice.act.rescaleClamp(
                    pdata[av], bdata, op.getShiftFactor(), op.getOutputZp(), op.getOutputMin(),
                    op.getOutputMax()
                );
                slice.append(output[batch], res);
            }
        }
    }

    // Pass weights first (IData) and input second (WData) so the runtime maps them
    // to the expected memories for FC lowering.
    return torq_hw::SliceTaskOp::create(
        rewriter, op.getLoc(), slice.name(), op.getWeights(), op.getInput(), op.getScaleBias(),
        op.getInit(), slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );
}

// Row-blocked variant of the kernel above: groups up to wram.transposeHeight() input rows
// into one WRAM transpose block and accumulates an outer product, like lowerToFastMatmul()
// but with WRAM holding the input rows. bias/scale is indexed by output channel only, so a
// group shares one BREG. A rowCount below the transpose height runs in one pass with a
// shorter block, the same way lowerToFastMatmul() handles M < 4.
static torq_hw::SliceTaskOp lowerToHwPerChannelFast(
    torq_hl::FullyConnectedOp op, PatternRewriter &rewriter, int rowOffset, int rowCount
) {

    LData input(op.getInput());
    LData output(op.getInit());
    LData weight(op.getWeights());
    Slice slice("fc-fast");
    const int rowVectSize = std::min(slice.wram.transposeHeight(), rowCount);
    const int aluWidth = slice.alu.iWidth(weight.elementType(), input.elementType(), rowVectSize);
    weight.vectorize(aluWidth);
    LData biasScale(op.getScaleBias());

    const auto inType = input.elementType();
    const auto wType = weight.elementType();
    const int weightVectSize = weight.dim(Weight::OCElement);
    const int actVectSize = std::min(weightVectSize, slice.act.width(wType, inType, true));
    biasScale.forceReshapeDim(
        0, {-1, (int)div_ceil(weightVectSize, actVectSize), actVectSize, biasScaleWidth(inType)}
    );

    input.subviewDim(In::N, rowOffset, rowCount);
    output.subviewDim(Out::N, rowOffset, rowCount);

    const int rowChunks = maxDivisor(input.dim(In::IC), slice.wram.transposeWidth());

    // input : [N, IC]      -> [NGroups, rowVectSize, ICGroups, rowChunks]
    // weight: [IC, ...]    -> [ICGroups, rowChunks, OCVect, OCElement]
    // output: [N, OC]      -> [NGroups, rowVectSize, OC]
    input.reshapeDim(In::IC, {-1, rowChunks});
    input.reshapeDim(In::N, {-1, rowVectSize});
    weight.reshapeDim(Weight::IC, {-1, rowChunks});
    output.reshapeDim(Out::N, {-1, rowVectSize});

    enum { NGroups, NRows, ICGroups, ICChunk }; // input dims after the reshapes
    enum { WICGroups, WICChunk, WOCVect };      // weight dims after the reshape

    For(auto nog = slice.iterate(input.dim(NGroups))) {
        For(auto ocv = slice.iterate(weight.dim(WOCVect))) {
            PData pdata;
            For(auto ico = slice.iterate(weight.dim(WICGroups))) {
                // One transpose block of rowVectSize input rows x rowChunks reduction steps.
                WData fcInput = slice.wram.transpose(input[nog][":"][ico]);
                For(auto ici = slice.iterate(fcInput.dim(0))) {
                    IData fcWeights = slice.iram.load(weight[ico][ici][ocv]);
                    pdata = slice.alu.outerProductAccumulate(fcWeights, fcInput[ici]);
                }
            }
            For(auto o = slice.iterate(pdata.dim(PData::Outer))) {
                For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                    BData bdata = slice.bram.load(biasScale[ocv][av]);
                    QData res = slice.act.rescaleClamp(
                        pdata[o][av], bdata, op.getShiftFactor(), op.getOutputZp(),
                        op.getOutputMin(), op.getOutputMax()
                    );
                    slice.append(output[nog][o], res);
                }
            }
        }
    }

    LLVM_DEBUG(llvm::dbgs() << "Lowered to fast fully_connected kernel\n");
    return slice.createSliceTaskOp(rewriter, op.getLoc());
}

// Rows per WRAM transpose block: a fixed HW property, but only reachable through a
// Slice, hence the throwaway one.
static int fastPathRowBlock() { return Slice().wram.transposeHeight(); }

// Needs a float datapath and a rank-2 input/output, which the row and reduction splits above
// assume.
//
// int8 is out, and what stops it is the drain rather than the outer product. A per-item
// bias/scale caps the ACT width (SlicePrivate::actWidth): the cap is 4 for int8 and 8 for
// bf16. One pass then has to drain iWidth/actWidth vectors, which is 32/8 = 4 for bf16 but
// 64/4 = 16 for int8. A block of R rows multiplies that count, and _reg_ndl_desc_gen()
// asserts dims.sn <= 16, so int8 passes the limit at R = 2 already. A narrower int8 pass
// would fit, but an iWidth of 16 instead of 64 needs four times as many passes over the
// output channels, which undoes the four rows the block gives.
static bool canUseFastPerChannel(torq_hl::FullyConnectedOp op) {
    LData input(op.getInput());
    LData output(op.getInit());

    return isFloat(input.elementType()) && input.dims().size() == 2 && output.dims().size() == 2;
}

static torq_hw::SliceTaskOp lowerToHw(torq_hl::FullyConnectedOp op, PatternRewriter &rewriter) {
    LData biasScale(op.getScaleBias());
    bool isSingleBias =
        (biasScale.shape().size() == 1 || biasScale.shape().size() == 2) && biasScale.dim(0) == 1;
    bool isBatchScaled = op.getIsBatchScaled() || isSingleBias;

    if (isBatchScaled)
        return lowerToHwBatchScaled(op, rewriter, isSingleBias);

    const int rowCount = LData(op.getInput()).dim(In::N);
    const int rowVectSize = fastPathRowBlock();
    if (canUseFastPerChannel(op)) {
        if (int peeledRows = rowCount % rowVectSize) {
            // A single row has nothing to block, so it goes to the scalar kernel the way
            // lowerToMatmul() picks up a single-row fast-matmul remainder. Two or three rows
            // still run in one pass, with a block that short.
            auto peeled =
                peeledRows == 1
                    ? lowerToHwPerChannel(op, rewriter, rowCount - peeledRows, peeledRows)
                    : lowerToHwPerChannelFast(op, rewriter, rowCount - peeledRows, peeledRows);
            if (!peeled)
                return nullptr;
            if (rowCount < rowVectSize)
                return peeled;
        }
        return lowerToHwPerChannelFast(op, rewriter, 0, rowCount - rowCount % rowVectSize);
    }
    return lowerToHwPerChannel(op, rewriter, 0, rowCount);
}

LogicalResult convertToHw(torq_hl::FullyConnectedOp op, PatternRewriter &rewriter) {
    torq_hw::SliceTaskOp hwOp;

    if (!(hwOp = lowerToHw(op, rewriter))) {
        return failure();
    }
    rewriter.replaceOp(op, hwOp.getOperation()->getResults());
    return success();
}

} // namespace mlir::syna::torq
