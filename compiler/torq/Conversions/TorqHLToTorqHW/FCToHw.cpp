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
static torq_hw::SliceTaskOp
lowerToHwPerChannel(torq_hl::FullyConnectedOp op, PatternRewriter &rewriter) {

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

    // Pass weights first (IData) and input second (WData) so the runtime maps them
    // to the expected memories for FC lowering.
    return torq_hw::SliceTaskOp::create(
        rewriter, op.getLoc(), slice.name(), op.getWeights(), op.getInput(), op.getScaleBias(),
        op.getInit(), slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );
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

static torq_hw::SliceTaskOp lowerToHw(torq_hl::FullyConnectedOp op, PatternRewriter &rewriter) {
    LData biasScale(op.getScaleBias());
    bool isSingleBias =
        (biasScale.shape().size() == 1 || biasScale.shape().size() == 2) && biasScale.dim(0) == 1;
    bool isBatchScaled = op.getIsBatchScaled() || isSingleBias;

    if (isBatchScaled)
        return lowerToHwBatchScaled(op, rewriter, isSingleBias);
    return lowerToHwPerChannel(op, rewriter);
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
