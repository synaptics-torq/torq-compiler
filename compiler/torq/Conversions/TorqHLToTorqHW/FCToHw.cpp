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
    enum { OCVect, IC, OCElement };
};

struct BiasScale {
    enum { OCVect, ActVect, ActItems, Items };
};

static torq_hw::SliceTaskOp lowerToHw(torq_hl::FullyConnectedOp op, PatternRewriter &rewriter) {

    // Wrap operands in LData (same as conv)
    LData input(op.getInput());
    LData output(op.getInit());
    LData weight(op.getWeights());
    LData biasScale(op.getScaleBias());

    // Create a Slice for FC
    Slice slice("fc");

    // Reshape biasScale to match processing layout
    const auto inType = input.elementType();
    const auto wType = weight.elementType();
    const int weightVectSize = weight.dim(Weight::OCElement);
    const int actVectSize = std::min(weightVectSize, slice.act.width(wType, inType, true));
    biasScale.reshapeDim(
        0, {-1, (int)div_ceil(weightVectSize, actVectSize), actVectSize, scaleBiasWidth(inType)},
        true
    );

    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto ocv = slice.iterate(weight.dim(Weight::OCVect))) {
            PData pdata;
            // Reduce over input channels
            For(auto icv = slice.iterate(input.dim(In::IC))) {
                // Load weights to IRAM and input to WRAM (to compute multiple out at the same time)
                IData fcWeights = slice.iram.load(weight[ocv][icv]);
                WData fcInput = slice.wram.load(input[batch][icv]);
                pdata = slice.alu.scalarProductAccumulate(fcWeights, fcInput);
            }

            // Reshape partials since ACT can only process actVectSize results at a time
            int actVects = biasScale.dim(BiasScale::ActVect);
            assert(elementCount(pdata.shape()) % actVects == 0);
            pdata.setShape({actVects, elementCount(pdata.shape()) / actVects});

            // Apply biaas and scale
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
    return rewriter.create<torq_hw::SliceTaskOp>(
        op.getLoc(), slice.name(), op.getWeights(), op.getInput(), op.getScaleBias(), op.getInit(),
        slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );
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
