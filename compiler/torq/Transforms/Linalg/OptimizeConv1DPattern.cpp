// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/TorqHw.h"
#include "torq/Utils/TorqUtils.h"

#include "algorithm"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "numeric"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-conv1d-pattern"

namespace mlir::syna::torq {

llvm::cl::opt<bool> clConv1dAsMatmul(
    "torq-convert-conv1d-to-matmul", llvm::cl::desc("Convert conv1d to imToCol + matmul"),
    llvm::cl::init(false)
);

llvm::cl::opt<bool> clConv1dToGenericConv1D(
    "torq-convert-conv1d-to-generic",
    llvm::cl::desc("Convert conv1d to generic conv1d (5D output with preserved kernel dim)"),
    llvm::cl::init(false)
);

llvm::cl::opt<bool> clConv1dToConv2D(
    "torq-convert-conv1d-to-conv2d", llvm::cl::desc("Convert conv1d to conv2d"),
    llvm::cl::init(false)
);

llvm::cl::opt<bool> clConv1dTruncateForReduce(
    "torq-conv1d-truncate-for-reduce",
    llvm::cl::desc("Truncate conv1d output to bf16 before reduce sum to save memory bandwidth. "
                   "When false (default): conv1d outputs f32 directly to reduce (accurate). "
                   "When true: insert truncf between conv1d and reduce (memory efficient)."),
    llvm::cl::init(false)
);

// LRAM budget for a single non-tileable op: total LRAM minus 14k headroom for descriptors
// (used to be TorqHw::get().getAvailableLramSize()).
static int64_t getLramTilingBudget() {
    return static_cast<int64_t>(TorqHw::get().getLramSize() - 14 * 1024);
}

static int64_t elemTypeBytes(Type elemType) { return (elemType.getIntOrFloatBitWidth() + 7) / 8; }

// The conv1d im2col materialises the collapsed input and the unfolded [Ow, C*Kw] result as a
// single non-tileable torq_hl.im2col op. Returns false when both buffers exceed the tiling budget.
static bool im2ColFitsInLram(int64_t C, int64_t W, int64_t Ow, int64_t Kw, Type elemType) {
    int64_t elemBytes = elemTypeBytes(elemType);
    int64_t bytes = (C * W) * elemBytes + (Ow * C * Kw) * elemBytes;
    return bytes <= getLramTilingBudget();
}

// Emit the im2col unfold ([Ow, C*Kw]) as an affine-map linalg.generic marked torq.im2col.
// Used when the NDL im2col op is too large to stay LRAM-resident: MarkHostExecutorPass routes
// the marked generic to the host (torq_hl.im2col is not tileable).
static Value emitHostIm2Col(
    PatternRewriter &rewriter, Location loc, Value input, int64_t C, int64_t W, int64_t Kw,
    int64_t Ow, int64_t stride, int64_t dilation, RankedTensorType unfoldedType
) {
    auto elemType = unfoldedType.getElementType();

    // Collapse [N,C,W] -> [C,W] so the indexing map is rank-2 with no constant dims (constant
    // dims get folded by canonicalization, causing an operand-rank vs map-result-rank mismatch
    // later).
    auto collapsedInputType = RankedTensorType::get({C, W}, elemType);
    auto collapsedInput = tensor::CollapseShapeOp::create(
        rewriter, loc, collapsedInputType, input, ArrayRef<ReassociationIndices>{{0, 1}, {2}}
    );

    // (d0=Ow, d1=C*Kw) -> (d1/Kw, d0*stride + (d1%Kw)*dilation)
    auto dim0 = rewriter.getAffineDimExpr(0);
    auto dim1 = rewriter.getAffineDimExpr(1);
    auto channelIdx = dim1.floorDiv(rewriter.getAffineConstantExpr(Kw));
    auto kernelIdx = dim1 % rewriter.getAffineConstantExpr(Kw);
    auto inputPosExpr = dim0 * rewriter.getAffineConstantExpr(stride) +
                        kernelIdx * rewriter.getAffineConstantExpr(dilation);

    SmallVector<AffineExpr> unfoldIndexExprs = {channelIdx, inputPosExpr};
    auto unfoldIndexMap = AffineMap::get(2, 0, unfoldIndexExprs, rewriter.getContext());
    auto outputIndexMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());

    auto unfoldedInit = tensor::EmptyOp::create(rewriter, loc, unfoldedType.getShape(), elemType);
    SmallVector<utils::IteratorType> iteratorTypes(2, utils::IteratorType::parallel);

    auto im2col = linalg::GenericOp::create(
        rewriter, loc, TypeRange{unfoldedType}, ValueRange{collapsedInput.getResult()},
        ValueRange{unfoldedInit}, ArrayRef<AffineMap>{unfoldIndexMap, outputIndexMap},
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
            linalg::YieldOp::create(nestedBuilder, nestedLoc, blockArgs[0]);
        }
    );
    im2col->setAttr("torq.im2col", rewriter.getBoolAttr(true));
    return im2col.getResult(0);
}

// Emit the im2col unfold ([Ow, C*Kw]) as torq_hl.im2col: the hardware lowering reads sliding input
// windows via an affine DEDR NDL and writes the unfolded layout via DEQW.
static Value emitIm2Col(
    PatternRewriter &rewriter, Location loc, Value input, int64_t C, int64_t W, int64_t Kw,
    int64_t Ow, int64_t stride, int64_t dilation, RankedTensorType unfoldedType
) {
    auto elemType = unfoldedType.getElementType();

    // Collapse [N,C,W] -> [C,W]; the op addresses input[c, ow*stride + kw*dilation].
    auto collapsedInputType = RankedTensorType::get({C, W}, elemType);
    auto collapsedInput = tensor::CollapseShapeOp::create(
        rewriter, loc, collapsedInputType, input, ArrayRef<ReassociationIndices>{{0, 1}, {2}}
    );

    auto unfoldedInit = tensor::EmptyOp::create(rewriter, loc, unfoldedType.getShape(), elemType);

    auto im2col = torq_hl::Im2ColOp::create(
        rewriter, loc, TypeRange{unfoldedType}, unfoldedInit.getResult(),
        collapsedInput.getResult(), rewriter.getI64IntegerAttr(stride),
        rewriter.getI64IntegerAttr(dilation), rewriter.getI64IntegerAttr(Kw)
    );
    return im2col.getOutput();
}

// Largest output-column block whose windowed im2col (sliced input window + unfolded patches)
// stays within the LRAM tiling budget. torq_hl.im2col is not tileable, so tiling along the
// output width keeps each unfold LRAM-resident. Returns 0 when even one column does not fit
// (pathological; caller falls back to the host im2col).
static int64_t computeIm2ColBlockOw(
    int64_t C, int64_t W, int64_t Ow, int64_t Kw, int64_t stride, int64_t dilation, Type elemType
) {
    int64_t K = C * Kw;
    int64_t elemBytes = elemTypeBytes(elemType);
    int64_t budget = getLramTilingBudget();

    // A block of bOw output columns reads windowW input columns; offsets are window-relative,
    // and valid-conv output sizing guarantees windowW <= W.
    auto blockFits = [&](int64_t bOw) {
        int64_t windowW = (bOw - 1) * stride + (Kw - 1) * dilation + 1;
        int64_t bytes = (C * windowW) * elemBytes + (bOw * K) * elemBytes;
        return bytes <= budget;
    };

    int64_t lo = 1, hi = Ow, best = 0;
    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (blockFits(mid)) {
            best = mid;
            lo = mid + 1;
        }
        else {
            hi = mid - 1;
        }
    }
    return best;
}

// LRAM estimate for one native-conv1d Ow block. The block runs as two sequential
// LRAM-resident stages that reuse the same budget, so size it by the larger stage:
//   - generic: sliced input window + full filter + [N,F,1,bOw,Kw] output
//   - kw reduce: [N,F,1,bOw,Kw] input + [N,F,1,bOw] output + per-channel f32 bias
// Accounting for the reduce stage matters because a conv bias folds into the reduce
// (FoldConvBiasIntoReducePattern): if the reduce stage overflows LRAM a later pass
// channel-splits it, and the per-channel bias (sized to full F) no longer matches the
// split reduce output.
static int64_t genericConv1dBlockBytes(
    int64_t N, int64_t C, int64_t F, int64_t bOw, int64_t Kw, int64_t stride, int64_t dilation,
    Type inputElemType, Type filterElemType, Type outputElemType
) {
    int64_t windowW = (bOw - 1) * stride + (Kw - 1) * dilation + 1;
    int64_t inBytes = N * C * windowW * elemTypeBytes(inputElemType);
    int64_t filterBytes = F * C * Kw * elemTypeBytes(filterElemType);
    int64_t genericOutBytes = N * F * bOw * Kw * elemTypeBytes(outputElemType);
    int64_t reduceOutBytes = N * F * bOw * elemTypeBytes(outputElemType);
    int64_t biasBytes = F * elemTypeBytes(outputElemType);

    int64_t genericStageBytes = inBytes + filterBytes + genericOutBytes;
    int64_t reduceStageBytes = genericOutBytes + reduceOutBytes + biasBytes;
    return std::max(genericStageBytes, reduceStageBytes);
}

// Largest Ow block for native conv1d generic + kw reduce that stays LRAM-resident.
// Returns 0 when even one output column does not fit (caller falls back to im2col).
static int64_t computeGenericConv1dBlockOw(
    int64_t N, int64_t C, int64_t F, int64_t Ow, int64_t Kw, int64_t stride, int64_t dilation,
    Type inputElemType, Type filterElemType, Type outputElemType
) {
    int64_t budget = getLramTilingBudget();

    auto blockFits = [&](int64_t bOw) {
        return genericConv1dBlockBytes(
                   N, C, F, bOw, Kw, stride, dilation, inputElemType, filterElemType, outputElemType
               ) <= budget;
    };

    int64_t lo = 1, hi = Ow, best = 0;
    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (blockFits(mid)) {
            best = mid;
            lo = mid + 1;
        }
        else {
            hi = mid - 1;
        }
    }
    return best;
}

// linalg.generic conv1d multiply-accumulate; output [N,F,1,Ow,Kw] f32. Multi-channel convs add a
// trailing reduction over C; a single-channel conv stays all-parallel (channel index folded to 0)
// so downstream tile-and-fuse tiles the parallel output dims instead of a trivial C reduction.
static Value emitGenericConv1dBlock(
    PatternRewriter &rewriter, Location loc, Value input4D, Value filter4D, int64_t N, int64_t F,
    int64_t Ow, int64_t Kw, int64_t stride, int64_t dilation
) {
    Type computeElemType = rewriter.getF32Type();
    int64_t C = cast<RankedTensorType>(input4D.getType()).getDimSize(1);
    bool reduceC = C > 1;
    unsigned numDims = reduceC ? 6 : 5;

    SmallVector<int64_t> output5DShape = {N, F, 1, Ow, Kw};
    auto output5DType = RankedTensorType::get(output5DShape, computeElemType);
    auto output5DInit =
        linalg::FillOp::create(
            rewriter, loc, ValueRange{createZeroConstant(rewriter, loc, computeElemType)},
            ValueRange{tensor::EmptyOp::create(rewriter, loc, output5DShape, computeElemType)}
        )
            .result();

    auto n = rewriter.getAffineDimExpr(0);
    auto f = rewriter.getAffineDimExpr(1);
    auto kh = rewriter.getAffineDimExpr(2);
    auto ow = rewriter.getAffineDimExpr(3);
    auto kw = rewriter.getAffineDimExpr(4);
    AffineExpr c = reduceC ? rewriter.getAffineDimExpr(5) : rewriter.getAffineConstantExpr(0);

    SmallVector<AffineExpr> inputMapExprs = {
        n, c, kh,
        ow * rewriter.getAffineConstantExpr(stride) + kw * rewriter.getAffineConstantExpr(dilation)
    };
    auto inputMap = AffineMap::get(numDims, 0, inputMapExprs, rewriter.getContext());

    SmallVector<AffineExpr> filterMapExprs = {f, c, kh, kw};
    auto filterMap = AffineMap::get(numDims, 0, filterMapExprs, rewriter.getContext());

    SmallVector<AffineExpr> outputMapExprs = {n, f, kh, ow, kw};
    auto outputMap = AffineMap::get(numDims, 0, outputMapExprs, rewriter.getContext());

    SmallVector<utils::IteratorType> iteratorTypes(5, utils::IteratorType::parallel);
    if (reduceC) {
        iteratorTypes.push_back(utils::IteratorType::reduction);
    }

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{output5DType}, ValueRange{input4D, filter4D},
        ValueRange{output5DInit}, ArrayRef<AffineMap>{inputMap, filterMap, outputMap},
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
            Value inputVal = blockArgs[0];
            Value filterVal = blockArgs[1];
            Value accum = blockArgs[2];

            if (inputVal.getType() != computeElemType) {
                inputVal =
                    arith::ExtFOp::create(nestedBuilder, nestedLoc, computeElemType, inputVal);
            }
            if (filterVal.getType() != computeElemType) {
                filterVal =
                    arith::ExtFOp::create(nestedBuilder, nestedLoc, computeElemType, filterVal);
            }
            if (accum.getType() != computeElemType) {
                accum = arith::ExtFOp::create(nestedBuilder, nestedLoc, computeElemType, accum);
            }

            auto mul = arith::MulFOp::create(nestedBuilder, nestedLoc, inputVal, filterVal);
            auto add = arith::AddFOp::create(nestedBuilder, nestedLoc, mul, accum);
            linalg::YieldOp::create(nestedBuilder, nestedLoc, add.getResult());
        }
    );
    return generic.getResult(0);
}

// Reduce kw on [N,F,1,Ow,Kw] -> [N,F,1,Ow] f32.
static Value emitKwReduceBlock(
    PatternRewriter &rewriter, Location loc, Value generic5D, int64_t N, int64_t F, int64_t Ow
) {
    Type reduceOutputType = rewriter.getF32Type();
    SmallVector<int64_t> reducedShape = {N, F, 1, Ow};
    Value zeroValue = createZeroConstant(rewriter, loc, reduceOutputType);
    auto reduceInit = tensor::EmptyOp::create(rewriter, loc, reducedShape, reduceOutputType);
    Value zeroTensor =
        linalg::FillOp::create(rewriter, loc, ValueRange{zeroValue}, ValueRange{reduceInit})
            .result();

    auto reduceOp = linalg::ReduceOp::create(
        rewriter, loc, ValueRange{generic5D}, ValueRange{zeroTensor}, 4,
        [&](OpBuilder &b, Location l, ValueRange args) {
            Value lhs = args[0];
            Value rhs = args[1];
            if (lhs.getType() != reduceOutputType) {
                lhs = arith::ExtFOp::create(b, l, reduceOutputType, lhs);
            }
            if (rhs.getType() != reduceOutputType) {
                rhs = arith::ExtFOp::create(b, l, reduceOutputType, rhs);
            }
            auto sum = arith::AddFOp::create(b, l, lhs, rhs);
            linalg::YieldOp::create(b, l, ValueRange{sum});
        }
    );
    return reduceOp.getResults()[0];
}

// Build [N,F,1,Ow] by tiling native conv1d generic + kw reduce along output width.
static Value emitTiledNativeConv1d(
    PatternRewriter &rewriter, Location loc, Value input, Value filter4D, int64_t N, int64_t C,
    int64_t F, int64_t Ow, int64_t Kw, int64_t stride, int64_t dilation, int64_t blockOw,
    Type inputElemType
) {
    Type reduceOutputType = rewriter.getF32Type();
    Value result =
        tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{N, F, 1, Ow}, reduceOutputType)
            .getResult();
    result = linalg::FillOp::create(
                 rewriter, loc, ValueRange{createZeroConstant(rewriter, loc, reduceOutputType)},
                 ValueRange{result}
    )
                 .result();

    SmallVector<ReassociationIndices> inputReassoc = {{0}, {1}, {2, 3}};

    for (int64_t ow0 = 0; ow0 < Ow; ow0 += blockOw) {
        int64_t curOw = std::min(blockOw, Ow - ow0);
        int64_t w0 = ow0 * stride;
        int64_t windowW = (curOw - 1) * stride + (Kw - 1) * dilation + 1;

        auto sliceType = RankedTensorType::get({N, C, windowW}, inputElemType);
        Value inputSlice = tensor::ExtractSliceOp::create(
            rewriter, loc, sliceType, input,
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(0), rewriter.getIndexAttr(0), rewriter.getIndexAttr(w0)
            },
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(N), rewriter.getIndexAttr(C), rewriter.getIndexAttr(windowW)
            },
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(1), rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)
            }
        );

        auto input4DType = RankedTensorType::get({N, C, 1, windowW}, inputElemType);
        Value input4D =
            tensor::ExpandShapeOp::create(rewriter, loc, input4DType, inputSlice, inputReassoc);

        Value block5D = emitGenericConv1dBlock(
            rewriter, loc, input4D, filter4D, N, F, curOw, Kw, stride, dilation
        );
        Value block4D = emitKwReduceBlock(rewriter, loc, block5D, N, F, curOw);

        result = tensor::InsertSliceOp::create(
            rewriter, loc, block4D, result,
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(0), rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
                rewriter.getIndexAttr(ow0)
            },
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(N), rewriter.getIndexAttr(F), rewriter.getIndexAttr(1),
                rewriter.getIndexAttr(curOw)
            },
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(1), rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                rewriter.getIndexAttr(1)
            }
        );
    }
    return result;
}

// Build the [Ow, F] conv1d matmul result by tiling im2col + matmul along the output width.
// Each block slices the minimal input window it reads, runs torq_hl.im2col on that window,
// and matmuls into the corresponding rows of the result.
static Value emitTiledIm2ColConv1dMatmul(
    PatternRewriter &rewriter, Location loc, Value input, Value transposedFilter, int64_t C,
    int64_t Kw, int64_t Ow, int64_t F, int64_t stride, int64_t dilation, int64_t blockOw,
    Type elemType, Type outElemType
) {
    int64_t K = C * Kw;
    Value zero = createZeroConstant(rewriter, loc, outElemType);
    Value result =
        tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{Ow, F}, outElemType).getResult();

    for (int64_t ow0 = 0; ow0 < Ow; ow0 += blockOw) {
        int64_t curOw = std::min(blockOw, Ow - ow0);
        int64_t w0 = ow0 * stride;
        int64_t windowW = (curOw - 1) * stride + (Kw - 1) * dilation + 1;

        // Slice the input window [1, C, w0 : w0 + windowW] this block reads from.
        auto sliceType = RankedTensorType::get({1, C, windowW}, elemType);
        Value inputSlice = tensor::ExtractSliceOp::create(
            rewriter, loc, sliceType, input,
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(0), rewriter.getIndexAttr(0), rewriter.getIndexAttr(w0)
            },
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(1), rewriter.getIndexAttr(C), rewriter.getIndexAttr(windowW)
            },
            SmallVector<OpFoldResult>{
                rewriter.getIndexAttr(1), rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)
            }
        );

        // Offsets are window-relative, so this is a full unfold over the slice with W = windowW.
        auto blockUnfoldedType = RankedTensorType::get({curOw, K}, elemType);
        Value blockUnfolded = emitIm2Col(
            rewriter, loc, inputSlice, C, windowW, Kw, curOw, stride, dilation, blockUnfoldedType
        );

        // matmul [curOw, C*Kw] x [C*Kw, F] -> [curOw, F]
        auto blockMatmulType = RankedTensorType::get({curOw, F}, outElemType);
        auto blockInit = linalg::FillOp::create(
            rewriter, loc, zero,
            tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{curOw, F}, outElemType)
                .getResult()
        );
        auto blockMatmul = linalg::MatmulOp::create(
            rewriter, loc, TypeRange{blockMatmulType}, ValueRange{blockUnfolded, transposedFilter},
            ValueRange{blockInit.getResult(0)}
        );

        // Write this block's rows into the [Ow, F] result.
        result = tensor::InsertSliceOp::create(
            rewriter, loc, blockMatmul.getResults()[0], result,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(ow0), rewriter.getIndexAttr(0)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(curOw), rewriter.getIndexAttr(F)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)}
        );
    }
    return result;
}

// Elementwise f32 -> bf16 truncf as a parallel linalg.generic; rank is taken from `value`.
static Value emitTruncfToBF16(PatternRewriter &rewriter, Location loc, Value value) {
    auto srcType = cast<RankedTensorType>(value.getType());
    int64_t rank = srcType.getRank();
    auto bf16Type = rewriter.getBF16Type();
    auto resultType = RankedTensorType::get(srcType.getShape(), bf16Type);
    Value init = tensor::EmptyOp::create(rewriter, loc, srcType.getShape(), bf16Type).getResult();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> iterators(rank, utils::IteratorType::parallel);

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, ValueRange{value}, ValueRange{init},
        ArrayRef<AffineMap>{identityMap, identityMap}, iterators,
        [&](OpBuilder &b, Location l, ValueRange args) {
            auto truncf = arith::TruncFOp::create(b, l, bf16Type, args[0]);
            linalg::YieldOp::create(b, l, truncf.getResult());
        }
    );
    return generic.getResult(0);
}

// Reroute a fused f32->bf16 truncf generic's users back to its f32 input and erase it, so a
// later stage re-materialises the cast where it belongs. Clears `truncfOp` so it can't be reused.
static void dropTruncf(PatternRewriter &rewriter, linalg::GenericOp &truncfOp) {
    truncfOp->replaceAllUsesWith(ValueRange{truncfOp->getOperand(0)});
    rewriter.eraseOp(truncfOp);
    truncfOp = nullptr;
}

/// Optimization pattern for linalg.conv_1d operation.

struct Conv1DNcwFcwToLinalgMatmulPattern : public OpRewritePattern<linalg::Conv1DNcwFcwOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv1DNcwFcwOp convOp, PatternRewriter &rewriter) const override {
        auto loc = convOp.getLoc();

        // Extract tensors and shapes
        Value input = convOp.getInputs()[0];   // Input tensor [N,C,W]
        Value filter = convOp.getInputs()[1];  // Filter tensor [F,C,Kw]
        Value output = convOp.getOutputs()[0]; // Output tensor [N,F,Ow]

        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        // Extract dimensions
        ArrayRef<int64_t> inputShape = inputType.getShape();
        ArrayRef<int64_t> filterShape = filterType.getShape();
        ArrayRef<int64_t> outputShape = outputType.getShape();

        if (inputShape.size() != 3 || filterShape.size() != 3 || outputShape.size() != 3) {
            return rewriter.notifyMatchFailure(convOp, "Expected 3D tensors for Conv1D");
        }

        if (!isAllZerosTensor(output)) {
            return rewriter.notifyMatchFailure(convOp, "init is not all zeros");
        }

        // Extract convolution parameters
        SmallVector<int64_t> strides = llvm::to_vector<4>(
            llvm::map_range(convOp.getStrides(), [](APInt v) { return v.getSExtValue(); })
        );
        SmallVector<int64_t> dilations = llvm::to_vector<4>(
            llvm::map_range(convOp.getDilations(), [](APInt v) { return v.getSExtValue(); })
        );

        int64_t N = inputShape[0];       // Batch size
        int64_t C = inputShape[1];       // Input channels
        int64_t F = filterShape[0];      // Output channels/filters
        int64_t Kw = filterShape[2];     // Kernel width
        int64_t Ow = outputShape[2];     // Output width
        int64_t stride = strides[0];     // Stride value
        int64_t dilation = dilations[0]; // Dilation value

        // Steps 1-3: im2col (unfold input patches into [Ow, C*Kw]) + matmul -> [Ow, F].
        //
        // The unfold is torq_hl.im2col (affine NDL window read -> unfolded buffer). The op is not
        // tileable, so when the full unfold does not fit in LRAM we tile im2col + matmul along
        // the output width. Only a pathological conv whose single-column unfold still overflows
        // LRAM falls back to an affine-map generic marked torq.im2col on the host.
        auto elemType = inputType.getElementType();
        auto outputElemType = outputType.getElementType();
        int64_t W = inputShape[2];

        // Filter: [F, C, Kw] -> [F, C*Kw] -> [C*Kw, F] (transpose the filter rather than the
        // im2col output so the matmul lands directly in [Ow, F]).
        auto reshapedFilterType = RankedTensorType::get({F, C * Kw}, filterType.getElementType());
        auto reshapedFilter = tensor::CollapseShapeOp::create(
            rewriter, loc, reshapedFilterType, filter, ArrayRef<ReassociationIndices>{{0}, {1, 2}}
        );
        auto transposedFilterInit = tensor::EmptyOp::create(
            rewriter, loc, ArrayRef<int64_t>{C * Kw, F}, filterType.getElementType()
        );
        Value transposedFilter = linalg::TransposeOp::create(
                                     rewriter, loc, reshapedFilter.getResult(),
                                     transposedFilterInit, ArrayRef<int64_t>{1, 0}
        )
                                     .getResults()[0];

        // [Ow, C*Kw] x [C*Kw, F] -> [Ow, F]; accumulate into the output element type (f32).
        auto emitMatmul = [&](Value unfolded) -> Value {
            auto matmulResultType = RankedTensorType::get({Ow, F}, outputElemType);
            auto matmulInit = linalg::FillOp::create(
                rewriter, loc, createZeroConstant(rewriter, loc, outputElemType),
                tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{Ow, F}, outputElemType)
                    .getResult()
            );
            return linalg::MatmulOp::create(
                       rewriter, loc, TypeRange{matmulResultType},
                       ValueRange{unfolded, transposedFilter}, ValueRange{matmulInit.getResult(0)}
            )
                .getResults()[0];
        };

        Value matmulResult;
        if (im2ColFitsInLram(C, W, Ow, Kw, elemType)) {
            auto unfoldedType = RankedTensorType::get({Ow, C * Kw}, elemType);
            matmulResult = emitMatmul(
                emitIm2Col(rewriter, loc, input, C, W, Kw, Ow, stride, dilation, unfoldedType)
            );
        }
        else if (int64_t blockOw = computeIm2ColBlockOw(C, W, Ow, Kw, stride, dilation, elemType)) {
            matmulResult = emitTiledIm2ColConv1dMatmul(
                rewriter, loc, input, transposedFilter, C, Kw, Ow, F, stride, dilation, blockOw,
                elemType, outputElemType
            );
        }
        else {
            auto unfoldedType = RankedTensorType::get({Ow, C * Kw}, elemType);
            matmulResult = emitMatmul(
                emitHostIm2Col(rewriter, loc, input, C, W, Kw, Ow, stride, dilation, unfoldedType)
            );
        }

        // Transpose result [Ow, F] -> [F, Ow]
        auto transposedResultInit =
            tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{F, Ow}, outputElemType);
        auto transposedResult = linalg::TransposeOp::create(
            rewriter, loc, matmulResult, transposedResultInit, ArrayRef<int64_t>{1, 0}
        );

        // Step 4: expand [F, Ow] -> [N, F, Ow]
        if (N == 1) {
            SmallVector<int64_t> expandedShape = {N, F, Ow};
            auto expandedType = RankedTensorType::get(expandedShape, outputElemType);
            auto expandedResult = tensor::ExpandShapeOp::create(
                rewriter, loc, expandedType, transposedResult.getResults()[0],
                ArrayRef<ReassociationIndices>{{0, 1}, {2}}
            );

            rewriter.replaceOp(convOp, expandedResult.getResult());
        }
        else {
            return rewriter.notifyMatchFailure(
                convOp, "Batched Conv1D not supported in this pattern"
            );
        }

        return success();
    }
};

// True if `v` traces back (through layout/cast ops) to a compile-time constant,
// i.e. a normal conv weight rather than a runtime activation.
//
// A one-level check is NOT enough: by this pass both a constant weight and a
// runtime-activation filter arrive as a linalg.transpose (transpose(const) vs
// transpose(activation)), so we must look THROUGH the value-preserving
// layout/cast ops to the root to tell them apart. The real chain is 1-2 ops
// (a transpose, sometimes an i8->i16 cast generic); kMaxDepth keeps the walk
// short and bounds it so a long/cyclic def chain can never spin.
static bool tracesToConstantWeight(Value v) {
    constexpr int kMaxDepth = 4;
    for (int i = 0; i < kMaxDepth && v; ++i) {
        Operation *def = v.getDefiningOp();
        if (!def)
            return false; // block argument / runtime input -> activation
        if (isa<arith::ConstantOp>(def))
            return true;
        // Look through pure layout/cast ops that just reshape or retype weights.
        if (isa<linalg::TransposeOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp>(def)) {
            v = def->getOperand(0);
            continue;
        }
        if (auto g = dyn_cast<linalg::GenericOp>(def)) {
            if (g.getNumDpsInputs() >= 1) {
                v = g.getDpsInputOperand(0)->get();
                continue;
            }
        }
        return false;
    }
    // Hit the depth bound without resolving -> treat as non-constant (activation)
    // so we don't raise something we couldn't prove is a weight.
    return false;
}

// Raise a 1x1 linalg.conv_2d_nhwc_hwcf (matmul-as-conv) into a linalg.matmul
// before the NHWC->NCHW pass runs, but ONLY when both conv inputs are runtime
// activations (e.g. a TFLite FULLY_CONNECTED with two activation inputs, as in
// attention). For a normal conv with a constant weight the existing conv path
// already handles it correctly, so we leave those untouched.
//
// The NHWC->NCHW pass cannot correctly relayout the quantized epilogue of an
// activation x activation matmul-as-conv; raising to matmul avoids the layout
// boundary and lets the matmul lowering handle the quant params.
//   input  [N,1,1,C]  -> collapse -> [N,C]
//   filter [1,1,C,F]  -> collapse -> [C,F]
//   matmul [N,C]x[C,F] -> [N,F]   -> expand -> [N,1,1,F]
struct Conv2D1x1NhwcHwcfToMatmulPattern : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp, PatternRewriter &rewriter) const override {
        auto loc = convOp.getLoc();

        Value input = convOp.getInputs()[0];   // [N, H, W, C]
        Value filter = convOp.getInputs()[1];  // [Kh, Kw, C, F]
        Value output = convOp.getOutputs()[0]; // [N, Ho, Wo, F]

        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        ArrayRef<int64_t> inputShape = inputType.getShape();
        ArrayRef<int64_t> filterShape = filterType.getShape();

        // Only handle the 1x1 pointwise kernel (matmul-as-conv).
        if (filterShape[0] != 1 || filterShape[1] != 1) {
            return rewriter.notifyMatchFailure(convOp, "not a 1x1 kernel");
        }

        // Only raise the activation x activation case. A constant-weight conv is
        // handled correctly by the existing conv path; raising it regresses the
        // full model (downstream rank mismatch).
        if (tracesToConstantWeight(filter)) {
            return rewriter.notifyMatchFailure(convOp, "constant weight; keep as conv");
        }

        // Unit stride/dilation only — a 1x1 stride-1 conv is a pure matmul.
        SmallVector<int64_t> strides = llvm::to_vector<4>(
            llvm::map_range(convOp.getStrides(), [](APInt v) { return v.getSExtValue(); })
        );
        SmallVector<int64_t> dilations = llvm::to_vector<4>(
            llvm::map_range(convOp.getDilations(), [](APInt v) { return v.getSExtValue(); })
        );
        if (strides[0] != 1 || strides[1] != 1 || dilations[0] != 1 || dilations[1] != 1) {
            return rewriter.notifyMatchFailure(convOp, "non-unit stride/dilation");
        }

        // The conv accumulator must start at zero so the matmul reproduces it.
        if (!isAllZerosTensor(output)) {
            return rewriter.notifyMatchFailure(convOp, "conv init is not all zeros");
        }

        int64_t N = inputShape[0];
        int64_t C = inputShape[3];
        int64_t F = filterShape[3];
        auto inElemType = inputType.getElementType();
        auto outElemType = outputType.getElementType();

        // input [N,1,1,C] -> [N,C]
        auto collapsedInputType = RankedTensorType::get({N, C}, inElemType);
        auto collapsedInput = tensor::CollapseShapeOp::create(
            rewriter, loc, collapsedInputType, input, ArrayRef<ReassociationIndices>{{0, 1, 2}, {3}}
        );

        // filter [1,1,C,F] -> [C,F] (no transpose: matmul contracts [N,C]x[C,F]).
        auto collapsedFilterType = RankedTensorType::get({C, F}, filterType.getElementType());
        auto collapsedFilter = tensor::CollapseShapeOp::create(
            rewriter, loc, collapsedFilterType, filter,
            ArrayRef<ReassociationIndices>{{0, 1, 2}, {3}}
        );

        // [N,C] x [C,F] -> [N,F], accumulating into the conv output element type.
        SmallVector<int64_t> matmulShape = {N, F};
        auto matmulType = RankedTensorType::get(matmulShape, outElemType);
        auto matmulInit = linalg::FillOp::create(
            rewriter, loc, createZeroConstant(rewriter, loc, outElemType),
            tensor::EmptyOp::create(rewriter, loc, matmulShape, outElemType).getResult()
        );
        auto matmulOp = linalg::MatmulOp::create(
            rewriter, loc, TypeRange{matmulType},
            ValueRange{collapsedInput.getResult(), collapsedFilter.getResult()},
            ValueRange{matmulInit.getResult(0)}
        );

        // [N,F] -> [N,1,1,F]
        auto expandedResult = tensor::ExpandShapeOp::create(
            rewriter, loc, outputType, matmulOp.getResult(0),
            ArrayRef<ReassociationIndices>{{0, 1, 2}, {3}}
        );

        rewriter.replaceOp(convOp, expandedResult.getResult());
        return success();
    }
};

struct Conv1DNcwFcwToLinalgConv2DPattern : public OpRewritePattern<linalg::Conv1DNcwFcwOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv1DNcwFcwOp convOp, PatternRewriter &rewriter) const override {
        auto loc = convOp.getLoc();

        // Get operands
        Value input = convOp.getInputs()[0];
        Value filter = convOp.getInputs()[1];
        Value output = convOp.getOutputs()[0];

        // Get types and shapes
        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        // Add extra dimension (1) to input as W: [N,C,W] -> [N,C,W,1]
        // Need to use proper reassociation indices
        SmallVector<ReassociationIndices> inputReassoc = {{0}, {1}, {2, 3}};
        auto expandedInputType = RankedTensorType::get(
            {inputType.getShape()[0], inputType.getShape()[1], inputType.getShape()[2], 1},
            inputType.getElementType()
        );

        auto expandedInput =
            tensor::ExpandShapeOp::create(rewriter, loc, expandedInputType, input, inputReassoc);

        // Add extra dimension (1) to filter: [F,C,W] -> [F,C,W,1]
        SmallVector<ReassociationIndices> filterReassoc = {{0}, {1}, {2, 3}};
        auto expandedFilterType = RankedTensorType::get(
            {filterType.getShape()[0], filterType.getShape()[1], filterType.getShape()[2], 1},
            filterType.getElementType()
        );

        auto expandedFilter =
            tensor::ExpandShapeOp::create(rewriter, loc, expandedFilterType, filter, filterReassoc);

        // Add extra dimension (1) to output: [N,F,W] -> [N,F,W,1]
        SmallVector<ReassociationIndices> outputReassoc = {{0}, {1}, {2, 3}};
        auto expandedOutputType = RankedTensorType::get(
            {outputType.getShape()[0], outputType.getShape()[1], outputType.getShape()[2], 1},
            outputType.getElementType()
        );

        auto expandedOutput =
            tensor::ExpandShapeOp::create(rewriter, loc, expandedOutputType, output, outputReassoc);

        // Get attributes
        auto stridesAttr = convOp.getStrides();
        auto dilationsAttr = convOp.getDilations();

        // Convert 1D strides/dilations to 2D (add 1 as width dimension)
        SmallVector<int64_t> strides2d = {stridesAttr.getValues<int64_t>()[0]};
        strides2d.push_back(1);
        SmallVector<int64_t> dilations2d = {dilationsAttr.getValues<int64_t>()[0]};
        dilations2d.push_back(1);

        auto attrType = RankedTensorType::get({2}, rewriter.getIntegerType(64));
        auto stridesAttr2d = DenseIntElementsAttr::get(attrType, strides2d);
        auto dilationsAttr2d = DenseIntElementsAttr::get(attrType, dilations2d);

        auto conv2d = linalg::Conv2DNchwFchwOp::create(
            rewriter, loc, expandedOutput.getType(), ValueRange{expandedInput, expandedFilter},
            ValueRange{expandedOutput}, stridesAttr2d, dilationsAttr2d
        );

        // check if there is linalg.genericOp does truncf for the conv1d result, if so, we need to
        // create a new linalg.genericOp to do the truncf for new conv2d result, and remove the
        // old one

        auto conv2dResult = conv2d.getResult(0);
        auto currentResult = conv2dResult;

        linalg::GenericOp truncfGenericOp = nullptr;

        auto conv1dResult = convOp.getResult(0);

        // if convOp has only one use
        if (conv1dResult.hasOneUse()) {
            auto userOp = conv1dResult.use_begin()->getOwner();
            if (auto genericOp = llvm::dyn_cast<linalg::GenericOp>(userOp)) {

                if (genericOp.getNumResults() == 1 &&
                    dyn_cast<RankedTensorType>(genericOp.getResult(0).getType())
                        .getElementType()
                        .isBF16() &&
                    dyn_cast<RankedTensorType>(conv1dResult.getType()).getElementType().isF32()) {

                    // Re-emit the conv1d's f32->bf16 truncf on the new conv2d result; the
                    // original generic is dropped below once the collapse is wired up.
                    currentResult = emitTruncfToBF16(rewriter, loc, conv2dResult);
                    truncfGenericOp = genericOp;
                }
            }
        }

        // Collapsing the width dimension ([N,F,Ow,1] -> [N,F,Ow])
        // Create a new reassociation indices for collapsing the last dimension
        SmallVector<ReassociationIndices> collapseReassoc = {{0}, {1}, {2, 3}};
        auto collapsedResultType = RankedTensorType::get(
            {outputType.getShape()[0], outputType.getShape()[1], outputType.getShape()[2]},
            cast<RankedTensorType>(currentResult.getType()).getElementType()
        );

        auto collapsedResult = tensor::CollapseShapeOp::create(
            rewriter, loc, collapsedResultType, currentResult, collapseReassoc
        );

        rewriter.replaceOp(convOp, collapsedResult.getResult());

        if (truncfGenericOp) {
            currentResult = collapsedResult.getResult();
            truncfGenericOp->replaceAllUsesWith(ValueRange{currentResult});
            rewriter.eraseOp(truncfGenericOp);
        }

        return success();
    }
};

/// Lowers linalg.conv_1d_ncw_fcw to a native torq_hl.conv1d via a 6D linalg.generic.
///
/// Shapes: input [N,C,W] -> [N,C,1,W], filter [F,C,Kw] -> [F,C,1,Kw]. The generic computes the
/// per-(n,f,ow,kw) products in f32 with c as the reduction, producing [N,F,1,Ow,Kw]; a kw reduce
/// then sums to [N,F,1,Ow] (kept f32 for accurate accumulation) and collapses to [N,F,Ow]. The
/// preserved Kw lets LinalgGenericConv1DToTorqHLConv1DPattern match torq_hl.conv1d downstream.
///
/// truncf: an f32 output (or a following bf16 truncf generic) gets a final f32->bf16 cast. With
/// --torq-conv1d-truncate-for-reduce the cast is moved before the reduce to save bandwidth at a
/// small precision cost; the default keeps the reduce input in f32.
///
/// When the full [N,F,1,Ow,Kw] generic exceeds LRAM the pattern tiles along Ow (see
/// emitTiledNativeConv1d), matching the im2col path's strategy.
struct Conv1DNcwFcwToGenericConv1DPattern : public OpRewritePattern<linalg::Conv1DNcwFcwOp> {
    // When `onlyNonPointwise` is set the pattern skips pointwise convs (Kw == 1),
    // which lower more efficiently through the matmul -> fully_connected path.
    Conv1DNcwFcwToGenericConv1DPattern(
        MLIRContext *context, PatternBenefit benefit = 1, bool onlyNonPointwise = false
    )
        : OpRewritePattern(context, benefit), onlyNonPointwise(onlyNonPointwise) {}

    LogicalResult
    matchAndRewrite(linalg::Conv1DNcwFcwOp convOp, PatternRewriter &rewriter) const override {
        auto loc = convOp.getLoc();

        // Get operands
        Value input = convOp.getInputs()[0];   // [N, C, W]
        Value filter = convOp.getInputs()[1];  // [F, C, Kw]
        Value output = convOp.getOutputs()[0]; // [N, F, Ow]

        // Get types and shapes
        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        // The native torq_hl.conv1d HW lowering (transformWithReduce) only supports bf16
        // data and weights. Routing any other element type (e.g. f32) to the native path
        // would hit an unsupported-dtype error in the HW lowering, so bail here and let the
        // im2col/matmul path handle (or cleanly reject) it instead.
        if (!inputType.getElementType().isBF16() || !filterType.getElementType().isBF16()) {
            return rewriter.notifyMatchFailure(
                convOp, "native conv1d requires bf16 data and weights"
            );
        }

        ArrayRef<int64_t> inputShape = inputType.getShape();
        ArrayRef<int64_t> filterShape = filterType.getShape();
        ArrayRef<int64_t> outputShape = outputType.getShape();

        int64_t N = inputShape[0];
        int64_t C = inputShape[1];
        int64_t F = filterShape[0];
        int64_t Kw = filterShape[2];
        int64_t Ow = outputShape[2];

        // Pointwise convs lower to a fully_connected that runs efficiently on the
        // NPU, so leave them to the matmul path when only non-pointwise convs were
        // requested (see populateOptimizeConv1DPatterns).
        if (onlyNonPointwise && Kw == 1) {
            return rewriter.notifyMatchFailure(convOp, "pointwise conv handled by matmul path");
        }

        int64_t W = inputShape[2];
        auto elemType = inputType.getElementType();
        auto filterElemType = filterType.getElementType();
        Type computeElemType = rewriter.getF32Type();

        SmallVector<int64_t> strides = llvm::to_vector<4>(
            llvm::map_range(convOp.getStrides(), [](APInt v) { return v.getSExtValue(); })
        );
        SmallVector<int64_t> dilations = llvm::to_vector<4>(
            llvm::map_range(convOp.getDilations(), [](APInt v) { return v.getSExtValue(); })
        );
        int64_t stride = strides[0];
        int64_t dilation = dilations[0];

        int64_t blockOw = computeGenericConv1dBlockOw(
            N, C, F, Ow, Kw, stride, dilation, elemType, filterElemType, computeElemType
        );
        // A single-channel conv lowers to an all-parallel generic that downstream tile-and-fuse
        // can tile at any size, so emit it untiled and skip Ow tiling: for a large Ow/Kw the
        // per-block budget would otherwise force a tiny blockOw and explode into thousands of
        // conv1d blocks, stalling compilation. A multi-channel conv needs the C reduction, whose
        // untiled intermediate cannot be tiled downstream, so it keeps the LRAM-aware Ow tiling
        // (the matmul path can't take over either: its C*Kw contraction overflows the HW
        // accumulate block).
        if (C == 1) {
            blockOw = Ow;
        }
        else if (blockOw == 0) {
            return rewriter.notifyMatchFailure(
                convOp, "conv too large for native conv1d; use im2col/matmul path"
            );
        }

        auto outputElemType = outputType.getElementType();

        // Detect if there's a truncf generic following the conv1d.
        auto convResult = convOp.getResult(0);
        linalg::GenericOp truncfGenericOp = nullptr;
        if (convResult.hasOneUse()) {
            auto userOp = convResult.use_begin()->getOwner();
            if (auto truncfOp = llvm::dyn_cast<linalg::GenericOp>(userOp)) {
                if (truncfOp.getNumResults() == 1 &&
                    cast<RankedTensorType>(truncfOp.getResult(0).getType())
                        .getElementType()
                        .isBF16() &&
                    cast<RankedTensorType>(convResult.getType()).getElementType().isF32()) {
                    truncfGenericOp = truncfOp;
                }
            }
        }
        bool needsBF16Output = outputElemType.isBF16() || truncfGenericOp;

        // Expand filter from [F, C, Kw] to [F, C, 1, Kw] (add height dimension)
        auto filter4DType = RankedTensorType::get({F, C, 1, Kw}, filterElemType);
        SmallVector<ReassociationIndices> filterReassoc = {{0}, {1}, {2, 3}};
        Value filter4D =
            tensor::ExpandShapeOp::create(rewriter, loc, filter4DType, filter, filterReassoc);

        SmallVector<ReassociationIndices> inputReassoc = {{0}, {1}, {2, 3}};
        Value reduced4D;

        if (blockOw >= Ow) {
            // Single LRAM-resident generic + kw reduce.
            auto input4DType = RankedTensorType::get({N, C, 1, W}, elemType);
            Value input4D =
                tensor::ExpandShapeOp::create(rewriter, loc, input4DType, input, inputReassoc);

            Value genericResult = emitGenericConv1dBlock(
                rewriter, loc, input4D, filter4D, N, F, Ow, Kw, stride, dilation
            );

            Value reduceInput = genericResult;
            // Memory-optimized: truncate the [N,F,1,Ow,Kw] generic to bf16 before the reduce.
            if (clConv1dTruncateForReduce && truncfGenericOp) {
                reduceInput = emitTruncfToBF16(rewriter, loc, genericResult);
            }
            if (truncfGenericOp) {
                dropTruncf(rewriter, truncfGenericOp);
            }

            reduced4D = emitKwReduceBlock(rewriter, loc, reduceInput, N, F, Ow);
        }
        else {
            // Ow-tiled native conv1d: each block runs generic + kw reduce on a sliced input window.
            if (truncfGenericOp) {
                dropTruncf(rewriter, truncfGenericOp);
            }
            reduced4D = emitTiledNativeConv1d(
                rewriter, loc, input, filter4D, N, C, F, Ow, Kw, stride, dilation, blockOw, elemType
            );
        }

        // Collapse [N, F, 1, Ow] -> [N, F, Ow]
        Type reduceOutputType = rewriter.getF32Type();
        SmallVector<int64_t> collapsedShape = {N, F, Ow};
        auto collapsedType = RankedTensorType::get(collapsedShape, reduceOutputType);
        SmallVector<ReassociationIndices> collapseReassoc = {{0}, {1}, {2, 3}};
        Value collapsedResult = tensor::CollapseShapeOp::create(
                                    rewriter, loc, collapsedType, reduced4D, collapseReassoc
        )
                                    .getResult();

        Value finalResult = collapsedResult;
        if (needsBF16Output) {
            finalResult = emitTruncfToBF16(rewriter, loc, collapsedResult);
        }

        rewriter.replaceOp(convOp, finalResult);
        return success();
    }

    bool onlyNonPointwise;
};

// Find the collapse_shape that assembles the native conv1d reduce output into [N,F,Ow].
// Untiled: the reduce result feeds the collapse directly. Ow-tiled (emitTiledNativeConv1d):
// each block reduce feeds an insert_slice tree whose root is collapsed, so walk the chain
// forward to the collapse. Returns null if the chain isn't this conv1d reduce->collapse shape.
static tensor::CollapseShapeOp findConvReduceCollapse(Value reduceResult) {
    Operation *user = getSingleUser(reduceResult);
    if (!user) {
        return nullptr;
    }
    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(user)) {
        return collapse;
    }
    auto insert = dyn_cast<tensor::InsertSliceOp>(user);
    if (!insert || insert.getSource() != reduceResult) {
        return nullptr;
    }
    for (Value cur = insert.getResult(); Operation *next = getSingleUser(cur);) {
        if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(next)) {
            return collapse;
        }
        auto nextInsert = dyn_cast<tensor::InsertSliceOp>(next);
        if (!nextInsert || nextInsert.getDest() != cur) {
            return nullptr;
        }
        cur = nextInsert.getResult();
    }
    return nullptr;
}

// Collect every block reduce feeding the assembled tensor that `collapse` consumes: one
// reduce for the untiled path, one per Ow block (via the insert_slice tree) for the tiled path.
static void collectConvBlockReduces(
    tensor::CollapseShapeOp collapse, SmallVectorImpl<linalg::ReduceOp> &reduces
) {
    Value cur = collapse.getSrc();
    while (auto insert = cur.getDefiningOp<tensor::InsertSliceOp>()) {
        if (auto red = insert.getSource().getDefiningOp<linalg::ReduceOp>()) {
            reduces.push_back(red);
        }
        cur = insert.getDest();
    }
    if (auto red = cur.getDefiningOp<linalg::ReduceOp>()) {
        reduces.push_back(red);
    }
}

/// Fold a per-channel bias add that follows the conv1d reduce_sum into the
/// reduce itself. The native conv1d path lowers to:
///   conv1d(mul) -> reduce_sum(f32) -> collapse -> addf(bias) -> truncf(bf16)
/// and, when the [N,F,1,Ow,Kw] generic exceeds LRAM, to a per-Ow-block reduce assembled
/// by an insert_slice tree before the collapse (emitTiledNativeConv1d).
/// The fp32 bias add has no NSS lowering, so when Host fallback is disabled it
/// cannot run anywhere. Attaching the per-channel bias to the reduce(s) lets the
/// reduce's activation stage apply it in fp32 (see ReduceOpConversion and the
/// reduce HW kernel), which is bit-exact with `round_bf16(sum_f32 + bias_f32)`.
struct FoldConvBiasIntoReducePattern : public OpRewritePattern<linalg::ReduceOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::ReduceOp reduceOp, PatternRewriter &rewriter) const override {
        if (reduceOp->hasAttr(kReducePerChannelBiasAttr) || reduceOp.getNumResults() != 1) {
            return failure();
        }

        // Only an fp32 reduce result can carry an fp32 per-channel bias.
        auto reduceResult = reduceOp.getResult(0);
        auto reduceType = dyn_cast<RankedTensorType>(reduceResult.getType());
        if (!reduceType || !reduceType.getElementType().isF32()) {
            return failure();
        }

        // reduce -> [insert_slice tree ->] collapse_shape -> per-channel bias add (+ truncf)
        auto collapseOp = findConvReduceCollapse(reduceResult);
        if (!collapseOp) {
            return failure();
        }
        Value collapseResult = collapseOp.getResult();
        auto collapseType = dyn_cast<RankedTensorType>(collapseResult.getType());

        // Channel dimension is dim 1 (NCHW) for the conv1d generic path.
        constexpr int channelDim = 1;
        if (!collapseType || collapseType.getRank() <= channelDim) {
            return failure();
        }
        const int64_t numChannels = collapseType.getDimSize(channelDim);
        if (numChannels <= 0) {
            return failure();
        }

        VectorIntOrFloat biasVec(numChannels, /*isInt=*/false);
        Value folded = collapseResult;
        if (!foldForwardPerChannelAdd(folded, channelDim, biasVec)) {
            return failure();
        }
        Operation *addOp = folded.getDefiningOp();
        if (!addOp) {
            return failure();
        }

        // The Ow-tiled path assembles several block reduces before the collapse; the same
        // per-channel bias applies to each (the blocks partition the output width).
        SmallVector<linalg::ReduceOp> blockReduces;
        collectConvBlockReduces(collapseOp, blockReduces);
        if (blockReduces.empty()) {
            return failure();
        }

        auto biasType = RankedTensorType::get({numChannels}, rewriter.getF32Type());
        auto biasAttr = DenseFPElementsAttr::get(biasType, biasVec.floats);
        for (linalg::ReduceOp red : blockReduces) {
            rewriter.modifyOpInPlace(red, [&]() {
                red->setAttr(kReducePerChannelBiasAttr, biasAttr);
            });
        }
        // Drop the folded bias add, rerouting its users back to the collapse result.
        rewriter.replaceOp(addOp, collapseResult);
        return success();
    }
};

void populateOptimizeConv1DPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    // Raise 1x1 matmul-as-conv to linalg.matmul before the NHWC->NCHW pass.
    patterns.insert<Conv2D1x1NhwcHwcfToMatmulPattern>(context);

    // We have multiple ways of lowering conv1d.
    // If none explicitly enabled, default to the matmul path.
    // Later we can enable multiple paths and pick the best one for each conv1d
    if (!clConv1dAsMatmul && !clConv1dToGenericConv1D && !clConv1dToConv2D) {
        // Automatic selection of the best lowering path for conv1d
        // Always use the matmul path for now
        clConv1dAsMatmul = true;
    }

    if (clConv1dAsMatmul) {
        patterns.insert<Conv1DNcwFcwToLinalgMatmulPattern>(context);
    }
    if (clConv1dToGenericConv1D) {
        patterns.insert<Conv1DNcwFcwToGenericConv1DPattern>(context);
    }
    if (clConv1dToConv2D) {
        patterns.insert<Conv1DNcwFcwToLinalgConv2DPattern>(context);
    }

    // A non-pointwise (Kw > 1) conv1d that the matmul/im2col path cannot lower on-NPU --
    // e.g. a biased conv (the matmul pattern bails on the non-zero init) or a wide kernel
    // whose C*Kw reduction overflows the matmul block -- must still compile under NSS-only.
    // Route those convs through the native generic path at a higher benefit so it wins over
    // the matmul pattern above, and fold the per-channel bias into the reduce, avoiding the
    // fp32 host bias-add. Pointwise convs already lower to an efficient fully_connected, so
    // they keep the path selected above. The native path only matches bf16 data/weights (see
    // Conv1DNcwFcwToGenericConv1DPattern); other dtypes fall back to the matmul/im2col path.
    patterns.insert<Conv1DNcwFcwToGenericConv1DPattern>(
        context, /*benefit=*/2, /*onlyNonPointwise=*/true
    );
    patterns.insert<FoldConvBiasIntoReducePattern>(context);
}

} // namespace mlir::syna::torq
