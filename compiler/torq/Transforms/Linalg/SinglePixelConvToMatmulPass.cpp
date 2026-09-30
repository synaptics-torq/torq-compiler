// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::syna::torq {

namespace {

static bool isZeroFill(Value value) {
    auto fill = value.getDefiningOp<linalg::FillOp>();
    if (!fill)
        return false;
    Value scalar = fill.getInputs()[0];
    return matchPattern(scalar, m_Zero()) || matchPattern(scalar, m_AnyZeroFloat());
}

// Returns a zero-filled tensor of `type`.
static Value zeros(PatternRewriter &rewriter, Location loc, RankedTensorType type) {
    Value zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(type.getElementType()));
    Value empty = tensor::EmptyOp::create(rewriter, loc, type.getShape(), type.getElementType());
    return linalg::FillOp::create(rewriter, loc, ValueRange{zero}, ValueRange{empty}).getResult(0);
}

// Returns lhs + rhs for two tensors of the same shape.
static Value addTensors(PatternRewriter &rewriter, Location loc, Value lhs, Value rhs) {
    auto type = cast<RankedTensorType>(lhs.getType());
    Value empty = tensor::EmptyOp::create(rewriter, loc, type.getShape(), type.getElementType());
    AffineMap identity = rewriter.getMultiDimIdentityMap(type.getRank());
    SmallVector<utils::IteratorType> iterators(type.getRank(), utils::IteratorType::parallel);
    return linalg::GenericOp::create(
               rewriter, loc, TypeRange{type}, ValueRange{lhs, rhs}, ValueRange{empty},
               ArrayRef<AffineMap>{identity, identity, identity}, iterators,
               [](OpBuilder &b, Location loc, ValueRange args) {
                   Value sum = isa<IntegerType>(args[0].getType())
                                   ? arith::AddIOp::create(b, loc, args[0], args[1]).getResult()
                                   : arith::AddFOp::create(b, loc, args[0], args[1]).getResult();
                   linalg::YieldOp::create(b, loc, sum);
               }
    ).getResult(0);
}

// Returns `init` [N, 1, 1, F] as [N, F]. A broadcast of a [F] or [1] bias is
// rebuilt as a 2-D broadcast rather than collapsed, so the bias/rescale chain the
// fully-connected lowering fuses is 2-D throughout and its channel dim is found.
static Value collapseInit(PatternRewriter &rewriter, Location loc, Value init) {
    auto initType = cast<RankedTensorType>(init.getType());
    auto resultType = RankedTensorType::get(
        {initType.getDimSize(0), initType.getDimSize(3)}, initType.getElementType()
    );

    auto broadcast = init.getDefiningOp<linalg::GenericOp>();
    if (broadcast && broadcast.getNumDpsInputs() == 1 && broadcast.getNumDpsInits() == 1 &&
        broadcast.getNumParallelLoops() == broadcast.getNumLoops() &&
        broadcast.getIndexingMapsArray()[1].isIdentity()) {
        auto yield = cast<linalg::YieldOp>(broadcast.getBody()->getTerminator());
        Value source = broadcast.getDpsInputs()[0];
        auto sourceType = dyn_cast<RankedTensorType>(source.getType());
        AffineMap sourceMap = broadcast.getIndexingMapsArray()[0];
        MLIRContext *ctx = rewriter.getContext();
        std::optional<AffineExpr> channel;
        if (sourceType && sourceType.getRank() == 1 && sourceMap.getNumResults() == 1 &&
            yield.getValues()[0] == broadcast.getBody()->getArgument(0)) {
            if (sourceMap.getResult(0) == getAffineDimExpr(3, ctx))
                channel = getAffineDimExpr(1, ctx);
            else if (sourceMap.getResult(0) == getAffineConstantExpr(0, ctx))
                channel = getAffineConstantExpr(0, ctx);
        }
        if (channel) {
            Value empty = tensor::EmptyOp::create(
                rewriter, loc, resultType.getShape(), resultType.getElementType()
            );
            AffineMap sourceMap2d = AffineMap::get(2, 0, {*channel}, ctx);
            return linalg::GenericOp::create(
                       rewriter, loc, TypeRange{resultType}, ValueRange{source}, ValueRange{empty},
                       ArrayRef<AffineMap>{sourceMap2d, rewriter.getMultiDimIdentityMap(2)},
                       SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
                       [](OpBuilder &b, Location loc, ValueRange args) {
                           linalg::YieldOp::create(b, loc, args[0]);
                       }
            ).getResult(0);
        }
    }
    return tensor::CollapseShapeOp::create(
        rewriter, loc, resultType, init, SmallVector<ReassociationIndices>{{0, 1, 2}, {3}}
    );
}

// Rewrites a 1x1 convolution over a single pixel,
//   input [N, 1, 1, C] * filter [1, 1, C, F] -> [N, 1, 1, F],
// as a matmul [N, C] x [C, F]. The TOSA import of TFLite FULLY_CONNECTED
// produces exactly this, and as a convolution it runs on the conv kernel, which
// is far slower than the fully-connected kernel for a single pixel.
template <typename ConvOp, typename MatmulOp>
struct SinglePixelConvToMatmul : public OpRewritePattern<ConvOp> {
    using OpRewritePattern<ConvOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ConvOp conv, PatternRewriter &rewriter) const override {
        if (!conv.hasPureTensorSemantics())
            return rewriter.notifyMatchFailure(conv, "expects tensor semantics");

        Value input = conv.getInputs()[0];
        Value filter = conv.getInputs()[1];
        Value init = conv.getOutputs()[0];
        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto initType = cast<RankedTensorType>(init.getType());
        if (!inputType.hasStaticShape() || !filterType.hasStaticShape() ||
            !initType.hasStaticShape())
            return rewriter.notifyMatchFailure(conv, "expects static shapes");

        // NHWC input and HWCF filter: one pixel and a 1x1 kernel
        ArrayRef<int64_t> in = inputType.getShape(), f = filterType.getShape();
        if (in[1] != 1 || in[2] != 1 || f[0] != 1 || f[1] != 1)
            return rewriter.notifyMatchFailure(conv, "not a 1x1 convolution over one pixel");
        // Only FULLY_CONNECTED, whose input is its [N, C] operand expanded to one pixel,
        // or a graph input. A single-pixel conv inside a 4-D graph (e.g. a squeeze-excite
        // after global pooling) stays a conv: 2-D ops there break the NHWC layout passes
        // that follow.
        auto inputExpand = input.getDefiningOp<tensor::ExpandShapeOp>();
        if (!isa<BlockArgument>(input) && (!inputExpand || inputExpand.getSrcType().getRank() != 2))
            return rewriter.notifyMatchFailure(conv, "input is not an [N, C] operand");
        ArrayRef<int64_t> out = initType.getShape();
        if (out[0] != in[0] || out[1] != 1 || out[2] != 1 || out[3] != f[3])
            return rewriter.notifyMatchFailure(conv, "expects a [N, 1, 1, F] result");

        Location loc = conv.getLoc();
        SmallVector<ReassociationIndices> reassociation = {{0, 1, 2}, {3}};
        auto collapse = [&](Value value) {
            return tensor::CollapseShapeOp::create(rewriter, loc, value, reassociation);
        };

        // The remaining operands (the zero points of the quantized form) keep their order.
        SmallVector<Value> matmulInputs = {collapse(input), collapse(filter)};
        llvm::append_range(matmulInputs, conv.getInputs().drop_front(2));

        // Accumulate into zeros and add the original init (usually the bias)
        // afterwards: the fully-connected lowering only takes a zero-initialized matmul.
        auto resultType = RankedTensorType::get({in[0], f[3]}, initType.getElementType());
        Value result = MatmulOp::create(
                           rewriter, loc, TypeRange{resultType}, matmulInputs,
                           ValueRange{zeros(rewriter, loc, resultType)}
        )
                           .getResult(0);
        if (!isZeroFill(init))
            result = addTensors(rewriter, loc, result, collapseInit(rewriter, loc, init));

        rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(conv, initType, result, reassociation);
        return success();
    }
};

// True for an operand produced by an expand_shape [N, F] -> [N, 1, 1, F]
static bool isSinglePixelExpand(OpOperand *operand) {
    auto expand = operand->get().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expand || expand.getSrcType().getRank() != 2)
        return false;
    ArrayRef<int64_t> shape = expand.getResultType().getShape();
    return shape.size() == 4 && shape[1] == 1 && shape[2] == 1;
}

class SinglePixelConvToMatmulPass
    : public impl::SinglePixelConvToMatmulBase<SinglePixelConvToMatmulPass> {
  public:
    using SinglePixelConvToMatmulBase::SinglePixelConvToMatmulBase;

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<
            SinglePixelConvToMatmul<linalg::Conv2DNhwcHwcfOp, linalg::MatmulOp>,
            SinglePixelConvToMatmul<linalg::Conv2DNhwcHwcfQOp, linalg::QuantizedMatmulOp>>(
            &getContext()
        );
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            return signalPassFailure();

        // Push the [N, F] -> [N, 1, 1, F] expand_shape of each rewrite through the
        // elementwise ops that follow (bias, requantization), so the chain the
        // fully-connected lowering fuses stays 2-D and the expand cancels against
        // the reshape back to [N, F] that TFLite FULLY_CONNECTED ends with.
        RewritePatternSet foldPatterns(&getContext());
        linalg::populateFoldReshapeOpsByCollapsingPatterns(foldPatterns, isSinglePixelExpand);
        tensor::CollapseShapeOp::getCanonicalizationPatterns(foldPatterns, &getContext());
        tensor::ExpandShapeOp::getCanonicalizationPatterns(foldPatterns, &getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(foldPatterns))))
            return signalPassFailure();
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createSinglePixelConvToMatmulPass() {
    return std::make_unique<SinglePixelConvToMatmulPass>();
}

} // namespace mlir::syna::torq
