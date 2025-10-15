// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Matchers.h"
#include "torq/Utils/TorqUtils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::syna::torq_hl;

#define DEBUG_TYPE "torqhl-ops"

namespace mlir::syna::torq_hl {

/// Pattern to fuse a load/store op with MemRefCast arguments.
///
/// Example:
/// ```
///   %s = memref.subview %2[...] : memref<1x3x64x64xi8> to memref<1x3x33x64xi8, offset: 2048>>
///   %c = memref.cast %s : memref<1x3x33x64xi8, offset: 2048>> to memref<1x3x33x64xi8, offset: ?>>
///  "torq_hl.load"(%12, %c) : (memref<1x3x33x64xi8>>, memref<1x3x33x64xi8, offset: ?>>) -> ()
/// ```
/// is rewritten into:
/// ```
///   %s = memref.subview %2[...] : memref<1x3x64x64xi8> to memref<1x3x33x64xi8, offset: 2048>>
///   "torq_hl.load"(%12, %s) : (memref<1x3x33x64xi8>>, memref<1x3x33x64xi8, offset: 2048>>) -> ()
/// ```
/// This condition arises quite commonly when the load/store operation comes from unrolled loops.
/// Idea of this canonicalization derived from SubViewOpMemRefCastFolder

template <class Op, int operandIndex = -1> class OpMemRefCastFolder : public OpRewritePattern<Op> {
  public:
    using OpRewritePattern<Op>::OpRewritePattern;

    LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const override {
        if (operandIndex == -1) {
            bool folded = false;
            // Iterate over all operands and try to fold memref.cast ops into the consumer op
            for (auto [idx, maybeCastOp] : llvm::enumerate(op.getOperands())) {
                if (auto castOp = maybeCastOp.template getDefiningOp<memref::CastOp>()) {
                    if (memref::CastOp::canFoldIntoConsumerOp(castOp)) {
                        auto canonicalType = llvm::cast<MemRefType>(castOp.getSource().getType());
                        op->getOperand(idx).setType(canonicalType);
                        folded = true;
                    }
                }
            }
            return folded ? success() : failure();
        }
        else {
            auto castOp = op->getOperand(operandIndex).template getDefiningOp<memref::CastOp>();
            if (!castOp) {
                return failure();
            }
            if (!memref::CastOp::canFoldIntoConsumerOp(castOp)) {
                return failure();
            }

            // Fuse the cast operation into the current operation.
            auto canonicalType = llvm::cast<MemRefType>(castOp.getSource().getType());
            op.getOperand(operandIndex).setType(canonicalType);

            return success();
        }
    }
};

void LoadOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
    results.add<OpMemRefCastFolder<LoadOp, 0>>(context);
    results.add<OpMemRefCastFolder<LoadOp, 1>>(context);
}

void StoreOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
    results.add<OpMemRefCastFolder<StoreOp, 0>>(context);
    results.add<OpMemRefCastFolder<StoreOp, 1>>(context);
}

// Removes any host copy from itself to itself
class FoldIdentityHostCopy : public OpRewritePattern<HostCopyOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(HostCopyOp op, PatternRewriter &rewriter) const override {

        if (op.getInput() != op.getOutput()) {
            return failure();
        }

        rewriter.eraseOp(op);

        return success();
    }
};

void HostCopyOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
    results.add<OpMemRefCastFolder<HostCopyOp, 0>>(context);
    results.add<OpMemRefCastFolder<HostCopyOp, 1>>(context);
    results.add<FoldIdentityHostCopy>(context);
}

//
// This pattern looks for torq_hl.program operations that have arguments with
// either dynamic offset or dynamic strides in their memref layout.
// If such a program is found, and if the program has exactly one use,
// we look for the corresponding start_program operation and try to infer
// the offsets and strides from the operands passed to the start_program operation.
// If we can infer the offsets and strides, we update the argument types
// in the program operation.
//
class InferProgramMemRefLayout : public OpRewritePattern<ProgramOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    // true if this type is a memref with a strided layout that has
    // either dynamic strides or a dynamic offset
    bool isMemRefWithDynamicLayout(Type type) const {
        auto memrefType = dyn_cast<MemRefType>(type);

        if (!memrefType) {
            return false;
        }

        auto layout = dyn_cast<StridedLayoutAttr>(memrefType.getLayout());

        if (!layout) {
            return false;
        }

        if (layout.getOffset() == ShapedType::kDynamic) {
            return true;
        }

        for (auto stride : layout.getStrides()) {
            if (stride == ShapedType::kDynamic) {
                return true;
            }
        }

        return false;
    }

    LogicalResult matchAndRewrite(ProgramOp op, PatternRewriter &rewriter) const override {

        // to keep things simple we infer the offsets only if the program has exactly one use
        if (!op->hasOneUse()) {
            return failure();
        }

        // find all the arguments with dynamic offset
        SmallVector<BlockArgument> dynamicArguments;

        for (auto argument : op.getBody().front().getArguments()) {
            if (isMemRefWithDynamicLayout(argument.getType())) {
                dynamicArguments.push_back(argument);
            }
        }

        if (dynamicArguments.size() == 0) {
            return failure();
        }

        // find the only use of the program
        CreateInvocationOp createInvocationOp =
            dyn_cast<CreateInvocationOp>(*op->getUsers().begin());

        if (!createInvocationOp) {
            return failure();
        }

        // find start program op that starts the program (via the invocation)
        StartProgramOp startProgramOp;
        for (auto user : createInvocationOp->getUsers()) {
            startProgramOp = dyn_cast<StartProgramOp>(user);
            if (startProgramOp) {
                break;
            }
        }

        if (!startProgramOp) {
            return failure();
        }

        bool changed = false;

        for (auto argument : dynamicArguments) {
            // find the index of the argument
            int argumentIndex = argument.getArgNumber();

            // get the corresponding operand of the start program op
            Value operand = startProgramOp.getArgs()[argumentIndex];

            // if the operand is a dynamic memref layout, we cannot infer the layout
            if (isMemRefWithDynamicLayout(operand.getType())) {
                return failure();
            }

            // update the type
            rewriter.modifyOpInPlace(op, [&] { argument.setType(operand.getType()); });

            changed = true;
        }

        return changed ? success() : failure();
    }
};

void ProgramOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
    results.add<InferProgramMemRefLayout>(context);
}

void StartProgramOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {

    // StartProgramOp receives a variadic number of operands (arguments)
    results.add<OpMemRefCastFolder<StartProgramOp>>(context);
}

// removes conversions from T to T
class FoldNoOpConversion : public OpRewritePattern<ConvertOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ConvertOp op, PatternRewriter &rewriter) const override {

        // only apply to non-bufferized converts since bufferized converts cannot fold away
        // since they copy the data between memrefs
        if (!op.getOutput()) {
            return failure();
        }

        // only match no-op conversions
        if (op.getInput().getType() != op.getOutput().getType()) {
            return failure();
        }

        rewriter.replaceOp(op, op.getInput());

        return success();
    }
};

// removes conversions from dense xram to no encoding since it's the same
// encoding, this can be done only if we can change the type on the producer
// of the input value (we cannot change the downstream operations because there
// might be other users that require the original encoding)
class FoldXramToDefaultConversion : public OpRewritePattern<ConvertOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ConvertOp op, PatternRewriter &rewriter) const override {

        // only apply to non-bufferized converts since bufferized converts cannot fold away
        // since they copy the data between memrefs
        if (!op.getOutput()) {
            return failure();
        }

        // check that the output has no encoding
        if (op.getOutput().getType().getEncoding()) {
            return failure();
        }

        // check that the input has a dense xram encoding
        auto inputEncoding = getEncoding(op.getInput().getType());

        if (inputEncoding.getMemSpace() != MemorySpace::Xram) {
            return failure();
        }

        if (!isDenseInMemory(op.getInput().getType())) {
            return failure();
        }

        // check that the value is used only once, otherwise we are not
        // sure we can change the encoding
        if (!op.getInput().hasOneUse()) {
            return failure();
        }

        // we need to process differently the various producers of the input value
        // to make sure the modified producer is still valid

        if (auto emptyOp = op.getInput().getDefiningOp<tensor::EmptyOp>()) {

            rewriter.setInsertionPoint(op);

            auto newEmptyOp = rewriter.create<tensor::EmptyOp>(
                emptyOp.getLoc(), op.getOutput().getType(), emptyOp.getOperands()
            );

            rewriter.replaceOp(op, newEmptyOp.getResult());

            return success();
        }
        else if (auto dsOp = op.getInput().getDefiningOp<DestinationStyleOpInterface>()) {

            if (dsOp.getNumDpsInits() != 1) {
                return failure();
            }

            auto emptyOp = dsOp.getDpsInitOperand(0)->get().getDefiningOp<tensor::EmptyOp>();

            if (!emptyOp) {
                return failure();
            }

            rewriter.setInsertionPoint(dsOp);

            auto newEmptyOp = rewriter.create<tensor::EmptyOp>(
                emptyOp.getLoc(), op.getOutput().getType(), emptyOp.getOperands()
            );

            rewriter.modifyOpInPlace(dsOp, [&] {
                dsOp.setDpsInitOperand(0, newEmptyOp.getResult());
                dsOp->getResult(0).setType(newEmptyOp.getResult().getType());
            });

            rewriter.replaceOp(op, dsOp->getResult(0));

            return success();
        }

        return failure();
    }
};

void ConvertOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
    results.add<FoldNoOpConversion>(context);
    results.add<FoldXramToDefaultConversion>(context);
}

OpFoldResult ImportProgramOp::fold(FoldAdaptor adaptor) { return getNameAttr(); }

} // namespace mlir::syna::torq_hl

static int64_t computeTotalByteSizeElements(MemRefType memrefType) {
    auto inputElementSizeBytes = (memrefType.getElementTypeBitWidth() + 7) / 8;
    return inputElementSizeBytes * memrefType.getNumElements();
}

static int64_t computeTransferredDataSize(ArrayRef<int64_t> shape, int64_t elementSizeBytes) {
    int64_t transferredData = elementSizeBytes;

    for (int i = 0; i < shape.size(); i++) {
        transferredData *= shape[i];
    }

    return transferredData;
}

LogicalResult HostCopyOp::verify() {
    auto inputType = dyn_cast<MemRefType>(getInput().getType());
    if (!inputType) {
        return emitOpError("input must be a memref type");
    }

    auto outputType = dyn_cast<MemRefType>(getOutput().getType());
    if (!outputType) {
        return emitOpError("output must be a memref type");
    }

    if (inputType.getShape() != outputType.getShape()) {
        return emitOpError("input and output shape must match");
    }

    auto shape = getShape();
    auto rank = shape.size();

    if (getInputStridesBytes().size() != rank) {
        return emitOpError("input strides rank must match shape rank");
    }
    if (getOutputStridesBytes().size() != rank) {
        return emitOpError("output strides rank must match shape rank");
    }

    auto transferredBytes = computeTransferredDataSize(shape, getElementSizeBytes());
    auto totalElementsBytes = computeTotalByteSizeElements(inputType);

    if (totalElementsBytes != transferredBytes) {
        return emitOpError(
            "transferred data size " + std::to_string(transferredBytes) +
            " bytes does not match memref data size " + std::to_string(totalElementsBytes) +
            " bytes"
        );
    }

    // FIXME: check that the input/output strides read data within input output memref

    return success();
}

LogicalResult StoreOp::verify() {

    auto inputType = dyn_cast<MemRefType>(getInput().getType());
    if (!inputType) {
        return emitOpError("input must be a memref type");
    }

    auto outputType = dyn_cast<MemRefType>(getOutput().getType());
    if (!outputType) {
        return emitOpError("output must be a memref type");
    }

    if (inputType.getShape() != outputType.getShape()) {
        return emitOpError("input and output shape must match");
    }

    if (getEncodingMemorySpace(inputType) != torq_hl::MemorySpace::Lram) {
        return emitOpError("input must be in lram");
    }

    if (getEncodingMemorySpace(outputType) != torq_hl::MemorySpace::Xram) {
        return emitOpError("output must be in xram");
    }

    auto shape = getShape();
    auto rank = shape.size();

    if (getOutputStridesBytes().size() != rank) {
        return emitOpError("output strides rank must match shape rank");
    }

    auto transferredBytes = computeTransferredDataSize(shape, getElementSizeBytes());
    auto totalElementsBytes = computeTotalByteSizeElements(inputType);

    if (getUnsafe()) {
        if (totalElementsBytes > transferredBytes) {
            return emitOpError(
                "transferred data size " + std::to_string(transferredBytes) +
                " bytes less than memref data size " + std::to_string(totalElementsBytes) + " bytes"
            );
        }

        if (transferredBytes > getEncodedTotalSizeBytes(outputType)) {
            return emitOpError(
                "transferred data size " + std::to_string(transferredBytes) +
                " bytes exceeds output memref buffer size " +
                std::to_string(getEncodedTotalSizeBytes(outputType)) + " bytes"
            );
        }
    }
    else {

        if (!isDenseInMemory(getInput().getType())) {
            return emitOpError("input must be a dense memref");
        }

        if (totalElementsBytes != transferredBytes) {
            return emitOpError(
                "transferred data size " + std::to_string(transferredBytes) +
                " bytes not equal to memref data size " + std::to_string(totalElementsBytes) +
                " bytes"
            );
        }
    }

    return success();
}

LogicalResult LoadOp::verify() {

    auto inputType = dyn_cast<MemRefType>(getInput().getType());
    if (!inputType) {
        return emitOpError("input must be a memref type");
    }

    auto outputType = dyn_cast<MemRefType>(getOutput().getType());
    if (!outputType) {
        return emitOpError("output must be a memref type");
    }

    if (inputType.getShape() != outputType.getShape()) {
        return emitOpError("input and output shape must match");
    }

    if (getEncodingMemorySpace(inputType) != torq_hl::MemorySpace::Xram) {
        return emitOpError("input must be in xram");
    }

    if (getEncodingMemorySpace(outputType) != torq_hl::MemorySpace::Lram) {
        return emitOpError("output must be in lram");
    }

    auto shape = getShape();
    auto rank = shape.size();

    if (getInputStridesBytes().size() != rank) {
        return emitOpError("input strides rank must match shape rank");
    }

    auto transferredBytes = computeTransferredDataSize(shape, getElementSizeBytes());
    auto totalElementsBytes = computeTotalByteSizeElements(inputType);

    if (getUnsafe()) {
        if (totalElementsBytes > transferredBytes) {
            return emitOpError(
                "transferred data size " + std::to_string(transferredBytes) +
                " bytes less than memref data size " + std::to_string(totalElementsBytes) + " bytes"
            );
        }

        if (transferredBytes > getEncodedTotalSizeBytes(outputType)) {
            return emitOpError(
                "transferred data size " + std::to_string(transferredBytes) +
                " bytes exceeds output memref buffer size " +
                std::to_string(getEncodedTotalSizeBytes(outputType)) + " bytes"
            );
        }
    }
    else {

        if (!isDenseInMemory(getOutput().getType())) {
            return emitOpError("output must be a dense memref");
        }

        if (totalElementsBytes != transferredBytes) {
            return emitOpError(
                "transferred data size " + std::to_string(transferredBytes) +
                " bytes not equal to memref data size " + std::to_string(totalElementsBytes) +
                " bytes"
            );
        }
    }

    return success();
}

LogicalResult ConvertOp::verify() {

    auto initEncoding = syna::getEncoding(getInit().getType());

    if (getEncoding()) {
        if (getRequirements()) {
            return emitOpError("encoding and requirements must not both be set");
        }

        if (initEncoding != getEncoding()) {
            return emitOpError("init encoding must match encoding");
        }
    }
    else {

        if (!getRequirements()) {
            return emitOpError("either encoding or requirements must be set");
        }

        if (!checkTypeMatchesEncodingRequirements(getInit().getType(), *getRequirements())) {
            return emitOpError("init type does not match requirements");
        }
    }

    return success();
}

void mlir::syna::torq_hl::getLayerOpEffects(
    Operation *op, SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects
) {

    auto dsOp = cast<DestinationStyleOpInterface>(op);

    for (auto opOperand : dsOp.getDpsInputOperands()) {
        if (!llvm::isa<MemRefType>(opOperand->get().getType()))
            continue;
        effects.emplace_back(
            MemoryEffects::Read::get(), opOperand, /*stage=*/0,
            /*effectOnFullRegion=*/true, SideEffects::DefaultResource::get()
        );
    }

    for (OpOperand &operand : dsOp.getDpsInitsMutable()) {
        if (!llvm::isa<MemRefType>(operand.get().getType()))
            continue;

        if (op->hasTrait<mlir::OpTrait::UpdateInPlaceTrait>()) {
            effects.emplace_back(
                MemoryEffects::Read::get(), &operand, /*stage=*/0,
                /*effectOnFullRegion=*/true, SideEffects::DefaultResource::get()
            );
        }

        effects.emplace_back(
            MemoryEffects::Write::get(), &operand, /*stage=*/0,
            /*effectOnFullRegion=*/true, SideEffects::DefaultResource::get()
        );
    }
}

#define GET_OP_CLASSES
#include "torq/Dialect/TorqHL/TorqHLOps.cpp.inc"
