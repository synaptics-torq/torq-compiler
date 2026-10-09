// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/StorageUtils.h"

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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

void StartProgramOp::getAsyncAccesses(SmallVectorImpl<AsyncAccess> &accesses) {

    // the program reads its code sections
    for (auto codeSection : getCodeSections()) {
        accesses.push_back({ArgAccessBitfield::Read, cast<TypedValue<MemRefType>>(codeSection)});
    }

    // without access information every argument may be read and written
    auto argAccesses = getArgAccessesAttr();
    for (auto [index, arg] : llvm::enumerate(getArgs())) {
        if (!isa<MemRefType>(arg.getType())) {
            continue;
        }
        auto access = argAccesses ? cast<ArgAccessBitfieldAttr>(argAccesses[index]).getValue()
                                  : ArgAccessBitfield::Read | ArgAccessBitfield::Write;
        if (access != ArgAccessBitfield::None) {
            accesses.push_back({access, cast<TypedValue<MemRefType>>(arg)});
        }
    }
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

class FoldBufferizedSelfCopy : public OpRewritePattern<ConvertOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ConvertOp op, PatternRewriter &rewriter) const override {

        // only applies to the bufferized (memref) form
        if (op.getOutput()) {
            return failure();
        }

        if (op.getInput() != op.getInit()) {
            return failure();
        }

        rewriter.eraseOp(op);
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

            auto newEmptyOp = tensor::EmptyOp::create(
                rewriter, emptyOp.getLoc(), op.getOutput().getType(), emptyOp.getOperands()
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

            auto newEmptyOp = tensor::EmptyOp::create(
                rewriter, emptyOp.getLoc(), op.getOutput().getType(), emptyOp.getOperands()
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
    results.add<FoldBufferizedSelfCopy>(context);
    results.add<FoldXramToDefaultConversion>(context);
}

OpFoldResult ImportProgramOp::fold(FoldAdaptor adaptor) { return getNameAttr(); }

} // namespace mlir::syna::torq_hl

static int64_t computeTotalByteSizeElements(MemRefType memrefType) {
    return mlir::syna::getStorageSizeBytes(
        memrefType.getNumElements(), memrefType.getElementTypeBitWidth()
    );
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

LogicalResult DynamicHostCopyOp::verify() {
    bool hasConstIdx = getInputByteOffsetConstantIndex().has_value();
    bool hasBindingIdx = getInputByteOffsetBindingIndex().has_value();
    if (hasConstIdx == hasBindingIdx) {
        return emitOpError("exactly one of input_byte_offset_constant_index or "
                           "input_byte_offset_binding_index must be set");
    }

    if (getIndexUnitBytes() <= 0) {
        return emitOpError("index_unit_bytes must be positive");
    }

    auto inputType = cast<MemRefType>(getInput().getType());
    auto outputType = cast<MemRefType>(getOutput().getType());

    if (inputType.getElementType() != outputType.getElementType()) {
        return emitOpError("input and output element type must match");
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

    // For PreDispatchRead the dynamic side is `input` (binding) and `output`
    // is the static destination, which must match the transfer exactly. For
    // PostDispatchWrite the dynamic side is `output` (binding) and `input` is
    // the static source, which must match instead.
    bool preRead = getDirection() == DynamicHostCopyDirection::PreDispatchRead;
    auto staticType = preRead ? outputType : inputType;
    auto staticSide = preRead ? "output" : "input";
    auto staticBytes = computeTotalByteSizeElements(staticType);
    if (staticBytes != transferredBytes) {
        return emitOpError(
            "transferred data size " + std::to_string(transferredBytes) + " bytes does not match " +
            staticSide + " memref size " + std::to_string(staticBytes) + " bytes"
        );
    }

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

ArgAccessBitfield mlir::syna::torq_hl::getValueAccessFromEffects(
    ArrayRef<SideEffects::EffectInstance<MemoryEffects::Effect>> effects, Value value
) {
    auto access = ArgAccessBitfield::None;
    if (!isa<MemRefType>(value.getType())) {
        return access;
    }
    for (auto &effect : effects) {
        if (effect.getValue() != value) {
            continue;
        }
        if (isa<MemoryEffects::Read>(effect.getEffect())) {
            access = access | ArgAccessBitfield::Read;
        }
        if (isa<MemoryEffects::Write>(effect.getEffect())) {
            access = access | ArgAccessBitfield::Write;
            if (!effect.getEffectOnFullRegion()) {
                access = access | ArgAccessBitfield::Read;
            }
        }
    }
    return access;
}

Speculation::Speculatability mlir::syna::torq_hl::getLayerOpSpeculatability(Operation *op) {

    if (!cast<DestinationStyleOpInterface>(op).hasPureTensorSemantics()) {
        return Speculation::NotSpeculatable;
    }

    // gather/scatter use dynamic indices. Speculating them can result in UB.
    // TODO: speculate if provably safe.
    if (isa<GatherOp, ScatterOp>(op)) {
        return Speculation::NotSpeculatable;
    }

    return Speculation::Speculatable;
}

LogicalResult mlir::syna::torq_hl::NextOp::verify() {

    if (getSuccessor()->getNumArguments() != getArguments().size()) {
        return emitOpError("number of successor block arguments must match number of operands");
    }

    return success();
}

LogicalResult mlir::syna::torq_hl::DescriptorOp::verify() {

    auto memrefType = cast<MemRefType>(getCodeSection().getType());

    if (!memrefType.getElementType().isInteger(8)) {
        return emitOpError("code section must have i8 elements");
    }

    if (getCode().size() != memrefType.getNumElements()) {
        return emitOpError("code holds ") << getCode().size() << " bytes for a code section of "
                                          << memrefType.getNumElements() << " bytes";
    }

    return success();
}

SmallVector<int64_t> mlir::syna::torq_hl::ProgramOp::getBlockOffsets() {

    auto blockSizes = *getBlockSizes();

    SmallVector<int64_t> offsets;
    offsets.reserve(blockSizes.size() + 1);
    offsets.push_back(0);
    for (auto blockSize : blockSizes) {
        offsets.push_back(offsets.back() + blockSize);
    }

    return offsets;
}

LogicalResult mlir::syna::torq_hl::ProgramOp::verify() {

    auto argAccesses = getArgAccessesAttr();

    // the arguments live on the entry block, which is not there yet while the op
    // is being built
    if (argAccesses && !getBody().empty() &&
        argAccesses.size() != getBody().front().getNumArguments()) {
        return emitOpError("arg_accesses size must match the number of program arguments");
    }

    if (auto blockSizes = getBlockSizes()) {
        if (blockSizes->size() != getBody().getBlocks().size()) {
            return emitOpError("block_sizes has ") << blockSizes->size() << " entries for "
                                                   << getBody().getBlocks().size() << " blocks";
        }
    }

    return success();
}

LogicalResult mlir::syna::torq_hl::StartProgramOp::verify() {

    if (auto argAccesses = getArgAccessesAttr()) {
        if (argAccesses.size() != getArgs().size()) {
            return emitOpError("arg_accesses size must match the number of arguments");
        }
    }

    auto invocation = getInvocation().getDefiningOp<CreateInvocationOp>();

    if (!invocation) {
        return success();
    }

    if (auto invocationArgs = invocation.getInvocationArgs()) {

        if (invocationArgs->size() != getArgs().size()) {
            return emitOpError(
                "invocation_args size of the invocation must match the number of arguments"
            );
        }

        for (auto [idx, arg] : llvm::enumerate(getArgs())) {
            Attribute entry = (*invocationArgs)[idx];
            if (isa<InvocationType>(arg.getType()) && !isa<InvocationAttr>(entry)) {
                return emitOpError() << "argument #" << idx
                                     << " is an invocation but its invocation_args entry is not "
                                        "an InvocationAttr";
            }
            if (isa<MemRefType>(arg.getType()) && !isa<BufferAttr>(entry)) {
                return emitOpError() << "argument #" << idx
                                     << " is a buffer but its invocation_args entry is not a "
                                        "BufferAttr";
            }
        }
    }

    auto invocationProgram = invocation.getProgram().getDefiningOp<ProgramOp>();

    if (!invocationProgram) {
        return success();
    }

    if (invocationProgram.getBody().front().getNumArguments() != getArgs().size()) {
        return emitOpError(
            "number of program arguments must match number of start_program arguments"
        );
    }

    return success();
}

LogicalResult mlir::syna::torq_hl::ProgramCodeOp::verify() {

    if (getProgram().getType().getExecutor() != Executor::CSS) {
        return emitOpError("only CSS programs have a position-independent shared code section");
    }

    return success();
}

#define GET_OP_CLASSES
#include "torq/Dialect/TorqHL/TorqHLOps.cpp.inc"

namespace mlir::syna::torq_hl {

// A clamp that covers the entire integer range is a no-op.
static bool isFullIntegerClamp(ActOp op) {
    if (op.getName() != "clamp")
        return false;

    // Only fold pure clamps (no extra weights).
    if (op.getWeights())
        return false;

    auto inTy = dyn_cast<RankedTensorType>(op.getInput().getType());
    Value output = op.getOutput();
    if (!output)
        return false;
    auto outTy = dyn_cast<RankedTensorType>(output.getType());
    if (!inTy || !outTy || inTy != outTy)
        return false;

    auto intTy = dyn_cast<IntegerType>(inTy.getElementType());
    if (!intTy)
        return false;

    // The zero points must match; otherwise the op would shift values.
    if (op.getInputZp() != op.getOutputZp())
        return false;

    unsigned width = intTy.getWidth();
    int64_t min = op.getMinInt();
    int64_t max = op.getMaxInt();
    bool isSigned = min < 0;

    APInt actualMin(width, min, isSigned);
    APInt actualMax(width, max, isSigned);
    APInt expectedMin(width, isSigned ? llvm::minIntN(width) : 0, isSigned);
    APInt expectedMax(width, isSigned ? llvm::maxIntN(width) : llvm::maxUIntN(width), isSigned);

    return actualMin == expectedMin && actualMax == expectedMax;
}

// Fold a torq_hl.act "clamp" whose limits span the whole integer range.
// It is a no-op and just forwards its input.
class FoldNoOpClamp : public OpRewritePattern<ActOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ActOp op, PatternRewriter &rewriter) const override {
        if (!isFullIntegerClamp(op))
            return failure();

        rewriter.replaceOp(op, op.getInput());
        return success();
    }
};

// Sink a widening float conversion into the multiply that produces its input.
//
// An act "f2f" from bf16 to f32 is a store format, not arithmetic: the ALU's float
// partials are fp32 already, and the producer's bf16 result is the rounding its own
// ACT applied on the way out. A producer whose partials are fp32 can therefore write
// f32 itself, which costs nothing and removes the conversion's whole pass.
//
// Restricted to torq_hl.mul with float operands, where the partials are fp32 by
// construction. An integer-partial producer would need an i2f conversion, which is
// arithmetic and not foldable this way.
class SinkWideningConvertIntoMul : public OpRewritePattern<ActOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ActOp op, PatternRewriter &rewriter) const override {
        if (op.getName() != "f2f" || op.getWeights())
            return failure();

        Value output = op.getOutput();
        auto inTy = dyn_cast<RankedTensorType>(op.getInput().getType());
        auto outTy = output ? dyn_cast<RankedTensorType>(output.getType()) : nullptr;
        if (!inTy || !outTy || inTy.getShape() != outTy.getShape() ||
            !inTy.getElementType().isBF16() || !outTy.getElementType().isF32())
            return failure();

        auto mul = op.getInput().getDefiningOp<MulOp>();
        if (!mul || !mul.getOutput() || !mul.getOutput().hasOneUse())
            return failure();
        for (Value in : {mul.getInput1(), mul.getInput2()}) {
            if (!isa<FloatType>(cast<ShapedType>(in.getType()).getElementType()))
                return failure();
        }

        // Reuse the conversion's own init rather than building one: it already has the
        // wider type, the same shape and whatever encoding the tensor carries at this
        // point in the pipeline. It only has to be moved above its new user.
        auto init = op.getInit().getDefiningOp<tensor::EmptyOp>();
        if (!init || !init.getDynamicSizes().empty())
            return failure();
        rewriter.moveOpBefore(init, mul);

        rewriter.modifyOpInPlace(mul, [&]() {
            mul.getInitMutable().assign(init.getResult());
            mul.getOutput().setType(outTy);
        });
        rewriter.replaceOp(op, mul.getOutput());
        return success();
    }
};

void ActOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
    results.add<FoldNoOpClamp, SinkWideningConvertIntoMul>(context);
}

/// CastOpInterface: the source and result must be compatible shaped types
/// (their shape compatibility is checked by the SameOperandsAndResultShape
/// trait) and the element types must be identical, or both int/float of the
/// same bit width (a byte re-interpretation, e.g. bf16 <-> i16).
bool BitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
    if (inputs.size() != 1 || outputs.size() != 1)
        return false;
    auto sourceType = dyn_cast<ShapedType>(inputs.front());
    auto resultType = dyn_cast<ShapedType>(outputs.front());
    if (!sourceType || !resultType)
        return false;
    Type sourceElement = sourceType.getElementType();
    Type resultElement = resultType.getElementType();
    if (sourceElement == resultElement)
        return true;
    return sourceElement.isIntOrFloat() && resultElement.isIntOrFloat() &&
           sourceElement.getIntOrFloatBitWidth() == resultElement.getIntOrFloatBitWidth();
}

/// bitcast(extract_slice(x)) -> extract_slice(bitcast(x))
///
/// Tiling places the extract_slice below the bitcast (bitcast of the tile).
/// Bufferized, the bitcast then becomes a torq_hl.reinterpret_cast over the
/// per-tile subview, which must carry the tile's non-zero offset - a form the
/// address assignment does not support (a view shares its source's address).
/// The element types have the same width, so the commutation is byte-exact;
/// with the bitcast above the slice, the bufferized reinterpret_cast sits
/// directly on the whole buffer with offset 0 and the tiles subview the
/// reinterpreted view instead.
class CommuteBitcastThroughExtractSlice : public OpRewritePattern<BitcastOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(BitcastOp op, PatternRewriter &rewriter) const override {
        auto slice = op.getSource().getDefiningOp<tensor::ExtractSliceOp>();
        if (!slice)
            return failure();
        auto sourceTensorType = dyn_cast<RankedTensorType>(slice.getSource().getType());
        auto resultTensorType = dyn_cast<RankedTensorType>(op.getType());
        if (!sourceTensorType || !resultTensorType)
            return failure();

        Value whole = rewriter.create<BitcastOp>(
            op.getLoc(),
            RankedTensorType::get(
                sourceTensorType.getShape(), resultTensorType.getElementType(),
                sourceTensorType.getEncoding()
            ),
            slice.getSource()
        );
        rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
            op, resultTensorType, whole, slice.getMixedOffsets(), slice.getMixedSizes(),
            slice.getMixedStrides()
        );
        return success();
    }
};

void BitcastOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
    results.add<CommuteBitcastThroughExtractSlice>(context);
}

//===----------------------------------------------------------------------===//
// ReinterpretCastOp
//===----------------------------------------------------------------------===//

/// Sets values[i] to constValues[i] if the latter is a static value, as
/// indicated by ShapedType::kDynamic. If constValues[i] is dynamic, tries to
/// extract a constant value from values[i] to allow for additional folding
/// opportunities. Also converts all existing attributes to index attributes
/// (they may be i64 attributes). Same helper MLIR uses for
/// memref.reinterpret_cast (mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp).
static void
constifyIndexValues(SmallVectorImpl<OpFoldResult> &values, ArrayRef<int64_t> constValues) {
    assert(constValues.size() == values.size() && "incorrect number of const values");
    for (auto [i, cstVal] : llvm::enumerate(constValues)) {
        Builder builder(values[i].getContext());
        if (ShapedType::isStatic(cstVal)) {
            // Constant value is known, use it directly.
            values[i] = builder.getIndexAttr(cstVal);
            continue;
        }
        if (std::optional<int64_t> cst = getConstantIntValue(values[i])) {
            // Try to extract a constant or convert an existing one to index.
            values[i] = builder.getIndexAttr(*cst);
        }
    }
}

/// Build with mixed static and dynamic entries: `static_offsets`,
/// `static_sizes` and `static_strides` are filled with sentinel values that
/// encode the dynamic entries, which become the corresponding operands.
void ReinterpretCastOp::build(
    OpBuilder &b, OperationState &result, MemRefType resultType, Value source, OpFoldResult offset,
    ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides, ArrayRef<NamedAttribute> attrs
) {
    SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
    SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
    dispatchIndexOpFoldResults(offset, dynamicOffsets, staticOffsets);
    dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
    dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
    result.addAttributes(attrs);
    build(
        b, result, resultType, source, dynamicOffsets, dynamicSizes, dynamicStrides,
        b.getDenseI64ArrayAttr(staticOffsets), b.getDenseI64ArrayAttr(staticSizes),
        b.getDenseI64ArrayAttr(staticStrides)
    );
}

/// Build with fully static entries.
void ReinterpretCastOp::build(
    OpBuilder &b, OperationState &result, MemRefType resultType, Value source, int64_t offset,
    ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<NamedAttribute> attrs
) {
    SmallVector<OpFoldResult> sizeValues = llvm::to_vector<4>(
        llvm::map_range(sizes, [&](int64_t v) -> OpFoldResult { return b.getI64IntegerAttr(v); })
    );
    SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
        llvm::map_range(strides, [&](int64_t v) -> OpFoldResult { return b.getI64IntegerAttr(v); })
    );
    build(
        b, result, resultType, source, b.getI64IntegerAttr(offset), sizeValues, strideValues, attrs
    );
}

LogicalResult ReinterpretCastOp::verify() {
    auto srcType = cast<MemRefType>(getSource().getType());
    auto resultType = cast<MemRefType>(getType());
    if (srcType.getMemorySpace() != resultType.getMemorySpace())
        return emitError("different memory spaces specified for source type ")
               << srcType << " and result memref type " << resultType;
    if (srcType.getElementType() != resultType.getElementType()) {
        // A reinterpret_cast may re-interpret the element type when both
        // element types have the same bit width (e.g. a bf16 buffer viewed as
        // i16): the bytes in memory are unchanged. Shape, layout and memory
        // space must still match, so the cast stays a free re-interpretation.
        if (!srcType.getElementType().isIntOrFloat() ||
            !resultType.getElementType().isIntOrFloat() ||
            srcType.getElementType().getIntOrFloatBitWidth() !=
                resultType.getElementType().getIntOrFloatBitWidth())
            return emitError("different element types specified for source type ")
                   << srcType << " and result memref type " << resultType;
    }
    if (srcType.getShape() != resultType.getShape())
        return emitError("the source and result must have the same shape, got ")
               << srcType << " to " << resultType;

    // Match sizes in result memref type and in static_sizes attribute.
    for (auto [idx, resultSize, expectedSize] :
         llvm::enumerate(resultType.getShape(), getStaticSizes())) {
        if (ShapedType::isStatic(resultSize) && resultSize != expectedSize)
            return emitError("expected result type with size = ")
                   << (ShapedType::isDynamic(expectedSize) ? std::string("dynamic")
                                                           : std::to_string(expectedSize))
                   << " instead of " << resultSize << " in dim = " << idx;
    }

    // Match offset and strides in static_offsets and static_strides
    // attributes. If the result memref type has no layout, this assumes an
    // identity layout.
    int64_t resultOffset;
    SmallVector<int64_t, 4> resultStrides;
    if (failed(resultType.getStridesAndOffset(resultStrides, resultOffset)))
        return emitError("expected result type to have strided layout but found ") << resultType;

    // Match offset in result memref type and in static_offsets attribute.
    int64_t expectedOffset = getStaticOffsets().front();
    if (ShapedType::isStatic(resultOffset) && resultOffset != expectedOffset)
        return emitError("expected result type with offset = ")
               << (ShapedType::isDynamic(expectedOffset) ? std::string("dynamic")
                                                         : std::to_string(expectedOffset))
               << " instead of " << resultOffset;

    // Match strides in result memref type and in static_strides attribute.
    for (auto [idx, resultStride, expectedStride] :
         llvm::enumerate(resultStrides, getStaticStrides())) {
        if (ShapedType::isStatic(resultStride) && resultStride != expectedStride)
            return emitError("expected result type with stride = ")
                   << (ShapedType::isDynamic(expectedStride) ? std::string("dynamic")
                                                             : std::to_string(expectedStride))
                   << " instead of " << resultStride << " in dim = " << idx;
    }

    return success();
}

/// Whether the op's result starts at the same address as its source's
/// (i.e. the op does not add an offset of its own). A view over such an op
/// can swap the op for its source without changing the view's geometry, which
/// is relative to the same base pointer.
static bool viewHasSameBaseAsSource(Operation *op) {
    if (isa<memref::CastOp>(op))
        return true;
    if (auto subview = dyn_cast<memref::SubViewOp>(op))
        // Only a zero-offset subview with the full source shape keeps both
        // the base pointer and the shape (this op requires the latter).
        return llvm::all_of(subview.getMixedOffsets(), isZeroInteger) &&
               cast<ShapedType>(subview.getType()).getShape() ==
                   cast<ShapedType>(subview.getSource().getType()).getShape();
    if (auto view = dyn_cast<ReinterpretCastOp>(op))
        return getConstantIntValue(view.getConstifiedMixedOffset()) == 0;
    return false;
}

OpFoldResult ReinterpretCastOp::fold(FoldAdaptor /*adaptor*/) {
    Value src = getSource();

    // reinterpret_cast(x) over an offset-free view of the same buffer can
    // drop the intermediate view: the geometry is carried by this op, which
    // is relative to the same base pointer.
    if (Operation *prev = src.getDefiningOp()) {
        if (viewHasSameBaseAsSource(prev)) {
            getSourceMutable().assign(prev->getOperand(0));
            return getResult();
        }
    }

    // A same-type reinterpret_cast without an offset change is a no-op.
    if (ShapedType::isStaticShape(getType().getShape()) && src.getType() == getType() &&
        getStaticOffsets().front() == 0)
        return src;

    return nullptr;
}

SmallVector<OpFoldResult> ReinterpretCastOp::getConstifiedMixedSizes() {
    SmallVector<OpFoldResult> values = getMixedSizes();
    constifyIndexValues(values, getType().getShape());
    return values;
}

SmallVector<OpFoldResult> ReinterpretCastOp::getConstifiedMixedStrides() {
    SmallVector<OpFoldResult> values = getMixedStrides();
    SmallVector<int64_t> staticValues;
    int64_t unused;
    LogicalResult status = getType().getStridesAndOffset(staticValues, unused);
    (void)status;
    assert(succeeded(status) && "could not get strides from type");
    constifyIndexValues(values, staticValues);
    return values;
}

OpFoldResult ReinterpretCastOp::getConstifiedMixedOffset() {
    SmallVector<OpFoldResult> values = getMixedOffsets();
    assert(values.size() == 1 && "reinterpret_cast must have one and only one offset");
    SmallVector<int64_t> staticValues, unused;
    int64_t offset;
    LogicalResult status = getType().getStridesAndOffset(unused, offset);
    (void)status;
    assert(succeeded(status) && "could not get offset from type");
    staticValues.push_back(offset);
    constifyIndexValues(values, staticValues);
    return values[0];
}

namespace {

/// Re-spell the view once more of its entries are known than when it was
/// created: they become constants as the pipeline resolves the bindings and
/// unrolls the loops. Mirrors MLIR's ReinterpretCastOpConstantFolder for
/// memref.reinterpret_cast, keeping this op's element type (which it may
/// re-interpret). Like for memref.reinterpret_cast and subview, this makes
/// any canonicalizer in the pipeline fold the view as soon as its geometry
/// is known, so no dedicated refine pass is needed.
struct ReinterpretCastOpConstantFolder : public OpRewritePattern<ReinterpretCastOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(ReinterpretCastOp op, PatternRewriter &rewriter) const override {
        unsigned srcStaticCount = llvm::count_if(
            llvm::concat<OpFoldResult>(
                op.getMixedOffsets(), op.getMixedSizes(), op.getMixedStrides()
            ),
            [](OpFoldResult ofr) { return isa<Attribute>(ofr); }
        );

        SmallVector<OpFoldResult> offsets = {op.getConstifiedMixedOffset()};
        SmallVector<OpFoldResult> sizes = op.getConstifiedMixedSizes();
        SmallVector<OpFoldResult> strides = op.getConstifiedMixedStrides();
        unsigned dstStaticCount = llvm::count_if(
            llvm::concat<OpFoldResult>(offsets, sizes, strides),
            [](OpFoldResult ofr) { return isa<Attribute>(ofr); }
        );

        // Nothing new was constified: leave the op alone.
        if (srcStaticCount == dstStaticCount)
            return failure();

        // Rebuild the result type from the constified entries, keeping the
        // op's element type and memory space.
        auto toStatic = [](OpFoldResult ofr) -> int64_t {
            return isa<Attribute>(ofr) ? cast<IntegerAttr>(cast<Attribute>(ofr)).getInt()
                                       : ShapedType::kDynamic;
        };
        SmallVector<int64_t> staticSizes =
            llvm::to_vector(llvm::map_range(sizes, [&](OpFoldResult s) { return toStatic(s); }));
        SmallVector<int64_t> staticStrides =
            llvm::to_vector(llvm::map_range(strides, [&](OpFoldResult s) { return toStatic(s); }));
        int64_t staticOffset = toStatic(offsets.front());

        MemRefType refinedType = MemRefType::get(
            staticSizes, op.getType().getElementType(),
            StridedLayoutAttr::get(op.getContext(), staticOffset, staticStrides),
            op.getType().getMemorySpace()
        );

        rewriter.replaceOpWithNewOp<ReinterpretCastOp>(
            op, refinedType, op.getSource(), offsets.front(), sizes, strides
        );
        return success();
    }
};

} // namespace

void ReinterpretCastOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context
) {
    results.add<ReinterpretCastOpConstantFolder>(context);
}

} // namespace mlir::syna::torq_hl
