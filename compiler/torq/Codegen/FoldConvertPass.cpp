// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/EncodingRequirements.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqHw.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-fold-convert"

namespace mlir::syna::torq {

namespace {

FailureOr<Value>
findEarliestCompatibleAncestor(TypedValue<ShapedType> val, EncodingRequirements reqs) {

    // check if the parent of this value is a convert op
    auto parentConvertOp = val.getDefiningOp<torq_hl::ConvertOp>();

    // if it is not a convert op, we try to use it's input as candidate
    if (!parentConvertOp) {
        return failure();
    }

    // check if the input of the parent op was converted from a compatible type
    auto parentValue = findEarliestCompatibleAncestor(parentConvertOp.getInput(), reqs);

    // if it's the case we return that value
    if (succeeded(parentValue)) {
        return parentValue;
    }

    // otherwise check if the input of the parent op is itself compatible and return that
    if (checkTypeMatchesEncodingRequirements(parentConvertOp.getInput().getType(), reqs)) {
        return parentConvertOp.getInput();
    }

    // otherwise we were not able to find any compatible ancestor, we abort
    return failure();
}

// Returns all the SSA values that contain the same value as `value` and that match the encoding
// requirements `req`
static void findCandidatesForValue(
    TypedValue<ShapedType> value, torq_hl::TensorEncodingRequirementsAttr req,
    SmallVectorImpl<Value> &candidates
) {

    LLVM_DEBUG({
        llvm::dbgs() << "    Considering value:\n      ";
        value.dump();
        llvm::dbgs() << "    with type:\n      ";
        value.getType().dump();
        llvm::dbgs() << "    and requirements:\n      ";
        req.dump();
        llvm::dbgs() << "\n";
    });

    if (checkTypeMatchesEncodingRequirements(value.getType(), req)) {
        LLVM_DEBUG({ llvm::dbgs() << "    Value matches requirements\n"; });
        candidates.push_back(value);
    }

    if (auto convertOp = value.getDefiningOp<torq_hl::ConvertOp>()) {
        findCandidatesForValue(convertOp.getInput(), req, candidates);
    }
}

// Simplify any chain of kernel(convert(T1, convert(T2, ... convert(T3, x : T4)))) to kernel(x : T4)
// if T4 is compatible with the kernel input requirements (this includes cross operand constraints)
class FoldConvertChainWithKernel : public OpInterfaceRewritePattern<torq_hl::KernelInterface> {

  public:
    using OpInterfaceRewritePattern<torq_hl::KernelInterface>::OpInterfaceRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::KernelInterface op, PatternRewriter &rewriter) const override {

        // converts are a special case and they don't provide valid requirements
        if (isa<torq_hl::ConvertOp>(op.getOperation())) {
            return failure();
        }

        // FIXME: we cannot simplify these operations because the kernel requirements are currently
        // not correctly defined for them
        if (isa<torq_hl::AddOp, torq_hl::FMAOp>(op.getOperation())) {
            return failure();
        }

        LLVM_DEBUG({
            llvm::dbgs() << "Considering kernel op:\n";
            op.dump();
        });

        // get the requirements for this kernel
        auto kr = op.getKernelEncoding();

        // for each input find all the candiates substitutions, sorted by distance from the
        // operation (i.e. how many convert ops we need to go through to reach that candidate)
        SmallVector<SmallVector<Value>> inputCandidates(op->getNumOperands());
        for (auto &opOperand : op->getOpOperands()) {

            auto ktr = torq_hl::getOperandEncoding(kr, opOperand);
            auto reqAttr = torq_hl::toTensorEncodingRequirementsAttr(ktr).toAttr(op.getContext());
            auto typedOperand = cast<TypedValue<ShapedType>>(opOperand.get());
            auto &candidates = inputCandidates[opOperand.getOperandNumber()];

            LLVM_DEBUG({
                llvm::dbgs() << "  Finding candiates for operand #" << opOperand.getOperandNumber()
                             << ":\n    with type:\n      ";
                typedOperand.getType().dump();
                llvm::dbgs() << "    and requirements:\n      ";
                reqAttr.dump();
                llvm::dbgs() << "\n";
            });

            findCandidatesForValue(typedOperand, reqAttr, candidates);

            LLVM_DEBUG({
                llvm::dbgs() << "  Found " << candidates.size() << " candidates for operand #"
                             << opOperand.getOperandNumber() << ":\n";
                for (auto candidate : candidates) {
                    llvm::dbgs() << "    ";
                    candidate.dump();
                }
                llvm::dbgs() << "\n";
            });

            assert(!candidates.empty() && "at least the original value must be a candidate");
        }

        // the code below works only we have at most one equal encoding constraint
        assert(
            kr.equalEncodingOperands.size() < 2 &&
            "only a single equal enconding constraint is supported"
        );

        LLVM_DEBUG({
            llvm::dbgs() << "  Found " << kr.equalEncodingOperands.size()
                         << " equal encoding constraints\n";
        });

        // for each input that must be the same of another input, remove all candidates for which
        // there is no candidate for the other input with the same encoding
        for (auto &eq : kr.equalEncodingOperands) {

            // best choices for lhs and rhs that removes as many converts as possible, intially we
            // use the original values
            Value bestLhs = inputCandidates[eq.first][0];
            Value bestRhs = inputCandidates[eq.second][0];

            // tracks how many converts can be removed by picking this candidate
            int benefit = 0;

            // for each pair of candidates, check if they have the same encoding and keep track the
            // pair that removes the most converts
            for (auto [lhsIdx, lhsCandidate] : llvm::enumerate(inputCandidates[eq.first])) {
                auto lhsEnc = getEncoding(cast<ShapedType>(lhsCandidate.getType()));

                for (auto [rhsIdx, rhsCandidate] : llvm::enumerate(inputCandidates[eq.second])) {
                    auto rhsEnc = getEncoding(cast<ShapedType>(rhsCandidate.getType()));

                    if (lhsIdx + rhsIdx > benefit && lhsEnc == rhsEnc) {
                        bestLhs = lhsCandidate;
                        bestRhs = rhsCandidate;
                        benefit = lhsIdx + rhsIdx;
                    }
                }
            }

            // keep only the best candidates for both sides
            inputCandidates[eq.first] = {bestLhs};
            inputCandidates[eq.second] = {bestRhs};
        }

        LLVM_DEBUG({
            llvm::dbgs() << "  After equal encoding constraints:\n";
            for (auto [idx, candidates] : llvm::enumerate(inputCandidates)) {
                llvm::dbgs() << "    operand #" << idx << " has " << candidates.size()
                             << " candidates:\n";
                for (auto candidate : candidates) {
                    llvm::dbgs() << "      ";
                    candidate.dump();
                    llvm::dbgs() << "\n";
                }
            }
        });

        // replace all the inputs with the earliest candidates
        bool changed = false;

        for (auto &opOperand : op->getOpOperands()) {

            LLVM_DEBUG({
                llvm::dbgs() << "  Considering operand #" << opOperand.getOperandNumber() << ":\n";
            });

            auto candidate = inputCandidates[opOperand.getOperandNumber()].back();

            if (candidate != opOperand.get()) {

                LLVM_DEBUG({
                    llvm::dbgs() << "  Substituting operand #" << opOperand.getOperandNumber()
                                 << "   from value:\n      ";
                    opOperand.get().dump();
                    llvm::dbgs() << "    to new value:\n      ";
                    candidate.dump();
                });

                rewriter.modifyOpInPlace(op, [&] { opOperand.set(candidate); });

                changed = true;
            }
        }

        if (!changed) {
            return failure();
        }

        return success();
    }
};

// Fold a round trip conversion of the type
//   convert(T1, convert(T2, ... convert(T0, x: T1))) = x
// this pattern also simplifies the special case:
//   convert(T, x: T) = x
class FoldRoundTripConversion : public OpRewritePattern<torq_hl::ConvertOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(torq_hl::ConvertOp op, PatternRewriter &rewriter) const override {

        auto inEncoding = getEncoding(op.getInput().getType());
        auto outEncoding = getEncoding(op.getOutput().getType());

        if (inEncoding.getMemSpace() == torq_hl::MemorySpace::Dtcm ||
            outEncoding.getMemSpace() == torq_hl::MemorySpace::Dtcm) {
            // since we don't support swap in/out of DTCM values we don't fold
            // to avoid running out of memory
            return failure();
        }

        if (inEncoding.getMemSpace() == torq_hl::MemorySpace::Itcm ||
            outEncoding.getMemSpace() == torq_hl::MemorySpace::Itcm) {
            // since we don't support swap in/out of ITCM values we don't fold
            // to avoid running out of memory
            return failure();
        }

        // try to find the earliest ancestor of the output value of this op that
        // is a convert op that produces the same type as the output of this op
        Value candidateAncestorValue;
        auto ancestorConvertOp = op;

        do {

            // only consider the ancestor is operating on tensors
            // memref operations may be need to ensure we copy the data for
            // read-after-write situations
            if (!ancestorConvertOp.getOutput()) {
                break;
            }

            // check if the current ancestor convert op input has the same type
            // as the output of this convert op, if that's the case we found
            // a candidate value to use instead of this convert op
            if (op.getOutput().getType() == ancestorConvertOp.getInput().getType()) {
                candidateAncestorValue = ancestorConvertOp.getInput();
            }

            // move to the next ancestor convert op that we can consider
            ancestorConvertOp = ancestorConvertOp.getInput().getDefiningOp<torq_hl::ConvertOp>();

        } while (ancestorConvertOp);

        // couldn't find a candidate ancestor value
        if (!candidateAncestorValue) {
            return failure();
        }

        rewriter.replaceOp(op, candidateAncestorValue);

        return success();
    }
};

// Fold a chain of convert ops into a single convert if the
// source and target destination are in LRAM, e.g. :
// convert(T2, convert(T1, (....convert(T0, x: T1)))) = convert(T2, x: T1)
class FoldLramToLramConversionChain : public OpRewritePattern<torq_hl::ConvertOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(torq_hl::ConvertOp op, PatternRewriter &rewriter) const override {

        if (!op.getOutput()) {
            return failure();
        }

        auto outEncoding = getEncoding(op.getOutput().getType());

        // we only apply this pattern to conversions that output in LRAM
        if (outEncoding.getMemSpace() != torq_hl::MemorySpace::Lram) {
            return failure();
        }

        LLVM_DEBUG({
            llvm::dbgs() << "Considering to LRAM convert op:\n";
            op.dump();
        });

        Value ancestorInput = op.getInput();
        torq_hl::ConvertOp ancestorConvertOp = op;

        do {

            // only consider the ancestor is operating on tensors
            // memref operations may be need to ensure we copy the data for
            // read-after-write situations
            if (!ancestorConvertOp.getOutput()) {
                break;
            }

            // find the next ancestor convert op
            auto parentOp = ancestorConvertOp.getInput().getDefiningOp<torq_hl::ConvertOp>();

            // defining op of the input is not a convert op, we stop here
            if (!parentOp) {
                LLVM_DEBUG({
                    llvm::dbgs() << "  Input is from:\n";
                    ancestorConvertOp.getInput().dump();
                    llvm::dbgs() << "  No more ancestor convert op found\n";
                });
                break;
            }

            ancestorConvertOp = parentOp;

            LLVM_DEBUG({
                llvm::dbgs() << "  Found ancestor convert op:\n";
                ancestorConvertOp.dump();
            });

            // check if the input of this ancestor convert op is in LRAM, if so
            // we can use it as the new input to the original convert op
            auto inEncoding = getEncoding(ancestorConvertOp.getInput().getType());
            if (inEncoding.getMemSpace() == torq_hl::MemorySpace::Lram) {
                LLVM_DEBUG({
                    llvm::dbgs() << "  Ancestor input is in LRAM, we can use it:\n      ";
                    ancestorConvertOp.getInput().dump();
                });
                ancestorInput = ancestorConvertOp.getInput();
            }

        } while (ancestorConvertOp);

        // we didn't find a better option
        if (ancestorInput == op.getInput()) {
            return failure();
        }

        LLVM_DEBUG({
            llvm::dbgs() << "  Replacing input with:\n      ";
            ancestorInput.dump();
        });

        rewriter.modifyOpInPlace(op, [&] { op.getInputMutable().set(ancestorInput); });

        return success();
    }
};

// When the output of a convert is only used by a tensor.extract_slice, then first perform
// the extract_slice and then the convert, e.g. :
//
// %1 = convert(%0)
// %2 = tensor.extract_slice(%1) {offsets = ..., sizes = ..., strides = ...}
//
// becomes
//
// %1 = tensor.extract_slice(%0) {offsets = ..., sizes = ..., strides = ...}
// %2 = convert(%1)
class SwapExtractAndConvert : public OpRewritePattern<torq_hl::ConvertOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(torq_hl::ConvertOp op, PatternRewriter &rewriter) const override {

        // only work on tensor converts
        if (!op.getOutput()) {
            return failure();
        }

        // if there is more than one use we cannot swap
        if (!op->hasOneUse()) {
            return failure();
        }

        // check that the user is a tensor.extract_slice
        auto extractOp = dyn_cast<tensor::ExtractSliceOp>(*op->user_begin());
        if (!extractOp) {
            return failure();
        }

        // we don't support rank reducing extract operations
        if (extractOp.getSource().getType().getRank() !=
            extractOp.getResult().getType().getRank()) {
            return failure();
        }

        // we insert the swapped operation at the location of the extract op
        // so that we are sure the operands of the extract op are available
        rewriter.setInsertionPoint(extractOp);

        // create an extract slice on the input of the convert
        auto preExtract = rewriter.create<tensor::ExtractSliceOp>(
            extractOp.getLoc(), op.getInput(), extractOp.getMixedOffsets(),
            extractOp.getMixedSizes(), extractOp.getMixedStrides()
        );

        // convert back the slice to the type of the original extract slice result
        auto postConvert =
            convertTensorToType(rewriter, preExtract.getResult(), extractOp.getResult().getType());

        // use the new converted tensor instead of the original extract_op
        rewriter.replaceOp(extractOp, postConvert);
        rewriter.eraseOp(op);

        return success();
    }
};

//
// Finds a concat operation in XRAM that has been expanded to two tensor.insert_slice
// operations and ensures it is done in LRAM if it is sufficiently small
//
// The IR matches looks like this:
//
//  %empty = tensor.empty(...) in XRAM
//  %insert1 = tensor.insert_slice(%input1, %empty, ...)
//  %insert2 = tensor.insert_slice(%input2, %insert1, ...)
//  %insert3 = tensor.insert_slice(%input3, %insert2, ...)
//  ...
//  %insertN = tensor.insert_slice(%inputN, %insertN-1, ...)
//
//  And substitutes it with:
//
//  %empty_lram = tensor.empty(...) in LRAM
//  %input1_lram = convert(%input1) to LRAM
//  %insert1_lram = tensor.insert_slice(%input1_lram, %empty_lram, ...)
//  %input2_lram = convert(%input2) to LRAM
//  %insert2_lram = tensor.insert_slice(%input2_lram, %insert1_lram, ...)
//  %insert2 = convert(%insert2_lram) back to XRAM
//  ...
//  %insertN = convert(%insertN_lram) to XRAM
//
class KeepConcatInLram : public OpRewritePattern<tensor::InsertSliceOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(tensor::InsertSliceOp rootOp, PatternRewriter &rewriter) const override {

        // we only work on tensor.insert_slice ops that insert into XRAM tensors
        auto destType = rootOp.getDest().getType();

        if (getEncodingMemorySpace(destType) != torq_hl::MemorySpace::Xram) {
            return failure();
        }

        // do not match if the rootOp source is a compile time constant
        if (isCompileTimeConst(rootOp)) {
            return failure();
        }

        // do not match if the result is used by another insert slice, we will
        // only the last insert_slice in a chain of insert_slices
        for (auto &use : rootOp.getResult().getUses()) {
            if (isa<tensor::InsertSliceOp>(use.getOwner())) {
                return failure();
            }
        }

        // keep track of the types the chain of insert slice ops
        SmallVector<tensor::InsertSliceOp> insertOps;

        // keep track of the total size of the inputs
        int totalSize = getEncodedTotalSizeBytes(destType);

        auto currentInsert = rootOp;

        // find the whole chain of insert slice ops
        do {
            insertOps.push_back(currentInsert);

            // check we can fit the concat in LRAM
            totalSize += getEncodedTotalSizeBytes(currentInsert.getSource().getType());

            // we cannot fit the whole contact in LRAM so we won't apply this pattern
            if (totalSize > TorqHw::get().getAvailableMemoryForTiling()) {
                return failure();
            }

            currentInsert = currentInsert.getDest().getDefiningOp<tensor::InsertSliceOp>();

        } while (currentInsert);

        // check that we start from an empty op
        auto emptyOp = insertOps.back().getDest().getDefiningOp<tensor::EmptyOp>();

        if (!emptyOp) {
            return failure();
        }

        auto newDestEncoding = createDenseEncoding(destType, torq_hl::MemorySpace::Lram);

        auto newDestValue =
            rewriter
                .create<tensor::EmptyOp>(
                    rootOp.getLoc(), destType.getShape(), destType.getElementType(), newDestEncoding
                )
                .getResult();

        for (auto insertOp : llvm::reverse(insertOps)) {

            // convert the source to LRAM
            auto sourceType = insertOp.getSource().getType();
            auto lramEncoding = createDenseEncoding(sourceType, torq_hl::MemorySpace::Lram);
            auto newSource = convertTensorToEncoding(rewriter, insertOp.getSource(), lramEncoding);

            // create a new insert slice op that works in LRAM
            auto newInsertOp = rewriter.create<tensor::InsertSliceOp>(
                insertOp.getLoc(), newSource, newDestValue, insertOp.getMixedOffsets(),
                insertOp.getMixedSizes(), insertOp.getMixedStrides()
            );

            newDestValue = newInsertOp.getResult();
        }

        // convert back the result to XRAM
        auto finalResult =
            convertTensorToType(rewriter, newDestValue, rootOp.getResult().getType());

        // replace the original insert_slice op result with the final result
        rewriter.replaceOp(rootOp, finalResult);

        return success();
    }
};

class FoldConvertPass : public FoldConvertBase<FoldConvertPass> {
  public:
    using FoldConvertBase<FoldConvertPass>::FoldConvertBase;
    void runOnOperation() override;
};

void FoldConvertPass::runOnOperation() {
    MLIRContext *ctx = getOperation().getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<FoldConvertChainWithKernel>(ctx);
    patterns.add<FoldRoundTripConversion>(ctx);
    patterns.add<FoldLramToLramConversionChain>(ctx);
    patterns.add<KeepConcatInLram>(ctx);

#if 0
    // This pattern creates a situation where elementwise ops have inputs with different
    // encodings (strides), which is not supported yet. So we disable it for now.
    // It can be re-enabled once we support elementwise ops with different input encodings.
    patterns.add<SwapExtractAndConvert>(ctx);
#endif

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createFoldConvertPass() {
    return std::make_unique<FoldConvertPass>();
}

} // namespace mlir::syna::torq
