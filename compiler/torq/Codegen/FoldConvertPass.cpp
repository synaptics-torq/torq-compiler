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

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-fold-convert"

// Budget for the residency guard in SwapExtractAndConvert, as a percentage of LRAM. It limits
// how much extra LRAM the swap keeps live. See the comment at that guard. 100 turns it off.
//
// We picked the default from measurement. It is above what a winning bf16 conv block needs, and
// well below the point where a full model runs out of LRAM. With this default, every full model
// we measured compiles and runs faster.
//
// The percentage is for the chip we tuned on. A smaller LRAM gets a smaller share. See
// kSwapExtractAndConvertTunedLramSize.
static llvm::cl::opt<int64_t> clSwapExtractAndConvertMaxWastePercent(
    "torq-swap-extract-convert-max-waste-percent",
    llvm::cl::desc("Max extra LRAM residency the SwapExtractAndConvert pattern may add, as a "
                   "percentage of the LRAM size, on the chip the default was tuned on"),
    llvm::cl::init(12)
);

// The LRAM size of the chip we tuned the percentage on. A chip with less LRAM than this gets a
// smaller share. The residency guard says why.
static constexpr int64_t kSwapExtractAndConvertTunedLramSize = 512 * 1024;

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
    SwapExtractAndConvert(MLIRContext *ctx, int64_t lramSize)
        : OpRewritePattern(ctx), lramSize_(lramSize) {}

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

        if (auto rejection = checkSliceIsOneBlock(extractOp)) {
            return reject(op, rewriter, *rejection);
        }

        if (auto rejection = checkConsumers(extractOp)) {
            return reject(op, rewriter, *rejection);
        }

        if (auto rejection = checkResidency(op, extractOp)) {
            return reject(op, rewriter, *rejection);
        }

        // we insert the swapped operation at the location of the extract op
        // so that we are sure the operands of the extract op are available
        rewriter.setInsertionPoint(extractOp);

        // create an extract slice on the input of the convert
        auto preExtract = tensor::ExtractSliceOp::create(
            rewriter, extractOp.getLoc(), op.getInput(), extractOp.getMixedOffsets(),
            extractOp.getMixedSizes(), extractOp.getMixedStrides()
        );

        // convert back the slice to the type of the original extract slice result
        auto postConvert =
            convertTensorToType(rewriter, preExtract.getResult(), extractOp.getResult().getType());

        // use the new converted tensor instead of the original extract_op
        rewriter.replaceOp(extractOp, postConvert);
        rewriter.eraseOp(op);

        LLVM_DEBUG(llvm::dbgs() << "SwapExtractAndConvert: applied\n");

        return success();
    }

  private:
    // Why a guard turned the swap down: a short tag for the debug log, and a reason for the
    // pattern driver. A guard returns nothing when it lets the swap through.
    struct Rejection {
        const char *tag;
        const char *reason;
    };

    static LogicalResult
    reject(torq_hl::ConvertOp op, PatternRewriter &rewriter, Rejection rejection) {
        LLVM_DEBUG(llvm::dbgs() << "SwapExtractAndConvert: blocked (" << rejection.tag << ")\n");
        return rewriter.notifyMatchFailure(op, rejection.reason);
    }

    // Kernels join dimensions before they read a tensor. The conv, the depthwise conv and the max
    // pool kernel all join H with W. This works only when the inner one is taken whole. If it is
    // not, the rows do not sit next to each other in memory. Then torq::fuse gives up and stops
    // with "Could not fuse the requested number of dimensions".
    //
    // A slice is safe when it cuts one dimension and takes every inner dimension whole. A halo
    // slab does this. A channel cut does this too. Those are the shapes this pattern is for. A
    // slice that cuts H and W at the same time is not safe. It breaks the conv kernels
    // (test_keras_app.py, the nasnetmobile separable conv layers). We saw this only on a chip with
    // small LRAM, where the compiler tiles both directions.
    //
    // We look at the slice, not at the consumer. Blocking every conv would block the halo slabs
    // too, and those are the case this pattern wins on.
    static std::optional<Rejection> checkSliceIsOneBlock(tensor::ExtractSliceOp extractOp) {

        ArrayRef<int64_t> parentShape = extractOp.getSource().getType().getShape();
        SmallVector<OpFoldResult> sizes = extractOp.getMixedSizes();
        SmallVector<OpFoldResult> strides = extractOp.getMixedStrides();

        // true once we have passed a dimension that the slice cuts
        bool pastCut = false;

        for (auto [dim, parentExtent] : llvm::enumerate(parentShape)) {

            std::optional<int64_t> size = getConstantIntValue(sizes[dim]);
            std::optional<int64_t> stride = getConstantIntValue(strides[dim]);

            if (!size || !stride || ShapedType::isDynamic(parentExtent)) {
                return Rejection{"dynamic slice", "slice is not static"};
            }

            // a stride above one leaves holes inside the dimension
            if (*stride != 1) {
                return Rejection{"stride", "slice skips elements"};
            }

            bool whole = (*size == parentExtent);

            if (pastCut && !whole) {
                return Rejection{"not contiguous", "slice cuts more than one dimension"};
            }

            pastCut = pastCut || !whole;
        }

        return std::nullopt;
    }

    // After the swap the kernel reads a strided LRAM subview instead of a dense buffer. Three
    // kernels cannot do that, and one more gets too big from it.
    //
    // The kernel is not the direct user of the extract_slice. The chain is
    // convert ; extract_slice ; convert ; kernel, so the kernel is two or more hops away. So we
    // follow every torq_hl.convert and check every consumer, not only the first one. One buffer
    // can feed more than one kernel (measured: fma and then table).
    static std::optional<Rejection> checkConsumers(tensor::ExtractSliceOp extractOp) {

        SmallVector<Operation *> worklist(
            extractOp.getResult().getUsers().begin(), extractOp.getResult().getUsers().end()
        );
        SmallPtrSet<Operation *, 8> visited;

        while (!worklist.empty()) {
            Operation *consumer = worklist.pop_back_val();
            if (!visited.insert(consumer).second) {
                continue;
            }

            // A two-tensor torq_hl.add reads both inputs in one traversal. Its NDL gets an extra
            // dimension of size 2, and the stride of that dimension is the gap between the two
            // input addresses. This only works when both inputs have the same strides. If they
            // differ, AddPattern.cpp stops with "Add input strides must matchh". A scalar rhs is
            // safe, because that path checks isDenseInMemory() first.
            if (auto addOp = dyn_cast<torq_hl::AddOp>(consumer)) {
                if (!addOp.getRhsIsScalar()) {
                    return Rejection{"add", "consumer is a two-tensor torq_hl.add"};
                }
            }

            // torq_hl.elementwisebinary and torq_hl.elementwiseshift read their operands the same
            // way, so they need the same thing. Neither of them has the scalar path that add has.
            // Both stop with "Input strides must match" -- see ElementWiseBinaryPattern.cpp and
            // ElementWiseShiftPattern.cpp. The binary one also compares the inputs against the
            // output ("Input and output strides must match"). So slicing both operands the same
            // way is not enough, because the init is still dense.
            //
            // This really happens. An i16 tosa.clamp lowers its min side to elementwisebinary
            // MAXIMUM and hits it (test_keras_ops.py, model004_conv_3x3_valid_bias_int16).
            if (isa<torq_hl::ElementWiseBinaryOp, torq_hl::ElementWiseShiftOp>(consumer)) {
                return Rejection{"elementwise", "consumer needs its operands to share strides"};
            }

            // torq_hl.call_program runs an outlined program, and its code lives in DTCM, which is
            // small. Handing it a subview instead of a dense buffer makes that code bigger. Two
            // keras_ops depthwise int8 cases went a few hundred bytes over the DTCM limit this way
            // (dw023_NNR_301..., dw025_NNR_301..., both reported as "Failed to allocate DTCM
            // addresses"). Those cases were hidden behind an xfail until it was lifted. The conv
            // blocks this pattern targets never feed a call_program, so skipping these costs
            // nothing there.
            if (isa<torq_hl::CallProgramOp>(consumer)) {
                return Rejection{"call_program", "consumer is a torq_hl.call_program"};
            }

            // look through converts: the kernel is behind the convert back to LRAM
            if (isa<torq_hl::ConvertOp>(consumer)) {
                for (Value res : consumer->getResults()) {
                    worklist.append(res.getUsers().begin(), res.getUsers().end());
                }
            }
        }

        return std::nullopt;
    }

    // Without the swap the consumer reads a small dense buffer, the size of the slice, and the
    // parent can move to XRAM and be freed. With the swap the consumer reads a subview, so the
    // whole parent has to stay live while the kernel runs. The allocator counts an alias at the
    // size of its root. That is on purpose, and a comment in VirtualMemory.cpp explains why.
    //
    // So the swap trades DMA for `parent - slice` bytes of peak LRAM. This is a good trade for a
    // halo slab, where the parent is only a little bigger than the slice. It is a bad trade for a
    // channel cut, where the parent is several times bigger. With too many bad trades the
    // allocator runs out of room and fails with "unable to free enough space for results and
    // operands".
    std::optional<Rejection>
    checkResidency(torq_hl::ConvertOp op, tensor::ExtractSliceOp extractOp) const {

        auto srcTy = dyn_cast<RankedTensorType>(op.getInput().getType());
        auto sliceTy = dyn_cast<RankedTensorType>(extractOp.getResult().getType());
        if (!srcTy || !sliceTy || !srcTy.hasStaticShape() || !sliceTy.hasStaticShape()) {
            return Rejection{"dynamic shape", "dynamic shape, cannot size the residency"};
        }

        int64_t elemBytes = llvm::divideCeil(srcTy.getElementTypeBitWidth(), 8);
        int64_t parentBytes = srcTy.getNumElements() * elemBytes;
        int64_t sliceBytes = sliceTy.getNumElements() * elemBytes;
        int64_t extraBytes = parentBytes - sliceBytes;
        int64_t budget = residencyBudget();

        LLVM_DEBUG(
            llvm::dbgs() << "SwapExtractAndConvert: residency parent=" << parentBytes << " slice="
                         << sliceBytes << " extra=" << extraBytes << " budget=" << budget
                         << " lram=" << lramSize_ << " type=" << srcTy << "\n"
        );

        if (extraBytes > budget) {
            return Rejection{"residency", "parent would pin too much extra LRAM"};
        }

        return std::nullopt;
    }

    // How many bytes of extra LRAM one swap may pin.
    //
    // The budget is a share of LRAM. A chip with less LRAM gets a smaller share.
    //
    // A model does not get smaller when the memory does. It runs the same layers in half the room.
    // So it already sits closer to the ceiling. It has less room to give away. The same share is
    // too much there. Two int8 layer models ran out of LRAM on a 256 KB chip. Both were fine on a
    // 512 KB chip.
    //
    // So we scale the share by this LRAM size over the tuned one, squared. One factor was not
    // enough. Half the LRAM then gets a quarter of the share. The budget itself drops to an
    // eighth, because that quarter is a share of an LRAM that is half as big. On the two chips we
    // have that is 12% of 512 KB and 3% of 256 KB, so 62 KB against 8 KB.
    //
    // A chip with the tuned size or more keeps the plain percentage.
    int64_t residencyBudget() const {

        int64_t budget = (lramSize_ * clSwapExtractAndConvertMaxWastePercent) / 100;

        // The ratio is at most one here, so this cannot grow the budget. It also cannot overflow:
        // the guard keeps lramSize_ under the tuned size, and the tuned size cubed still fits in
        // an int64_t.
        if (lramSize_ < kSwapExtractAndConvertTunedLramSize) {
            budget = (budget * lramSize_ * lramSize_) /
                     (kSwapExtractAndConvertTunedLramSize * kSwapExtractAndConvertTunedLramSize);
        }

        return budget;
    }

    int64_t lramSize_;
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
    KeepConcatInLram(MLIRContext *ctx, int64_t lramSize)
        : OpRewritePattern(ctx), lramSize_(lramSize) {}

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

            // Do not pull into LRAM if the source is already LRAM-encoded or
            // comes from a ConvertOp that decodes from LRAM; the kernel
            // already manages its own LRAM allocation for the tile.
            if (getEncodingMemorySpace(currentInsert.getSource().getType()) !=
                torq_hl::MemorySpace::Xram) {
                return failure();
            }
            if (auto convertOp = currentInsert.getSource().getDefiningOp<torq_hl::ConvertOp>()) {
                if (getEncodingMemorySpace(convertOp.getInput().getType()) ==
                    torq_hl::MemorySpace::Lram) {
                    return failure();
                }
            }

            // we cannot fit the whole contact in LRAM so we won't apply this pattern
            if (totalSize > lramSize_) {
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

        auto newDestValue = tensor::EmptyOp::create(
                                rewriter, rootOp.getLoc(), destType.getShape(),
                                destType.getElementType(), newDestEncoding
        )
                                .getResult();

        for (auto insertOp : llvm::reverse(insertOps)) {

            // convert the source to LRAM
            auto sourceType = insertOp.getSource().getType();
            auto lramEncoding = createDenseEncoding(sourceType, torq_hl::MemorySpace::Lram);
            auto newSource = convertTensorToEncoding(rewriter, insertOp.getSource(), lramEncoding);

            // create a new insert slice op that works in LRAM
            auto newInsertOp = tensor::InsertSliceOp::create(
                rewriter, insertOp.getLoc(), newSource, newDestValue, insertOp.getMixedOffsets(),
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

  private:
    int64_t lramSize_;
};

// Fold duplicate conversions where two convert ops convert the same input tensor
// %extracted_slice_38 = tensor.extract_slice
// %123 = torq_hl.convert out(%122 : tensor<1x1x89928xbf16, #torq_hl<enc mem_space = lram>>)
// in(%extracted_slice_38 : tensor<1x1x89928xbf16>) %125 = torq_hl.convert out(%124 :
// tensor<1x1x89928xbf16, #torq_hl<enc mem_space = lram>>) in(%extracted_slice_38 :
// tensor<1x1x89928xbf16>) %126 = "torq_hl.mul"(%119, %121, %123, %125)
// --> fold to %126 = "torq_hl.mul"(%119, %121, %123, %123)
// to reduce lram usage

class FoldDuplicateConversion : public OpRewritePattern<torq_hl::ConvertOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(torq_hl::ConvertOp op, PatternRewriter &rewriter) const override {

        Value currentOutput = op.getOutput();

        auto inEncoding = getEncoding(op.getInput().getType());
        auto outEncoding = getEncoding(op.getOutput().getType());

        if (inEncoding.getMemSpace() != torq_hl::MemorySpace::Xram &&
            outEncoding.getMemSpace() != torq_hl::MemorySpace::Lram) {
            return failure();
        }

        for (auto use : currentOutput.getUsers()) {

            Value lhs, rhs;

            if (isa<torq_hl::MulOp>(use)) {
                lhs = dyn_cast<torq_hl::MulOp>(use).getInput1();
                rhs = dyn_cast<torq_hl::MulOp>(use).getInput2();
            }
            else if (isa<torq_hl::AddOp>(use)) {
                lhs = dyn_cast<torq_hl::AddOp>(use).getInput1();
                rhs = dyn_cast<torq_hl::AddOp>(use).getInput2();
            }
            else if (isa<torq_hl::ElementWiseBinaryOp>(use)) {
                lhs = dyn_cast<torq_hl::ElementWiseBinaryOp>(use).getInput1();
                rhs = dyn_cast<torq_hl::ElementWiseBinaryOp>(use).getInput2();
            }
            else {
                continue;
            }

            // if both lhs and rhs defining Op are convertOp
            // check if two convertOp input is the same tensor
            // if so, we can replace one of the convertOp output with the other
            auto lhsConvertOp = dyn_cast<torq_hl::ConvertOp>(lhs.getDefiningOp());
            auto rhsConvertOp = dyn_cast<torq_hl::ConvertOp>(rhs.getDefiningOp());

            if (lhsConvertOp && rhsConvertOp) {
                if (lhsConvertOp == rhsConvertOp) {
                    return failure();
                }

                Value lhsInput = lhsConvertOp.getInput();
                Value rhsInput = rhsConvertOp.getInput();

                if (lhsInput == rhsInput) {

                    if (lhsConvertOp == op) {
                        rhsConvertOp.getOutput().replaceAllUsesWith(lhsConvertOp.getOutput());
                    }
                    else {
                        lhsConvertOp.getOutput().replaceAllUsesWith(rhsConvertOp.getOutput());
                    }

                    return success();
                }
            }
        }
        return failure();
    }
};

class FoldConvertPass : public impl::FoldConvertBase<FoldConvertPass> {
  public:
    using FoldConvertBase<FoldConvertPass>::FoldConvertBase;
    FoldConvertPass(const FoldConvertOptions &options) { this->lramSize = options.lramSize; }
    FoldConvertPass(const FoldConvertPass &pass) { this->lramSize = pass.lramSize; }
    void runOnOperation() override;
};

void FoldConvertPass::runOnOperation() {
    assert(this->lramSize > 0 && "LRAM size must be greater than 0");
    MLIRContext *ctx = getOperation().getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<FoldConvertChainWithKernel>(ctx);
    patterns.add<FoldRoundTripConversion>(ctx);
    patterns.add<FoldLramToLramConversionChain>(ctx);
    patterns.add<KeepConcatInLram>(ctx, this->lramSize);
    patterns.add<FoldDuplicateConversion>(ctx);

    // This pattern used to be disabled here with `#if 0`. The reason given at that time was:
    //
    //   This pattern creates a situation where elementwise ops have inputs with different
    //   encodings (strides), which is not supported yet.
    //
    // Three kernels need matching strides: a two-tensor torq_hl.add, torq_hl.elementwisebinary
    // and torq_hl.elementwiseshift. The pattern now skips the rewrite when one of them consumes
    // the slice. A second guard limits the extra LRAM the swap keeps live, which is what broke
    // efficientnetb0. Both guards are inside the pattern, each with its own comment.
    patterns.add<SwapExtractAndConvert>(ctx, this->lramSize);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createFoldConvertPass(int64_t lramSize) {
    return std::make_unique<FoldConvertPass>(FoldConvertOptions{lramSize});
}

} // namespace mlir::syna::torq
