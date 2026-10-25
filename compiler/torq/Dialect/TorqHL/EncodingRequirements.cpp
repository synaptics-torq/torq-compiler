#include "torq/Dialect/TorqHL/EncodingRequirements.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqUtils.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-encoding-requirements"

namespace mlir::syna::torq_hl {

EncodingRequirements toTensorEncodingRequirementsAttr(KernelTensorEncoding reqs) {
    return {
        torq_hl::MemorySpace::Lram, reqs.stridesAlign, reqs.paddingAlign, /* onlyDense = */ false,
        reqs.denseInnerDims
    };
}

KernelTensorEncoding getOperandEncoding(const KernelEncoding &encoding, OpOperand &opOperand) {

    auto dsOp = cast<DestinationStyleOpInterface>(opOperand.getOwner());

    if (&opOperand == dsOp.getDpsInitOperand(0)) {
        return encoding.outputEncoding;
    }

    for (const auto &input : encoding.inputEncodings) {
        if (input.opIndex == opOperand.getOperandNumber()) {
            return input.encoding;
        }
    }

    // if no encoding is specified then this means the kernel only accepts dense tensors
    return KernelTensorEncoding{{}, 0, true};
}

FailureOr<bool> encodeKernelInputOutputs(
    DestinationStyleOpInterface op, const KernelEncoding &encoding, RewriterBase &rewriter,
    Value initValue
) {

    if (op->getNumResults() != 1) {
        return op->emitError("operation must have exactly one result tensor");
    }

    bool changed = false;

    auto outType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!outType || !outType.hasStaticShape())
        return false;

    for (OpOperand &opOperand : op->getOpOperands()) {
        auto inType = dyn_cast<RankedTensorType>(opOperand.get().getType());
        if (inType && !inType.hasStaticShape())
            return false;
    }

    auto outReqs = toTensorEncodingRequirementsAttr(encoding.outputEncoding);

    // convert the output if it doesn't match encoding requirements
    if (!checkTypeMatchesEncodingRequirements(outType, outReqs)) {

        auto outEncodingAttr = createAlignedEncoding(outType, outReqs);
        auto encodedOutType = createRankedTensorTypeWithEncoding(outType, outEncodingAttr);

        // create a new init tensor with a suitable type
        rewriter.setInsertionPoint(op);

        Value newInitValue;
        // if the operation updates the init tensor instead of replacing it we need to convert this
        // to the expected encoding
        if (op.getOperation()->hasTrait<mlir::OpTrait::UpdateInPlaceTrait>()) {
            newInitValue = convertTensorToType(
                rewriter, cast<TypedValue<RankedTensorType>>(op.getDpsInitOperand(0)->get()),
                encodedOutType
            );
        }
        // otherwise we can simply create a new empty tensor
        else {
            newInitValue =
                tensor::EmptyOp::create(rewriter, op->getLoc(), encodedOutType, ValueRange{});
        }

        // replace the output type of the op with the encoded type and the input with the new init
        // tensor
        rewriter.modifyOpInPlace(op, [&] {
            op->getOpResult(0).setType(encodedOutType);
            op.getDpsInitOperand(0)->set(newInitValue);
        });

        // convert back the output to the original type
        rewriter.setInsertionPointAfter(op);
        auto convertedOutput = convertTensorToType(
            rewriter, cast<TypedValue<RankedTensorType>>(op->getResult(0)), outType, initValue
        );

        // ensure all previous users (therefore excluding the newly inserted convert op) use the
        // output of the conversion
        rewriter.replaceAllUsesExcept(
            op->getResult(0), convertedOutput, convertedOutput.getDefiningOp()
        );

        LLVM_DEBUG({
            llvm::dbgs() << "Encoded output of op:\n  ";
            op->dump();
            llvm::dbgs() << "  Replaced:\n  ";
            outType.dump();
            llvm::dbgs() << "  With:\n  ";
            encodedOutType.dump();
            llvm::dbgs() << "  to satisfy encoding requirements:\n  ";
            outReqs.toAttr(op.getContext()).dump();
        });

        changed = true;
    }

    SmallVector<EncodingRequirements> inEncodingsReqs;

    // find the encoding requirements for each operand
    for (auto &opOperand : op->getOpOperands()) {
        auto req = getOperandEncoding(encoding, opOperand);
        inEncodingsReqs.push_back(toTensorEncodingRequirementsAttr(req));
    }

    // find the encoding for each operand
    SmallVector<TensorEncodingAttr> operandEncodings;
    for (auto &opOperand : op->getOpOperands()) {
        auto inType = cast<RankedTensorType>(opOperand.get().getType());

        auto inEncoding = inEncodingsReqs[opOperand.getOperandNumber()];

        if (checkValueMatchesEncodingRequirements(opOperand.get(), inEncoding)) {
            operandEncodings.push_back(getEncoding(inType));
        }
        else if (checkTypeMatchesEncodingRequirements(inType, inEncoding)) {
            // the type is fine but the value is a slice that cuts inside a dense block: copy it
            // into a fresh dense buffer. Spell the natural strides out so the copy has a type of
            // its own and is not folded away as a no-op conversion.
            EncodingRequirements explicitEncoding = inEncoding;
            explicitEncoding.stridesAlign.assign(inType.getRank(), 0);
            operandEncodings.push_back(createAlignedEncoding(inType, explicitEncoding));
        }
        else {
            operandEncodings.push_back(createAlignedEncoding(inType, inEncoding));
        }
    }

    // ensure all operands that need to have the same encoding have it
    for (auto &eq : encoding.equalEncodingOperands) {

        auto lhsEnc = operandEncodings[eq.first];
        auto rhsEnc = operandEncodings[eq.second];

        if (lhsEnc == rhsEnc) {
            continue;
        }

        operandEncodings[eq.second] = lhsEnc;
    }

    // convert the operands to the required encoding if necessary
    for (auto &opOperand : op->getOpOperands()) {

        auto inType = cast<RankedTensorType>(opOperand.get().getType());

        if (operandEncodings[opOperand.getOperandNumber()] == getEncoding(inType)) {
            continue;
        }

        rewriter.setInsertionPoint(op);

        auto convertedInput = convertTensorToEncoding(
            rewriter, cast<TypedValue<RankedTensorType>>(opOperand.get()),
            operandEncodings[opOperand.getOperandNumber()]
        );

        rewriter.modifyOpInPlace(op, [&] {
            op->getOpOperand(opOperand.getOperandNumber()).set(convertedInput);
        });

        LLVM_DEBUG({
            llvm::dbgs() << "Encoded input #" << opOperand.getOperandNumber() << " of op:\n  ";
            op->dump();
            llvm::dbgs() << "  Replaced:\n  ";
            inType.dump();
            llvm::dbgs() << "  With:\n  ";
            convertedInput.getType().dump();
            llvm::dbgs() << "  to satisfy encoding requirements:\n  ";
            inEncodingsReqs[opOperand.getOperandNumber()].toAttr(op.getContext()).dump();
        });

        changed = true;
    }

    return changed;
}

static KernelTensorEncoding
defaultEncoding(ShapedType type, int channelAlign = 0, bool alignData = true) {
    int defaultByteAlign = 64;
    auto alignWithType =
        alignData ? getAlignmentByType(defaultByteAlign, type.getElementType()) : 0;
    if (type.getRank() == 4) {
        return {{-channelAlign, alignWithType, 0, 0}, 0};
    }
    else if (type.getRank() == 3) {
        return {{-channelAlign, alignWithType, 0}, 0};
    }

    // By default align the overall tensor size
    assert(channelAlign == 0);
    SmallVector<int64_t> align(type.getRank(), 0);

    return {align, alignWithType};
}

template <typename ConvT> static KernelEncoding getConvLikeKernelEncoding(ConvT op) {

    VectorizationModeEnum mode = op.getVectorizationMode();

    if (mode == torq_hl::VectorizationModeEnum::None) {
        // no encoding when vectorization is not selected yet
        return {{}, {{}, 0}};
    }

    if (mode == torq_hl::VectorizationModeEnum::_32x32) {
        return {{}, {{0, 32, 0, 0}, 0}};
    }

    const int parallelOuts = static_cast<int>(mode);

    return {{}, defaultEncoding(op.getInit().getType(), parallelOuts, !op.getSegmentOutput())};
}

KernelEncoding TransposeOp::getKernelEncoding() {
    // The block transpose only reads and writes inside its input and output (TransposePattern.cpp
    // transposes the rows and cols that don't fill a block with the generalized transpose), so the
    // transposes only need the minimal output alignment
    KernelTensorEncoding outEncodingAttr;
    outEncodingAttr.paddingAlign = 64;
    return {{}, outEncodingAttr};
}

static KernelEncoding getDenseEncoding() { return {{}, {{}, 64, true}}; }
static KernelEncoding getNoEncoding() { return {{}, {{}, 0}}; }

template <typename OpT> static KernelEncoding getDenseTwoInputEncoding(OpT op) {
    auto req = getDenseEncoding();

    req.equalEncodingOperands = {
        {op.getInput1Mutable().getOperandNumber(), op.getInput2Mutable().getOperandNumber()}
    };

    /// add padding for small outputs
    constexpr size_t KERNEL_W_SZ = 64;
    const RankedTensorType resultType = cast<RankedTensorType>(op->getResult(0).getType());
    const auto numElements = resultType.getNumElements();
    assert(resultType.hasStaticShape() && "Output 0 has non-static shape");
    assert(numElements != ShapedType::kDynamic && "Output 0 has a dynamic number of elements");
    const auto outSizeBytes = numElements * getScalarSizeBytes(resultType.getElementType());
    if (outSizeBytes < static_cast<int64_t>(KERNEL_W_SZ)) {
        req.outputEncoding.paddingAlign = KERNEL_W_SZ;
    }

#if 0
    // Actually this method is not requiring the two inputs to be dense, just to have the same enc.
    // But this request doesn't seem to work. We could try to explicitly requirign each input
    // to be dense but this also doesn't work (nobody checks the denseOnly flag).
    // So let's keep this disabled for now, all this can be removed once we have proper support
    // for elementwise ops with different input strides.
    KernelTensorEncoding denseTensorEncoding = {{}, 0, true};
    req.inputEncodings = {
        {op.getInput1Mutable().getOperandNumber(), denseTensorEncoding},
        {op.getInput2Mutable().getOperandNumber(), denseTensorEncoding}
    };
#endif
    return req;
}

static KernelEncoding getDefault1InputEncoding(Operation *op) {
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    return {{}, defaultEncoding(resultType)};
}
template <typename OpT> static KernelEncoding getResizeLikeEncoding(OpT op) {
    RankedTensorType resultType = cast<RankedTensorType>(op->getResult(0).getType());

    int resultRank = resultType.getShape().size();

    assert(resultRank == 4);

    int inIndex = op.getInputMutable().getOperandNumber();

    auto inputType = cast<RankedTensorType>(op->getOperand(inIndex).getType());

    auto outShape = resultType.getShape();

    int shapeAlign =
        outShape[2] *
        (torq::align_ceil(outShape[3], 32)
        ); // Currently resize processes each row with 32-byte alignment for the last
           // dimension to ensure proper memory alignment and vectorization efficiency.
           // TODO: Can later optimize alignment size if hardware supports different values

    LLVM_DEBUG({
        llvm::dbgs() << "encodeResize: outShape " << outShape[1] << "x" << outShape[2] << "x"
                     << outShape[3] << ", shapeAlign " << shapeAlign << "\n";
    });

    if ((outShape[1] == 512 && outShape[2] == 20 && outShape[3] == 20) ||
        (outShape[1] == 128 && outShape[2] == 40 && outShape[3] == 40)) {
        // FIXME: above alignment configuration causes memory usage explosion
        // We use 128 as a workaround for YoloV8-bodypose and yolov8 od model compilation
        shapeAlign = 128;
        llvm::dbgs() << "Using workaround shapeAlign " << shapeAlign << "\n";
    }

    auto inEnc = defaultEncoding(inputType);
    KernelTensorEncoding outEnc{{0, shapeAlign, 0, 0}, 0};

    return {{{op.getInputMutable().getOperandNumber(), inEnc}}, outEnc};
}

KernelEncoding AddOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding TableOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding ReduceOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding DeriveQuantParamsOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding ElementWiseShiftOp::getKernelEncoding() {
    return {
        {},
        {{}, torq::HwInfo::max_input, true},
        {{getInput1Mutable().getOperandNumber(), getInput2Mutable().getOperandNumber()}}
    };
}

template <typename ActOp> static KernelEncoding getActLikeEncoding(ActOp op) {

#if 0
    // FIXME: this must be enabled (currently disabled as it crashes FoldUselessConvertPattern)
    auto inputEnc = denseEncoding(rewriter.getContext(), resultType.getShape());
    inEncoding = {
        {op.getInputMutable().getOperandNumber(), inputEnc},
    };
#endif

    // Just adds padding to the size of the out vector.
    return {{}, {{}, 64}};
}

KernelEncoding MulOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding MatMulOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding ZeroPointBiasTableOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding MixedQuantMatMulOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding ZeroPointCorrectOp::getKernelEncoding() { return getNoEncoding(); }

// The NCHW conv, depthwise conv and max pool kernels fuse H and W into one scan, so the H-W
// plane of the input and of the output must be dense: the rows of a W-cut slice do not sit
// next to each other in memory
template <typename OpT> static KernelEncoding withDenseHWPlane(OpT op, KernelEncoding encoding) {
    KernelTensorEncoding inputEncoding;
    inputEncoding.denseInnerDims = 2;
    encoding.inputEncodings.push_back({op.getInputMutable().getOperandNumber(), inputEncoding});
    encoding.outputEncoding.denseInnerDims = 2;
    return encoding;
}

KernelEncoding MaxPool2dOp::getKernelEncoding() {
    if (torq::hasEkLoweringMaxPool(*this)) {
        return withDenseHWPlane(*this, getDefault1InputEncoding(*this));
    }

    auto resultType = cast<RankedTensorType>(this->getResult(0).getType());
    return withDenseHWPlane(*this, {{}, defaultEncoding(resultType, 4)});
}

KernelEncoding SegmentationOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding ReduceMeanOp::getKernelEncoding() {
    auto resultType = getInit().getType();
    int resultRank = resultType.getRank();

    assert(resultRank == 4);

    auto inputType = cast<RankedTensorType>(getInput().getType());

    auto outputType = cast<RankedTensorType>(getOutput().getType());
    auto outShape = outputType.getShape();
    int rank = outShape.size();

    // Output encoding: align dimension 3 to 32-byte boundaries
    KernelTensorEncoding outEnc;
    outEnc.stridesAlign.resize(rank, 0);
    int align = torq::align_ceil(outShape[3], 32);
    outEnc.stridesAlign[2] = align;
    outEnc.paddingAlign = 0;

    int inRank = inputType.getRank();

    KernelInputEncoding inEnc;
    inEnc.opIndex = getInputMutable().getOperandNumber();
    inEnc.encoding.stridesAlign.resize(inRank, 0);

    return {{inEnc}, outEnc};
}

KernelEncoding Conv2DOp::getKernelEncoding() {
    auto encoding =
        torq::hasEkLoweringConv(*this) ? getNoEncoding() : getConvLikeKernelEncoding(*this);
    return getNhwcInput() ? encoding : withDenseHWPlane(*this, encoding);
}

KernelEncoding DepthwiseConv2DOp::getKernelEncoding() {
    auto encoding =
        torq::hasEkLoweringConv(*this) ? getNoEncoding() : getConvLikeKernelEncoding(*this);
    return getNhwcInput() || getIsDw1dStride1() ? encoding : withDenseHWPlane(*this, encoding);
}

KernelEncoding FullyConnectedOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding ActOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding ElementWiseUnaryOp::getKernelEncoding() { return getActLikeEncoding(*this); }

KernelEncoding FMAOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding ExpandWeightsOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding BroadcastOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding ResizeNearestNeighborOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding DepthToSpaceOp::getKernelEncoding() { return getResizeLikeEncoding(*this); }

KernelEncoding InterleavedInsertOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding IdentityOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding ElementWiseBinaryOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding ArgMaxOp::getKernelEncoding() { return getDenseEncoding(); }

KernelEncoding FillOp::getKernelEncoding() { return getDenseEncoding(); }

KernelEncoding ScatterOp::getKernelEncoding() { return getDenseEncoding(); }

KernelEncoding GatherOp::getKernelEncoding() { return getDenseEncoding(); }

KernelEncoding Im2ColOp::getKernelEncoding() { return getDenseEncoding(); }

KernelEncoding AvgPool2DOp::getKernelEncoding() { return {{}, {{}, 64}}; }

KernelEncoding ConvertOp::getKernelEncoding() { return getDenseEncoding(); }

KernelEncoding SelectOp::getKernelEncoding() { return getNoEncoding(); }

} // namespace mlir::syna::torq_hl
