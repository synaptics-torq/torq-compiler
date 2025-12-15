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
    return {torq_hl::MemorySpace::Lram, reqs.stridesAlign, reqs.paddingAlign};
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

    auto outType = cast<RankedTensorType>(op->getResult(0).getType());

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
                rewriter.create<tensor::EmptyOp>(op->getLoc(), encodedOutType, ValueRange{});
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

        if (checkTypeMatchesEncodingRequirements(inType, inEncoding)) {
            operandEncodings.push_back(getEncoding(inType));
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

static KernelTensorEncoding pad256Encoding() { return {{}, 256}; }

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
    // We can't check the output tensor as it is not encoded (or we could add identity encoding)
    RankedTensorType inputType = dyn_cast<RankedTensorType>(getInput().getType());

    KernelTensorEncoding inEncodingAttr;
    KernelTensorEncoding outEncodingAttr;

    // optimized transpose requires alignment
    if (torq::supportedByOptimizedTranspose(getPerm())) {

        auto outType = dyn_cast<RankedTensorType>(getOutput().getType());
        auto outShape = outType.getShape();
        int rank = outShape.size();

        if (rank < 2) {
            llvm::report_fatal_error("Transpose rank must be at least 2", true);
        }

        int32_t rightDim = 1;
        int32_t leftDim = 0;
        int32_t transposeDim = -1;
        int32_t leftDimSize = 0;
        int32_t transposeDimSize = 0;
        int32_t rightDimSize = 0;

        torq::identifyTransposeDim(*this, outShape, transposeDim, leftDim, rightDim);
        LLVM_DEBUG({
            dump();
            llvm::dbgs() << "Identified (leftDim, transposeDim, rightDim) -> (" << leftDim << ","
                         << transposeDim << "," << rightDim << ")\n";
        });

        auto inputShape = inputType.getShape();
        torq::groupContinuousDim(
            inputShape, leftDim, transposeDim, rightDim, leftDimSize, transposeDimSize, rightDimSize
        );

        LLVM_DEBUG({
            llvm::dbgs() << "Identified (leftDimSize, transposeDimSize, rightDimSize) -> ("
                         << leftDimSize << "," << transposeDimSize << "," << rightDimSize << ")\n";
        });

        outEncodingAttr.stridesAlign.resize(rank, 0);
        outEncodingAttr.stridesAlign[leftDim] =
            torq::align_ceil(rightDimSize, 32) * torq::align_ceil(transposeDimSize, 32);

        inEncodingAttr.stridesAlign.resize(rank, 0);

        if (transposeDim > 0) {
            inEncodingAttr.stridesAlign[transposeDim - 1] = 64;
        }
        else {
            inEncodingAttr.paddingAlign = 64;
        }
    }

    return {{{getInputMutable().getOperandNumber(), inEncodingAttr}}, outEncodingAttr};
}

static KernelEncoding getDenseEncoding() { return {{}, {{}, 0, true}}; }
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
static KernelEncoding getPaddedOutEncoding(Operation *op) { return {{}, pad256Encoding()}; }
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

KernelEncoding AddOp::getKernelEncoding() {
    RankedTensorType resultType = dyn_cast<RankedTensorType>(getResult(0).getType());

    auto input1Type = cast<RankedTensorType>(getInput1().getType());
    auto input2Type = cast<RankedTensorType>(getInput2().getType());

    assert(
        input1Type.getRank() == resultType.getRank() && "Input and output tensor ranks must match"
    );

    auto enc1 = defaultEncoding(input1Type);
    auto enc2 = defaultEncoding(input2Type);
    auto enc = defaultEncoding(resultType, 0, !getSegmentOutput());

    SmallVector<KernelInputEncoding> inEncoding = {
        {getInput1Mutable().getOperandNumber(), enc1}, {getInput2Mutable().getOperandNumber(), enc2}
    };

    SmallVector<std::pair<unsigned, unsigned>> equalEncodingOperands = {
        {getInput1Mutable().getOperandNumber(), getInput2Mutable().getOperandNumber()}
    };

    return {inEncoding, enc, equalEncodingOperands};
}

KernelEncoding TableOp::getKernelEncoding() {
    // The table kernel writes 64 bytes of data. To handle cases where the input is not aligned
    // to 64 bytes, the output tensor size should be align to 64 bytes.
    return {{}, {{}, torq::HwInfo::max_input}};
}

KernelEncoding ReduceOp::getKernelEncoding() {
    // The reduce kernel writes 64 elements of data. The output tensor size should be align to 64
    // elements.
    return {{}, {{}, torq::HwInfo::max_input}};
}

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

KernelEncoding MulOp::getKernelEncoding() {

    const int inIndex = getInput1Mutable().getOperandNumber();
    const int inIndex2 = getInput2Mutable().getOperandNumber();

    int inRank = cast<RankedTensorType>(getOperand(inIndex).getType()).getRank();
    int inRank2 = cast<RankedTensorType>(getOperand(inIndex2).getType()).getRank();
    // if any input is scalar, no encoding required
    if (inRank == 1 || inRank2 == 1) {
        return getNoEncoding();
    }

    // FIXME: correct encoding
    // right now this configs seems work for all the cases
    // but have to check later
    return {{}, {{}, 64}};

#if 0
    auto resultType = getInit().getType();
    int resultRank = resultType.getRank();
    const int inIndex = getInput1Mutable().getOperandNumber();

    int inRank = cast<RankedTensorType>(getOperand(inIndex).getType()).getRank();

    assert(inRank == resultRank && "Input and output tensor ranks must match");

    KernelTensorEncoding enc;
    enc.stridesAlign.resize(resultRank, 0);

    if (resultRank < 2) {
        enc.stridesAlign[0] = 32;
    }
    else {
        // Require alignment on the 2nd dimension of the tensor.
        // This also ensures the overall tensor size is aligned.
        enc.stridesAlign[1] = 32;
    }

    SmallVector<KernelInputEncoding> inEncoding = {
        {getInput1Mutable().getOperandNumber(), enc}, {getInput2Mutable().getOperandNumber(), enc}
    };

    return {inEncoding, enc};
#endif
}

KernelEncoding MatMulOp::getKernelEncoding() {

    // matmul op [batch, m, k] x [batch, k, n] -> [batch, m, n]

    // input1
    ShapedType inputType = getInput1().getType();
    int in1Rank = inputType.getRank();

    KernelInputEncoding in1Enc;
    in1Enc.opIndex = getInput1Mutable().getOperandNumber();
    in1Enc.encoding.stridesAlign.resize(in1Rank, 0);
    // if rank is 1, just make sure the overall tensor size is aligned
    // otherwise make sure the last dimension(k need to align to wram_width=32) is aligned to
    // get overall tensor size aligned
    if (in1Rank == 1) {
        in1Enc.encoding.paddingAlign = 32;
    }
    else {
        in1Enc.encoding.stridesAlign[in1Rank - 2] = 32;
    }

    // output encoding
    int in2Rank = getInput2().getType().getRank();

    KernelTensorEncoding outEnc;
    outEnc.stridesAlign.resize(in2Rank, 0);

    // if rank is 1, just make sure the overall tensor size is aligned
    // otherwise make sure the last dimension(n need to align to alu_group_width=64) is aligned
    // to get overall tensor size aligned
    if (in2Rank == 1) {
        outEnc.paddingAlign = 64;
    }
    else {
        outEnc.stridesAlign[in2Rank - 2] =
            64 / (inputType.getElementType().getIntOrFloatBitWidth() / 8);
    }

    // output
    auto resultType = getInit().getType();

    int resultRank = resultType.getRank();
    if (resultRank == 0) {
        outEnc.stridesAlign.resize(0);
        outEnc.paddingAlign = 1;
    }

    return {{in1Enc}, outEnc};
}

KernelEncoding TransposeReshapeOp::getKernelEncoding() {

    auto input_type = llvm::dyn_cast<RankedTensorType>(getInput().getType());
    auto inputShape = input_type.getShape();

    // Check for Conv1D shape: [1, 1, W, 1]
    if (inputShape.size() != 4 || inputShape[0] != 1 || inputShape[1] != 1 || inputShape[3] != 1) {
        return {{}, {}};
    }

    auto outputType = cast<RankedTensorType>(getOutput().getType());
    auto outShape = outputType.getShape();
    int rank = outShape.size();

    KernelTensorEncoding outEncoding;

    outEncoding.stridesAlign.resize(rank, 0);
    int align = torq::align_ceil(outShape[3], 32);
    outEncoding.stridesAlign[2] = align;
    outEncoding.stridesAlign[1] = align * torq::align_ceil(outShape[2], 32);

    return {{}, outEncoding};
}

KernelEncoding MaxPool2dOp::getKernelEncoding() { return getDefault1InputEncoding(*this); }

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
    return torq::hasEkLoweringConv(*this) ? getNoEncoding() : getConvLikeKernelEncoding(*this);
}

KernelEncoding DepthwiseConv2DOp::getKernelEncoding() {
    return torq::hasEkLoweringConv(*this) ? getNoEncoding() : getConvLikeKernelEncoding(*this);
}

KernelEncoding FullyConnectedOp::getKernelEncoding() { return getDefault1InputEncoding(*this); }

KernelEncoding Conv1DOp::getKernelEncoding() { return getPaddedOutEncoding(*this); }

KernelEncoding ActOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding ElementWiseUnaryOp::getKernelEncoding() { return getActLikeEncoding(*this); }

KernelEncoding FMAOp::getKernelEncoding() { return getActLikeEncoding(*this); }

KernelEncoding BroadcastOp::getKernelEncoding() { return getActLikeEncoding(*this); }

KernelEncoding ResizeNearestNeighborOp::getKernelEncoding() { return getResizeLikeEncoding(*this); }

KernelEncoding DepthToSpaceOp::getKernelEncoding() { return getResizeLikeEncoding(*this); }

KernelEncoding InterleavedInsertOp::getKernelEncoding() { return getNoEncoding(); }

KernelEncoding IdentityOp::getKernelEncoding() { return getDenseEncoding(); }

KernelEncoding ElementWiseBinaryOp::getKernelEncoding() { return getDenseTwoInputEncoding(*this); }

KernelEncoding ArgMaxOp::getKernelEncoding() { return getDenseEncoding(); }

KernelEncoding FillOp::getKernelEncoding() { return getDenseEncoding(); }

KernelEncoding ScatterOp::getKernelEncoding() { return getDenseEncoding(); }

KernelEncoding GatherOp::getKernelEncoding() { return getDenseEncoding(); }

KernelEncoding AvgPool2DOp::getKernelEncoding() { return {{}, {{}, 64}}; }

KernelEncoding ConvertOp::getKernelEncoding() { return getDenseEncoding(); }

KernelEncoding SelectOp::getKernelEncoding() { return getNoEncoding(); }

} // namespace mlir::syna::torq_hl
