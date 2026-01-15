// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"
#include "llvm/Support/Debug.h"
#include <cstring>

#define DEBUG_TYPE "torq-lower-torqhl"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

namespace {

// Helper to convert DType to string for diagnostics
std::string dtypeToString(DType dtype) {
    std::string result;
    llvm::raw_string_ostream os(result);
    os << dtype;
    return os.str();
}

// Helper to validate float type consistency for operations requiring matching input/output types
LogicalResult validateFloatTypeConsistency(
    torq_hl::ActOp op, DType inputDType, DType outputDType, llvm::StringRef opName
) {
    if (!isFloat(inputDType)) {
        return op.emitError() << "Input element type for " << opName
                              << " must be float, got: " << dtypeToString(inputDType);
    }
    if (inputDType != outputDType) {
        return op.emitError() << "Input and output types for " << opName
                              << " must match: " << dtypeToString(inputDType) << " vs "
                              << dtypeToString(outputDType);
    }
    return success();
}

// Configure activation mode and clip ranges based on operation type
struct ActConfig {
    torq_hw::ACTMode mode;
    int32_t clipMin;
    int32_t clipMax;
};

FailureOr<ActConfig> configureActivation(
    torq_hl::ActOp op, DType inputDType, DType outputDType, Type outputElementType
) {
    ActConfig config;
    auto opName = op.getName().str();

    // Initialize default clip range based on output type
    auto setClipRange = [&config](auto type) {
        using T = decltype(type);
        config.clipMin = std::numeric_limits<T>::min();
        config.clipMax = std::numeric_limits<T>::max();
    };

    if (outputElementType.isInteger(1)) {
        setClipRange(bool{});
    }
    else if (isInt(outputDType)) {
        switch (outputDType) {
        case DType::int32:
            setClipRange(int32_t{});
            break;
        case DType::int16:
            setClipRange(int16_t{});
            break;
        case DType::int8:
            setClipRange(int8_t{});
            break;
        default:
            return op.emitError() << "Unsupported integer type: " << dtypeToString(outputDType);
        }
    }
    else if (isFloat(outputDType)) {
        config.clipMin = 0xff800000; // -inf for float32
        config.clipMax = 0x7f800000; // +inf for float32
    }
    else {
        return op.emitError() << "Unsupported output element type: " << dtypeToString(outputDType);
    }

    // Set activation mode and override clip ranges if needed
    if (opName == "abs") {
        config.mode = torq_hw::ACTMode::ABS;
    }
    else if (opName == "negate") {
        config.mode = torq_hw::ACTMode::NEG;
    }
    else if (opName == "clz") {
        config.mode = torq_hw::ACTMode::CLZ;
    }
    else if (opName == "ceil") {
        config.mode = torq_hw::ACTMode::CEL;
    }
    else if (opName == "floor") {
        config.mode = torq_hw::ACTMode::FLR;
    }
    else if (opName == "i2f") {
        config.mode = torq_hw::ACTMode::I2F;
    }
    else if (opName == "f2i") {
        config.mode = torq_hw::ACTMode::F2I;
        config.clipMin = 0xff800000;
        config.clipMax = 0x7f800000;
    }
    else if (opName == "i2i") {
        config.mode = torq_hw::ACTMode::ACT;
        // We should not use int8 range here as we want to truncate without clipping
        setClipRange(int32_t{});
    }
    else if (opName == "f2f") {
        config.mode = torq_hw::ACTMode::ACT;
    }

    else if (opName == "clamp") {
        config.mode = torq_hw::ACTMode::ACT;
        if (isInt(inputDType)) {
            config.clipMin = op.getMinInt();
            config.clipMax = op.getMaxInt();
        }
        else if (isFloat(inputDType)) {
            float minFp = op.getMinFp().convertToFloat();
            float maxFp = op.getMaxFp().convertToFloat();
            std::memcpy(&config.clipMin, &minFp, sizeof(float));
            std::memcpy(&config.clipMax, &maxFp, sizeof(float));
        }
    }
    else {
        config.mode = torq_hw::ACTMode::ACT;
    }

    return config;
}

} // anonymous namespace

template <>
LogicalResult ActPattern::transform(torq_hl::ActOp op, PatternRewriter &rewriter) const {
    auto ctx = op.getContext();
    LData input(op.getInput());
    LData output(op.getInit());
    DType inputDType = input.elementType();
    DType outputDType = output.elementType();

    // Setup input dimensions and vectorization
    struct In : Vectorized {
        enum { NonDenseDims };
    };

    auto opName = op.getName().str();
    Slice slice(std::string("Act-" + opName));
    int vectorSize = std::min(slice.act.width(inputDType), slice.act.width(outputDType));
    input.fuse(std::min(input.denseDims(), output.denseDims())).vectorize(vectorSize);

    // Configure activation mode and clip ranges
    auto outputElementType = llvm::cast<MemRefType>(op.getInit().getType()).getElementType();
    FailureOr<ActConfig> configResult =
        configureActivation(op, inputDType, outputDType, outputElementType);
    if (failed(configResult))
        return failure();
    ActConfig config = *configResult;

    if (opName == "ceil" || opName == "floor") {
        if (failed(validateFloatTypeConsistency(op, inputDType, outputDType, opName)))
            return failure();
    }

    // Generate hardware kernel
    For(auto ndd = slice.iterate(input.dims(In::NonDenseDims, In::Vectors))) {
        For(auto iv = slice.iterate(input.dim(In::Vectors))) {
            IData idata = slice.iram.load(input[ndd][iv]);
            PData pdata = slice.alu.load(idata);
            QData res = slice.act.clamp(pdata, config.clipMin, config.clipMax, config.mode);
            slice.append(output[ndd], res);
        }
    }

    // Replace with hardware operation
    rewriter.replaceOpWithNewOp<SliceTaskOp>(
        op, op.getName(), ValueRange{op.getInput()}, ValueRange{}, ValueRange{},
        ValueRange{op.getInit()}, ValueRange{}, slice.getCfgAttr(ctx), slice.getNdls()
    );

    LLVM_DEBUG(llvm::dbgs() << "Successfully lowered " << opName << " to ActHwPattern\n");
    return success();
}

} // namespace mlir::syna::torq
