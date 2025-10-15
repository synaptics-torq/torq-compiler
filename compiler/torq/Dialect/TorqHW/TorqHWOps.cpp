// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Dialect/TorqHW/TorqHWOps.h"
#include "torq/Dialect/TorqHW/TorqHWDialect.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
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

#include "llvm/Support/CommandLine.h"
#include <numeric>

#define GET_OP_CLASSES
#include "torq/Dialect/TorqHW/TorqHWOps.cpp.inc"

using namespace mlir;
using namespace mlir::syna::torq_hw;

#define DEBUG_TYPE "torqhw-ops"

namespace mlir::syna::torq_hw {

static llvm::cl::opt<bool> clEnableDeqwVerify(
    "torq-enable-deqw-verify", llvm::cl::desc("Enable bound checking for DEQW"),
    llvm::cl::init(true)
);

void SliceTaskOp::build(
    ::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef op_name, Value d,
    Value w, Value b, Value q, SliceCFGAttr slice_cfg_attr, Ndls ndls
) {
    build(
        odsBuilder, odsState, op_name, d ? ValueRange{d} : ValueRange{},
        w ? ValueRange{w} : ValueRange{}, b ? ValueRange{b} : ValueRange{},
        q ? ValueRange{q} : ValueRange{}, ValueRange{}, ValueRange{}, ValueRange{}, slice_cfg_attr,
        ndls
    );
}

void SliceTaskOp::build(
    ::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef op_name,
    ValueRange d, ValueRange w, ValueRange b, ValueRange q, ValueRange symbols,
    SliceCFGAttr slice_cfg_attr, Ndls ndls
) {
    build(
        odsBuilder, odsState, op_name, d, w, b, q, ValueRange{}, ValueRange{}, symbols,
        slice_cfg_attr, ndls
    );
}

void SliceTaskOp::build(
    ::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef op_name,
    ValueRange d, ValueRange w, ValueRange b, ValueRange q, ValueRange dx, ValueRange bx,
    ValueRange symbols, SliceCFGAttr slice_cfg_attr, Ndls ndls
) {

    auto numSyms = symbols.size();
    auto context = odsBuilder.getContext();

    SmallVector<MemNdlAttr> memNdls;
    for (auto &ndl : ndls.memNdls) {
        memNdls.push_back(MemNdlAttr::get(context, ndl, numSyms));
    }
    SmallVector<RegNdlAttr> regNdls;
    for (auto &ndl : ndls.regNdls) {
        regNdls.push_back(RegNdlAttr::get(context, ndl));
    }

    build(
        odsBuilder, odsState, op_name, d, w, b, q, dx, bx, symbols, slice_cfg_attr, memNdls, regNdls
    );
}

/// Checks if a given offset is included in the memory represented by a memref,
/// considering only the actual data elements and iterating over dimensions.
static bool isOffsetInMemRefData(
    int64_t relativeOffset, const ArrayRef<int64_t> &shape, const ArrayRef<int64_t> &alignedStrides,
    const ArrayRef<int64_t> &memrefStrides
) {

    if (relativeOffset < 0) {
        llvm::errs() << "Offset is negative: " << relativeOffset << "\n";
        return false;
    }

    // Iterate over each dimension to check if the offset is valid.
    for (size_t i = 0; i < shape.size(); ++i) {
        assert(shape[i] != ShapedType::kDynamic && "Dynamic shapes are not supported");

        // If the stride is zero, skip this dimension (it doesn't contribute to the offset).
        if (memrefStrides[i] == 0) {
            continue;
        }

        // Compute the index for this dimension.
        int64_t index = relativeOffset / memrefStrides[i];
        int64_t offsetInDimension = relativeOffset % memrefStrides[i];
        if (offsetInDimension > alignedStrides[i]) {
            // The offset is beyond the allowed padded area, so it's invalid.
            // llvm::errs() << "Offset is beyond padded area for dimension " << i << "  : " <<
            // offsetInDimension << "\n";
            return false;
        }

        // Subtract the contribution of this dimension from the relative offset.
        relativeOffset -= index * memrefStrides[i];
    }

    // If we reach here, the offset is valid.
    assert(relativeOffset == 0);
    return true;
}

// Group elements in a vector into contiguous ranges.
static std::vector<std::pair<int, int>> groupIntoRanges(const std::vector<int> &input) {
    if (input.empty()) {
        return {};
    }

    // Sort the input to ensure elements are in order.
    std::vector<int> sortedInput = input;
    sort(sortedInput.begin(), sortedInput.end());

    std::vector<std::pair<int, int>> ranges;
    int start = sortedInput[0];
    int end = sortedInput[0];

    for (size_t i = 1; i < sortedInput.size(); ++i) {
        if (sortedInput[i] == end + 1) {
            // Extend the current range.
            end = sortedInput[i];
        }
        else {
            // Close the current range and start a new one.
            ranges.emplace_back(start, end);
            start = sortedInput[i];
            end = sortedInput[i];
        }
    }

    // Add the last range.
    ranges.emplace_back(start, end);

    return ranges;
}

// Increment the indices for the next element in a multi-dimensional array.
// return false if wrapped around the end of count on all the dimensions
static bool incrementIndex(SmallVector<int> &indices, const SmallVector<int64_t> &dimCount) {
    size_t rank = dimCount.size();
    for (int i = rank - 1; i >= 0; i--) { // Start from the last dimension.
        indices[i]++;
        if (indices[i] < dimCount[i]) {
            return true; // Move to the next element in this dimension.
        }
        indices[i] = 0; // Reset this dimension and carry to the next.
        if (i == 0) {
            return false; // We've exhausted all dimensions.
        }
    }
    return false;
}

template <typename T> static LogicalResult checkDmaNdls(T op) {

    auto readNdl = op.getReadNdl();
    auto writeNdl = op.getWriteNdl();

    auto maxWriteOffset = 0;
    auto totalWriteCount = 1;

    for (auto dim : writeNdl.getDims()) {

        if (dim.getStride() < 0) {
            return op.emitOpError("Write NDL stride must be non-negative");
        }

        maxWriteOffset += (dim.getCount() - 1) * dim.getStride();
        totalWriteCount *= dim.getCount();
    }

    auto totalReadCount = 1;

    for (auto dim : readNdl.getDims()) {

        if (dim.getStride() < 0) {
            return op.emitOpError("Read NDL stride must be non-negative");
        }

        totalReadCount *= dim.getCount();
    }

    if (totalReadCount != totalWriteCount) {
        return op.emitOpError("Read and write NDLs must have the same number of elements");
    }

    auto outputSize = getEncodedTotalSizeBytes(op.getWrite().getType());

    if (maxWriteOffset >= outputSize) {
        return op.emitOpError("Write NDL is writing outside the output buffer:\n")
               << " output memref size = " << outputSize
               << "\n  maximum address accessed = " << maxWriteOffset;
    }

    return success();
}

LogicalResult DmaInCfgOp::verify() { return checkDmaNdls(*this); }

LogicalResult DmaOutCfgOp::verify() { return checkDmaNdls(*this); }

LogicalResult CSSStartOp::verify() { return success(); }

LogicalResult SliceTaskOp::verify() {

    SmallVector<MemNdlAttr> deqwMemNdls;

    llvm::copy_if(getMemNdls(), std::back_inserter(deqwMemNdls), [](MemNdlAttr ndl) {
        return ndl.getType() == NdlType::DEQW;
    });

    if (deqwMemNdls.size() == 0) {
        return emitOpError("At least one DEQW NDL is required");
    }

    if (deqwMemNdls.size() != getQ().size()) {
        return emitOpError("Q values do not match DEQW NDLs number");
    }

    if (!clEnableDeqwVerify) {
        return success();
    }

    for (auto pair : llvm::zip(deqwMemNdls, getQ())) {
        auto ndl = std::get<0>(pair);
        auto q = std::get<1>(pair);

        auto qType = dyn_cast<MemRefType>(q.getType());

        if (!qType) {
            return emitOpError("Q value must be a memref type");
        }

        auto bufferSizeBytes = getEncodedTotalSizeBytes(qType);

        int ndlOffset = ndl.getOffset();
        int minLdimAddr = 0;
        int minHdimAddr = ndlOffset;
        int minSdimAddr = 0;
        int maxLdimAddr = 0;
        int maxHdimAddr = ndlOffset;
        int maxSdimAddr = 0;
        bool hasSdims = false;

        SmallVector<int64_t> dimCount;
        SmallVector<int64_t> dimStrideBytes;
        for (auto dim : ndl.getDims()) {
            auto count = dim.getCount();
            auto stride = dim.getStrideAsI64();

            // we cannot compute the stride because it is an affine expression
            // that is not constant, regardless of the value we will for sure
            // access the position with count 0 (even if the stride turns out
            // to be negative) so we keep going
            if (!stride) {
                continue;
            }

            if (count == 0 || *stride == 0) {
                // Skip this dimension (it doesn't contribute to the offset).
                continue;
            }

            dimCount.push_back(count);
            dimStrideBytes.push_back(*stride);

            int maxDimAddr = (*stride) * (count - 1);

            if (dim.getType() == DimType::L) {
                if (maxDimAddr > 0)
                    maxLdimAddr += maxDimAddr;
                else
                    minLdimAddr += maxDimAddr;
            }
            else if (dim.getType() == DimType::H) {
                if (maxDimAddr > 0)
                    maxHdimAddr += maxDimAddr;
                else
                    minHdimAddr += maxDimAddr;
            }
            else if (dim.getType() == DimType::S) {
                hasSdims = true;
                if (maxDimAddr > 0)
                    maxSdimAddr += maxDimAddr;
                else
                    minSdimAddr += maxDimAddr;
            }
        }

        int maxAddr = maxHdimAddr;
        int minAddr = minHdimAddr;

        if (hasSdims) {
            maxAddr += maxSdimAddr;
            minAddr += minSdimAddr;
        }
        else {
            maxAddr += maxLdimAddr;
            minAddr += minLdimAddr;
        }

        if (maxAddr >= bufferSizeBytes) {
            return emitOpError("DEQW NDL is writing outside the corresponding Q buffer:\n")
                   << "  deqw = " << ndl << "\n  q = " << qType
                   << "\n  Q buffer size = " << bufferSizeBytes << " bytes"
                   << "\n  maximum address accessed = " << maxAddr;
        }
        else if (minAddr < 0) {
            return emitOpError("DEQW NDL is reading outside the corresponding Q buffer:\n")
                   << "  deqw = " << ndl << "\n  q = " << qType
                   << "\n  Q buffer size = " << bufferSizeBytes << " bytes"
                   << "\n  minimum address accessed = " << minAddr;
        }

        /*
        // Get shape and padding info for this MemRef if present
        auto enc = dyn_cast_or_null<torq_hl::TensorEncodingAttr>(qType.getMemorySpace());
        SmallVector<int64_t, 8> nullAlign(qType.getRank() + 1, 0);
        llvm::ArrayRef<int64_t> align = enc && !enc.getAlign().empty() ? enc.getAlign() : nullAlign;

        // Get the strides and offset of the memref.
        int64_t baseOffset;
        SmallVector<int64_t, 8> memrefStridesElements;
        assert(!failed(getStridesAndOffset(qType, memrefStridesElements, baseOffset)));

        // Get the shape of the memref.
        ArrayRef<int64_t> shape = qType.getShape();
        SmallVector<int64_t> alignedStridesElements = getAlignedStrides(shape, align);

        if (memrefStridesElements == alignedStridesElements) {
            // the output is not strided, skip the stride check
            return success();
        }

        // Iterate over all possible NDL offsets given dimCount and dimStride and check if
        // this is within the bounds of the memref data including alignment
        // TODO: check if this works in all cases
        SmallVector<int> indices(dimCount.size(), 0);
        std::vector<int> badOffsetsBytes;
        do {
            auto offsetBytes =
                std::inner_product(indices.begin(), indices.end(), dimStrideBytes.begin(), 0);
            auto offsetElements = offsetBytes * 8 / qType.getElementTypeBitWidth();
            if (!isOffsetInMemRefData(
                    offsetElements, shape, alignedStridesElements, memrefStridesElements
                )) {
                badOffsetsBytes.push_back(offsetBytes);
            }

        } while (incrementIndex(indices, dimCount));

        if (!badOffsetsBytes.empty()) {
            // Show the bad offsets grouped into ranges
            auto ranges = groupIntoRanges(badOffsetsBytes);
            std::string badOffsetsStr;
            llvm::raw_string_ostream os(badOffsetsStr);
            for (const auto &range : ranges) {
                os << "  [" << range.first << ", " << range.second << "] ";
            }

            return emitOpError("DEQW NDL is writing outside the corresponding Q buffer:\n")
                   << "  deqw = " << ndl << "\n  q = " << qType
                   << "\n  Q buffer size = " << bufferSizeBytes
                   << " bytes\n  Bad offset byte ranges: \n"
                   << badOffsetsStr << "\n";
        }
                   */
    }
    return success();
}

} // namespace mlir::syna::torq_hw
