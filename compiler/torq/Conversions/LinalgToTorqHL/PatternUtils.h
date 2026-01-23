// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir::syna::torq {

extern const std::string TORQ_FUSE_GROUP_ID;
extern const std::string TORQ_FUSE_GROUP;

// Input Scale Information
struct ScaleInfo {
    // Zero Point
    int32_t zp{};
    // Scale value
    double scale{1};
};

// Output Scale and Clamp Information
struct ScaleClampInfo {
    // Bias applied to the output before scaling
    int32_t bias{};
    // Output Zero Point
    int32_t zp{};
    // Min Output Value, default minimum int
    int32_t min{std::numeric_limits<int32_t>::min()};
    // Max Output Value
    int32_t max{std::numeric_limits<int32_t>::max()};
    // Shift amount of the scale values
    int32_t scaleShift{};
    // Int scale values computed with the scaleShift applied, can be used directly in the NPU
    std::vector<int32_t> scaleNpu;
    // Scale values expressed in double (not modified with scaleShift)
    // Useful when we need to combine multiple scale values together before converting to int
    std::vector<double> scaleDouble;

    // Invalid if no scale value
    operator bool() const { return !scaleNpu.empty(); }

    // Return true if has clamp
    bool hasClamp() const {
        return min != std::numeric_limits<int32_t>::min() ||
               max != std::numeric_limits<int32_t>::max();
    }
};

struct PaddingInfo {
    llvm::SmallVector<int64_t> lrtbPad = llvm::SmallVector<int64_t>(4, 0);
    int64_t padValue{};
};

struct APIntOrFloat {
    std::optional<APInt> apInt;
    std::optional<APFloat> apFloat;
};

struct APIntOrFloatArray {
    SmallVector<APInt> apInts;
    SmallVector<APFloat> apFloats;
};

struct VectorIntOrFloat {
    VectorIntOrFloat(int size, bool isInt) {
        if (isInt) {
            ints.resize(size);
        }
        else {
            floats.resize(size);
        }
    }
    std::vector<int32_t> ints;
    std::vector<float> floats;
};

bool isI8Type(Value val);
bool isI32Type(Value val);
bool isI8Type(Value val, PatternRewriter &rewriter);  // Deprecated
bool isI32Type(Value val, PatternRewriter &rewriter); // Deprecated

std::optional<int64_t> getConstIntValue(Value val);

// return the single operation using the value
// return nullptr if no user or more than one user
Operation *getSingleUser(Value value);

template <class OpT> OpT getSingleUser(Value value) {
    return dyn_cast_or_null<OpT>(getSingleUser(value));
}

// If maybeFuseGroupAttr is not std::nullopt and op implements TilingInterface,
// add maybeFuseGroupAttr to the TORQ_FUSE_GROUP array attribute of op. If op
// does not implement TilingInterface, it must be one of: arith::ConstantOp, or
// tensor::EmptyOp.
// Return true iff maybeFuseGroupAttr is not std::nullopt.
bool markOpFuseGroup(
    Operation *op, PatternRewriter &rewriter, const std::optional<IntegerAttr> &maybeFuseGroupAttr
);

// Starting from output, walk upwards until until any of the inputs are
// encountered. Mark all the operations on the way with the fuseGroupattr.
void markFuseGroupBackward(
    const Value &output, const llvm::SmallVector<Value> &inputs, PatternRewriter &rewriter,
    const IntegerAttr &fuseGroupAttr
);

// Return all the values that feed the fuse group from outside the group (with no duplicates).
// root - the bottom most operation in the fuse group.
SmallVector<Value> getFuseGroupOperands(Operation *root, const IntegerAttr &fuseGroupAttr);

// Return true iff op is the principal Operation of the the fuse group fuseGroupAttr (i.e. the
// operation from which the pattern matching started).
bool isFuseGroupPrincipalOp(Operation *op, IntegerAttr fuseGroupAttr);

// Return the principal Operation of the fuse group (i.e. the operation from
// which the pattern matching started) op belongs to, or nullptr if something
// goes wrong.
// op - an Operation in the output of the principal Operation (it must have
// exactly one source that is in the same fuse group);
Operation *getFuseGroupPrincipalOpBackward(Operation *op);

// Walks forward from result, over operations that belong to fuseGroupAttr, and
// return all the OpOperands that are owned by the principal operation of
// fuseGroupAttr, and reachable by the walk.
SmallVector<OpOperand *>
getFuseGroupPrincipalOpOperandsForward(IntegerAttr fuseGroupAttr, OpResult result);

// Return true iff op is already marked as part of a fuse group.
bool isMarkedFuseGroup(Operation *op);

// If op is the bottom most operation of a fuse group, return the id of that group,
// otherwise return std::nullopt.
std::optional<int64_t> isFuseGroupOutput(Operation *op);

// Deduce Left,Right,Top,Bottom padding info by analyzing value backward
// - value comes from an tensor.slice_insert into a filled tensor, or from tensor.pad
// - return padding info if found and update value to the source tensor
// - return null padding if not found
// Update value to the input of the folded operations
PaddingInfo foldBackwardPadding(Value &value, PatternRewriter &rewriter, bool nchw = false);

// Deduce scaling and zp by looking for a rescale backward
// If a rescale is found, ScaleInfo information is adjusted accordingly
// Update value to the input of the folded operations
// return true if success, false if nothing folded
bool foldBackwardRescale(Value &value, ScaleInfo &scaleInfo);

// Deduce per-channel added biases by analyzing value forward
// Add per-channel biases to the bias vector
// Advance value to the result of the folded operations
// If inValue is not null it is used as a reference to extract the weight zero point
// return true if success
bool foldForwardPerChannelAdd(
    Value &value, int channelDim, VectorIntOrFloat &bias, int32_t *input_zp = nullptr,
    Value inValue = nullptr, int32_t *w_zp = nullptr
);

// Deduce ScaleClampInfo forward
// scaleValuesCount is the expected number of scale values
// shift8b is the shift amount of the scale values for 8bits computations
// shift16b is the shift amount of the scale values for 16bits computations
// isElementWiseOp indicates if this is called for element-wise operations (add/sub) vs convolution
// return valid ScaleClampInfo if success
ScaleClampInfo foldForwardScaleClamp(
    Value &value, int scaleValuesCount, int shift8b, int shift16b, bool isElementWiseOp = false
);

// Deduce weights zero-point forward
// return zero-point if success, 0 if no zero-point found
int foldForwardWeightZp(Value &value);

// Tosa multiplier and shift info
struct MultiplierShiftInfo {
    std::vector<int32_t> multiplier;
    std::vector<int8_t> shift;
    // Invalid if no multiplier value
    operator bool() const { return !multiplier.empty(); }
};

// Get the multiplier and shift values from tosa apply_scale op
MultiplierShiftInfo getMultiplierAndShift(
    linalg::GenericOp genericOp, tosa::ApplyScaleOp applyScaleOp, int scaleValuesCount
);

ScaleClampInfo getDefaultScaleClampInfo(Type outElemType, Operation *srcOp);

Operation *getElementwiseTernaryOp(linalg::GenericOp op, bool allowConstants = false);

Operation *getElementwiseBinaryOp(linalg::GenericOp op, bool allowConstants = false);

Operation *getElementwiseUnaryOp(linalg::GenericOp op);

// Fold the transposeOp to DepthToSpace if it is following either of the permutation
// DCR perm 0, 1, 3, 2, 4, 5
// CRD perm 0, 1, 4, 2, 5, 3
// Refer https://onnx.ai/onnx/operators/onnx__DepthToSpace.html
// Folds the ExpandShapeOp above the TransposeOp and CollapseShapeOp below.
LogicalResult foldForwardDepthToSpace(
    linalg::TransposeOp transposeOp, PatternRewriter &rewriter,
    const std::optional<IntegerAttr> &maybeFuseGroupAttr
);

bool isRoundingRightShiftOp(linalg::GenericOp op, arith::ShRSIOp &shrsiOp1);

// Return true iff op is a linalg.generic that is the result of conversion from
// tensor.collapse_shape/expand_shape.
bool isCollapseOrExpandShapeGeneric(Operation *op);

StringRef getCastOpName(Value input, Value output);

bool getIntegerConstantValue(arith::ConstantOp constOp, int32_t *value);

Value create1DimTensorFromRescaleScalar(
    linalg::GenericOp srcOp, tosa::ApplyScaleOp applyScaleOp, ScaleInfo &scaleInfo,
    const Type &elementType, PatternRewriter &rewriter
);

bool foldScalarRescale(
    Value &input, ScaleInfo &scaleInfo, const Type &elementType, PatternRewriter &rewriter
);

Value convertWeights(mlir::linalg::MatmulOp srcOp, mlir::Value weights, PatternRewriter &rewriter);

} // namespace mlir::syna::torq
