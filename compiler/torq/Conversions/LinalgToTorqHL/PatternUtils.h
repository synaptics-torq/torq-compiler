// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Dominance.h"

namespace mlir::syna::torq {

extern const std::string TORQ_FUSE_GROUP_ID;
extern const std::string TORQ_FUSE_GROUP;
// Set on the scf.for loops ReductionTilingPass creates. Each loop writes each
// slot of its partial-result buffer once.
inline constexpr llvm::StringLiteral TORQ_REDUCTION_TILING_LOOP = "torq-reduction-tiling-loop";
// Marks a generic rewritten from a batch-invariant-weight batch_matmul
// (replaceBatchMatmulWithBroadcastGeneric): still a matmul to the tiling heuristics.
extern const std::string TORQ_BROADCAST_MATMUL;
// Marks the matmul RaiseMatMulInteger emits; see Conv1DMatmulPattern.cpp.
// Always set together with TORQ_MATMUL_LHS_UNSIGNED. The LinalgToTorqHL pass
// boundaries hard-fail compilation if either attribute is missing or is not a
// UnitAttr; a software fallback cannot recover the unsigned semantics after
// the original zero-extended MatMulInteger chain has been raised.
// The flag chain is linalg UnitAttr -> torq_hl.matmul input1_unsigned
// BoolAttr -> toUnsigned() dtype in TorqHLToTorqHW; a typo or a missed
// set/consume anywhere in it is invisible to the compiler, so keep the two
// attributes at their declaration and the set/consume sites in sync.
extern const std::string TORQ_RAISED_MATMUL_INTEGER;
// Marks a matmul whose LHS holds unsigned 8-bit data (MLIR i8 is signless).
// A semantic fact, not a hint: if it is dropped between the raise and the
// materialization, every activation byte >= 128 would multiply as signed.
extern const std::string TORQ_MATMUL_LHS_UNSIGNED;

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

std::optional<int64_t> getConstIntValue(Value val);

// If maybeFuseGroupAttr is not std::nullopt and op implements TilingInterface,
// add maybeFuseGroupAttr to the TORQ_FUSE_GROUP array attribute of op. If op
// does not implement TilingInterface, it must be one of:
// - arith::ConstantOp
// - tensor::EmptyOp
// - IREE::TensorExt::DispatchTensorLoadOp
// - IREE::HAL::InterfaceBindingSubspanOp
// Return true iff a mark was added.
bool markOpFuseGroup(
    Operation *op, PatternRewriter &rewriter, const std::optional<IntegerAttr> &maybeFuseGroupAttr
);

// True if `value` is a constant or a pure reshape/broadcast/transpose/
// elementwise/reduction pipeline over constants (typical weight/bias
// preprocessing). Shared producers of this kind stay out of NPU fuse groups;
// unique ones join the consumer's group.
bool isConstantLikeProducer(Value value);

// Starting from output, walk upwards until any of the inputs are encountered.
// TilingInterface ops are marked; volume-preserving shape ops are walked through.
// Size-changing extract_slice, residual side-branches, and constant-like
// weight/bias pipelines are fusion boundaries. Returns true if any new mark
// was applied (so greedy rewrite can stop when the group is already annotated).
bool markFuseGroupBackward(
    const Value &output, const llvm::SmallVector<Value> &inputs, PatternRewriter &rewriter,
    const IntegerAttr &fuseGroupAttr
);

// Starting from outputOp, walk upwards and remove the fuseGroup marking from all operations.
void removeFuseGroupMarkingBackwards(Operation *outputOp, int64_t fuseGroup);

// Return true iff op is the principal Operation of the the fuse group fuseGroupAttr (i.e. the
// operation from which the pattern matching started).
bool isFuseGroupPrincipalOp(Operation *op, IntegerAttr fuseGroupAttr);

// Return the ID of the pattern fuse group for which `op` is the principal
// operation, if it is such, otherwise return `nullptr`.
IntegerAttr isFuseGroupPrincipalOp(Operation *op);

// Return the principal Operation of the fuse group (i.e. the operation from
// which the pattern matching started) outputOp belongs to, or nullptr if something
// goes wrong.
// outputOp - an Operation in the output of the principal Operation (it must have
// exactly one source that is in the same fuse group);
Operation *getFuseGroupPrincipalOpBackward(Operation *outputOp);

// Return true iff op is already marked as part of a fuse group.
bool isMarkedFuseGroup(Operation *op);

// A QDQ quantized matmul chain (i8 x i8 -> i32 matmul, dequant/requant
// epilogue). Owned by QMatmulToFCConvert (QMatmulPattern.cpp); other
// linalg::MatmulOp patterns must check this and decline to avoid racing it.
bool isQuantizedMatmulChain(linalg::MatmulOp op);

// Block-quantized dequant-matmul body matchers
bool matchMatmulRhsBlockQuantizedBody(linalg::GenericOp genericOp);

// True when such a dequant body subtracts a zero-extended (unsigned) zero point.
bool dequantHasUnsignedZeroPoint(linalg::GenericOp genericOp);

// `TorqStructuredOpMatcher::addPredicate` helper: reject ops already marked
// as part of a fuse group when `active` is true (typically `_markFuseGroups`).
// Used to gate discovery-mode patterns from re-marking already-marked ops.
inline auto notMarkedFuseGroupIf(bool active) {
    return [active](auto op) { return !(active && isMarkedFuseGroup(op.getOperation())); };
}

// Return true iff op1 and op2 share at least one fuse group.
bool checkShareFuseGroup(Operation *op1, Operation *op2);

// If op is the bottom most operation of a fuse group, return the id of that group,
// otherwise return std::nullopt.
std::optional<int64_t> isFuseGroupOutput(Operation *op);

// Assuming `op` is a member of the pattern fuse group `fuseGroupAttr`, return
// the bottom most member of the group.
Operation *getFuseGroupOutputOp(Operation *op, IntegerAttr fuseGroupAttr);

// Walk past tensor.pad and producer linalg.transpose so fuse-group marking
// includes the NCHW layout transpose. Does not walk expand/collapse
// (space-to-depth). Weights should not be passed through this helper.
Value peelPadAndReshapeSource(Value value);

// True if `lrtbPad` {left,right,top,bottom} is representable by conv HW.
// Stride 1 requires left/right equal to the kernel border. Stride > 2 and 2D
// k>1 stride 2 are always legal (the stride-2 lowering cannot run the VALID
// conv an explicit pad would leave). 1D-as-2D stride-2 (exactly one of kH,kW
// is 1) also accepts SAME [0,1] on the long axis.
bool isHwLegalConvPad(
    llvm::ArrayRef<int64_t> lrtbPad, int64_t kH, int64_t kW, int64_t strideH = 1,
    int64_t strideW = 1
);

// After foldBackwardPadding: keep `input` if the pad is HW-legal. Otherwise
// revert to `origInput` and zero pad (explicit tensor.pad, VALID HW pad).
void selectFoldedOrExplicitConvPad(
    Value &input, Value origInput, PaddingInfo &padInfo, PatternRewriter &rewriter,
    llvm::ArrayRef<Operation *> foldOps, int64_t kH, int64_t kW, int64_t strideH, int64_t strideW
);

// Erase ops a pad fold materialized that are now dead (the fold was reverted
// or its slice translation fell through). Erase newest first so chained ops
// become use-empty in order.
void eraseDeadFoldOps(PatternRewriter &rewriter, llvm::ArrayRef<Operation *> foldOps);

// Deduce Left,Right,Top,Bottom padding info by analyzing value backward
// - value comes from an tensor.slice_insert into a filled tensor, or from tensor.pad
// - return padding info if found and update value to the source tensor
// - return null padding if not found
// Update value to the input of the folded operations. On a committed fold,
// `createdOps` (optional) collects the ops the fold materialized (peel
// reshapes, adjusted extract slices) so a caller that reverts the fold can
// erase them.
PaddingInfo foldBackwardPadding(
    Value &value, PatternRewriter &rewriter, bool nchw = false, Value outputValue = nullptr,
    SmallVectorImpl<Operation *> *createdOps = nullptr
);

// Peel tensor.expand_shape / tensor.collapse_shape when this is the sole user.
// Does not change ScaleInfo. Returns true if peeled.
bool foldBackwardShapeCast(Value &value);

// Peel a yield-only linalg.generic (a plain broadcast/copy with no payload op)
// when this is the sole user. Does not change ScaleInfo. Returns true if peeled.
bool foldBackwardYieldOnlyGeneric(Value &value);

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

// Build a fusion plan rooted at `value` by selecting a terminal fusible result
// and collecting intermediate operations needed to reconstruct that chain.
// On success, `fusionPlan.anchor` and `fusionPlan.neededOps` are populated.
// Also updates the defining op of `value` to the terminal fused result, so that
// subsequent patterns can match against it.
FailureOr<FusionPlan>
buildFusionPlanAndRebindOutput(Value &value, std::optional<int64_t> knownChannelDim = std::nullopt);

// Build per-channel bias from the fusion plan rooted at `fusionPlan.anchor`.
// Optionally surfaces folded weight zero-point through `optionalWeightZp`.
// `biasChDim` can override the channel dimension used for reduction.
FailureOr<Value>
computeBias(FusionPlan &fusionPlan, int biasChDim, std::optional<Value> &optionalWeightZp);
// Convenience wrapper for matmul-like cases, selecting bias channel mapping
// based on `channelDim`.
FailureOr<Value>
computeBiasForMatmul(FusionPlan &fusionPlan, int biasChDim, std::optional<Value> &optionalWeightZp);

// Extract scale/clamp/zero-point info from the trailing rescale pattern in
// `fusionPlan` and interleave scale into `biasScale` for NPU consumption.
FailureOr<Value>
computeRescaleInfo(FusionPlan &fusionPlan, Value biasScale, ScaleClampInfo &scInfo);

// Convenience wrapper: compute per-channel bias from the fusion plan and then
// interleave scale/clamp info from the trailing rescale stage into biasScale.
// Equivalent to calling computeBias followed by computeRescaleInfo.
FailureOr<Value> computeBiasAndRescaleInfo(
    FusionPlan &fusionPlan, int channelDim, std::optional<Value> &optionalWeightZp,
    ScaleClampInfo &scInfo
);

// Apply weight zero-point correction to quantized weights:
// adjusted_w = sext(weight) - trunc(weightZp), materialized as i16 tensor.
FailureOr<Value> buildWeightWithZp(Value weights, Value weightZp, PatternRewriter &rewriter);

// Normalize filter layout and optionally fold a weight zero-point into weights.
Value preConversionWeights(
    Value weights, const Permutation &_weightsPerm, std::optional<Value> weightZpV,
    ScaleClampInfo &scInfo, PatternRewriter &rewriter, bool isDepthwise
);

// Materialize dilation into the weight tensor when dilations > 1.
FailureOr<Value> getDilatedWts(
    Value weights, std::vector<int64_t> &finalDilationVec, bool isDW1DStride1,
    PatternRewriter &rewriter
);

// Loose upper bound on the magnitude an element of `input` can reach once its zero point is
// subtracted, covering both the signed and unsigned reading of the element type.
double operandRange(Value input, int32_t zeroPoint);

// Return the largest scale shift, starting from minShift, for which both multipliers still fit
// a fixed-point int16 weight and their weighted operands still fit an int32 accumulator.
// A unit that evaluates (w0 * d0 + w1 * d1) >> shift keeps only as many fractional bits of the
// multipliers as the shift allows, so too small a shift quantizes the smaller multiplier
// coarsely enough to show up on the output.
int maximizeScaleShift(
    int minShift, double multiplier0, double multiplier1, double range0, double range1
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

// Match a float [N] bias, optionally rank-aligned or broadcast over `outShape`.
Value matchPerChannelFloatBias(Value value, ArrayRef<int64_t> outShape);

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

struct LinalgTableMatchInfo {
    Value input;                  // i16 input being looked up
    DenseIntElementsAttr lutData; // tensor<513xi16> LUT
};

// Match a linalg.generic whose body is 2-point linear interpolation over a
// tensor<513xi16> constant: result = (lut[i] << 7) + (lut[i+1] - lut[i]) * frac.
LogicalResult matchI16InterpolatedTable(linalg::GenericOp op, LinalgTableMatchInfo &info);

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

// `srcOp` is any linalg op (typically the generic being lowered); only its
// location is used. Creates a torq_hl.bitcast, which bufferizes to a free
// torq_hl.reinterpret_cast view (no data movement).
Value makeBitcast(
    linalg::LinalgOp srcOp, PatternRewriter &rewriter, RankedTensorType resultType, Value input
);

// Create a torq_hl.table looking up the i16 `input` in the HW's 512-entry
// piecewise-linear ACT table. `values` are the packed (slope<<16)|base words as
// written to the table memory. The ACT rescales the input before the lookup,
// tc = s16(clamp((x - inputZp) * scaleFactor >> shiftFactor) + outputZp), with
// the bias/scale carried in the table's scale_bias operand and
// shift/output_zp as attributes, so a pre-table rescale is fused into the
// table. The default (1, 0, 0, 0) rescale is the identity and interpolates
// linearly across the full i16 range (512 entries, 128 inputs each). A
// (256, 0, 0, 0) rescale instead makes tc a multiple of 128, so the 7-bit
// fraction is always 0 and the lookup is an exact read of the even entries
// only — the index then depends solely on the low byte of the input (tx =
// 2*((x & 0xFF) + 128) mod 512); that table must leave `saturateI16` unset so
// the (x * 256) rescale wraps at the s16 index instead of saturating. When
// `saturateI16` is set, out-of-domain inputs saturate to the table endpoints,
// matching the torq_hl.fma rescale this table fuses.
Value makeI16LUTFromVals(
    linalg::LinalgOp srcOp, PatternRewriter &rewriter, Value input, SmallVector<int32_t> values,
    int32_t scaleFactor = 1, int shiftFactor = 0, int32_t inputZp = 0, int32_t outputZp = 0,
    bool saturateI16 = true
);

// The bf16 reciprocal (see scripts/torch/bfloat16_softmax.py for the
// derivation): for the i16 bit pattern x of a bf16 value, the reciprocal's
// bit pattern is
//   recip = (kBfloatReciprocalExpoConst - x) + lut[s8(x) + 128]
// in i16 wrap-around arithmetic, where lut = getBfloatReciprocalLutData().
constexpr int32_t kBfloatReciprocalExpoConst = 0x7E80;

// The 256-entry adjusted bf16 reciprocal mantissa LUT (i8): the 128-entry
// magic base duplicated twice, with lut[i] += i % 128 (mod 256) applied.
SmallVector<int8_t> getBfloatReciprocalLutData();

// 'a', 'b', 'c' correspond to those in scripts/bf16luts.py comments/prints
// The input (c) rescale is fused into the table lookup (no separate FMA),
// unless cInputUnsigned is set: the table's P read path (ALU bypass) always
// reads the i16 input signed, so an unsigned rescale must stay a FMA.
// cInputUnsigned makes that FMA read `input` unsigned in the ALU (the
// magnitude, sign bit clear), so that bf16 bit-cast domains spanning the
// sign change (e.g. softmax's x - max, which is <= 0 with +0.0 when x ==
// max) clip +0.0 toward -0.0 instead of toward +inf. When set, the c input
// zero point must be expressed in that magnitude domain.
Value makeScaledLut(
    linalg::GenericOp srcOp, PatternRewriter &rewriter, Value input, int32_t aScaleFactor,
    int aShiftFactor, int32_t aInputZp, int32_t aOutputZp, SmallVector<int32_t> bValues,
    int32_t cScaleFactor, int cShiftFactor, int32_t cInputZp, int32_t cOutputZp,
    bool cInputUnsigned = false
);

Value makeSelect(
    linalg::GenericOp srcOp, PatternRewriter &rewriter, Value pred, Value ifTrue, Value ifFalse
);

Value makeElementWiseBinary(
    linalg::GenericOp srcOp, PatternRewriter &rewriter, Value input0, Value input1,
    torq_hl::ElementwiseOpEnum opType
);

// The splat value of `val` if it is a constant splat, a constant fill, or a slice
// or cast of one; null otherwise. `val` must have a RankedTensorType.
Attribute getConstantSplatAttr(Value val);

// Return true if `value` is all zeros. The function can only recognize simple cases. It does not do
// any computation. In particular, the function does handles extract_slice from an all zeros tensor,
// to support tiled operations.
// value - must have a RankedTensorType
bool isAllZerosTensor(mlir::Value value);

// If initVal is not all zeros, torq_hl.add it to val.
// This is useful when lowering a linalg operation that accumulates over init, to a torq_hl
// operation that does not.
FailureOr<Value> addInitToResult(Value initVal, Value val, PatternRewriter &rewriter);

} // namespace mlir::syna::torq
