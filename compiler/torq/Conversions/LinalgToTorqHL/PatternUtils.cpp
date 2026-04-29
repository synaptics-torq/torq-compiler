// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TargetSelect.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "llvm/Support/Debug.h"

#include "iree/hal/local/executable_library.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include <deque>
#include <numeric>
#include <optional>

#define DEBUG_TYPE "torq-pattern-utils"

using namespace mlir::syna::torq_hl;
using namespace mlir::linalg;

namespace mlir::syna::torq {

const std::string TORQ_FUSE_GROUP_ID = "torq-fuse-group-id";
const std::string TORQ_FUSE_GROUP = "torq-fuse-group";

bool isI8Type(Value val, PatternRewriter &rewriter) {
    auto shapedType = dyn_cast<ShapedType>(val.getType());

    if (!shapedType) {
        return val.getType() == rewriter.getI8Type();
    }
    else {
        return shapedType.getElementType() == rewriter.getI8Type();
    }
}

bool isI32Type(Value val, PatternRewriter &rewriter) {
    auto shapedType = dyn_cast<ShapedType>(val.getType());

    if (!shapedType) {
        return val.getType() == rewriter.getI32Type();
        ;
    }
    else {
        return shapedType.getElementType() == rewriter.getI32Type();
    }
}

bool isI8Type(Value val) {
    auto shapedType = dyn_cast<ShapedType>(val.getType());
    Type elementType = shapedType ? shapedType.getElementType() : val.getType();
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
        return intType.getWidth() == 8;
    }
    return false;
}

bool isI32Type(Value val) {
    auto shapedType = dyn_cast<ShapedType>(val.getType());
    Type elementType = shapedType ? shapedType.getElementType() : val.getType();
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
        return intType.getWidth() == 32;
    }
    return false;
}

std::optional<int64_t> getConstIntValue(Value val) {
    auto constOp = val.getDefiningOp<arith::ConstantOp>();

    if (!constOp)
        return std::nullopt;

    return getConstantIntValue(constOp.getValue());
}

std::optional<float> getFloatValue(Value val) {
    // Handle truncf/extf operations that convert from constant float
    if (auto truncOp = val.getDefiningOp<arith::TruncFOp>()) {
        return getFloatValue(truncOp.getIn());
    }
    if (auto extOp = val.getDefiningOp<arith::ExtFOp>()) {
        return getFloatValue(extOp.getIn());
    }

    auto constOp = val.getDefiningOp<arith::ConstantOp>();

    if (!constOp)
        return std::nullopt;

    FloatAttr fillValueAttr = dyn_cast<FloatAttr>(constOp.getValue());
    if (!fillValueAttr)
        return std::nullopt;

    APFloat fillValueAPFloat = fillValueAttr.getValue();

    return fillValueAPFloat.convertToFloat();
}

Operation *getSingleUser(Value value) {
    if (!value.hasOneUse())
        return {};
    return *value.getUsers().begin();
}

bool markOpFuseGroup(
    Operation *op, PatternRewriter &rewriter, const std::optional<IntegerAttr> &maybeFuseGroupAttr
) {
    if (!maybeFuseGroupAttr) {
        return false;
    }

    if (isa<arith::ConstantOp, tensor::EmptyOp,
            mlir::iree_compiler::IREE::TensorExt::DispatchTensorLoadOp,
            mlir::iree_compiler::IREE::HAL::InterfaceBindingSubspanOp>(op)) {
        return true;
    }

    assert(op && "Trying to mark null op!");
    assert(
        (isa<TilingInterface, tensor::InsertSliceOp>(op) &&
         "Trying to mark an op that does not implement TilingInterface")
    );
    assert(*maybeFuseGroupAttr && "op does not have torq-fuse-group-id attribute");

    SmallVector<Attribute> newAttr;
    if (ArrayAttr oldAttr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP)) {
        if (llvm::is_contained(oldAttr, *maybeFuseGroupAttr))
            return true;

        llvm::append_range(newAttr, oldAttr);
    }
    newAttr.push_back(*maybeFuseGroupAttr);
    ArrayAttr fuseGroupAttr = rewriter.getArrayAttr(newAttr);

    rewriter.modifyOpInPlace(op, [&]() { op->setAttr(TORQ_FUSE_GROUP, fuseGroupAttr); });

    return true;
}

void markFuseGroupBackward(
    const Value &output, const llvm::SmallVector<Value> &inputs, PatternRewriter &rewriter,
    const IntegerAttr &fuseGroupAttr
) {
    std::deque<Value> stack = {output};
    while (!stack.empty()) {
        auto output = stack.front();
        stack.pop_front();

        if (llvm::find(inputs, output) != inputs.end()) {
            continue;
        }

        Operation *op = output.getDefiningOp();
        assert(op != nullptr && "Expected an op");

        // If we already visited this op, no need to do it again.
        if (auto attr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
            attr && llvm::is_contained(attr, fuseGroupAttr)) {
            continue;
        }
        markOpFuseGroup(op, rewriter, fuseGroupAttr);

        for (auto operand : op->getOperands()) {
            stack.push_back(operand);
        }
    }
}

void removeFuseGroupMarkingBackwards(Operation *outputOp, int64_t fuseGroup) {
    MLIRContext *context = outputOp->getContext();
    IntegerAttr fuseGroupAttr = IntegerAttr::get(IntegerType::get(context, 64), fuseGroup);

    std::deque<Operation *> stack = {outputOp};
    while (!stack.empty()) {
        Operation *op = stack.front();
        stack.pop_front();

        ArrayAttr oldAttr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
        if (!oldAttr)
            continue;

        if (!llvm::is_contained(oldAttr, fuseGroupAttr))
            continue;

        SmallVector<Attribute> newAttr;
        for (Attribute attr : oldAttr) {
            if (attr != fuseGroupAttr)
                newAttr.push_back(attr);
        }

        if (newAttr.empty()) {
            op->removeAttr(TORQ_FUSE_GROUP);
        }
        else {
            op->setAttr(TORQ_FUSE_GROUP, ArrayAttr::get(context, newAttr));
        }

        for (Value operand : op->getOperands()) {
            if (Operation *srcOp = operand.getDefiningOp())
                stack.push_back(srcOp);
        }
    }
}

SmallVector<Value> getFuseGroupOperands(Operation *root, const IntegerAttr &fuseGroupAttr) {
    SmallVector<Value> inputs;

    SmallVector<Value, 2> stack = root->getOperands();
    while (!stack.empty()) {
        auto output = stack.pop_back_val();

        Operation *op = output.getDefiningOp();
        assert(op != nullptr && "Expected an op");

        if (auto attr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
            !attr || !llvm::is_contained(attr, fuseGroupAttr)) {
            if (!llvm::is_contained(inputs, output)) {
                inputs.push_back(output);
            }
            continue;
        }

        stack.append(op->getOperands().begin(), op->getOperands().end());
    }

    return inputs;
}

bool isFuseGroupPrincipalOp(Operation *op, IntegerAttr fuseGroupAttr) {
    auto fuseGroupIdAttr = op->getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
    return fuseGroupIdAttr && fuseGroupIdAttr == fuseGroupAttr;
}

IntegerAttr isFuseGroupPrincipalOp(Operation *op) {
    ArrayAttr fuseGroupAttr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);

    if (!fuseGroupAttr)
        return nullptr;

    for (IntegerAttr intAttr : fuseGroupAttr.getAsRange<IntegerAttr>()) {
        if (isFuseGroupPrincipalOp(op, intAttr))
            return intAttr;
    }

    return nullptr;
}

Operation *getFuseGroupPrincipalOpBackward(Operation *outputOp) {
    ArrayAttr fuseGroupArrAttr = outputOp->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
    assert(fuseGroupArrAttr && "outputOp is not part of a fuse-group");

    assert(fuseGroupArrAttr.size() == 1 && "outputOp is part of multiple fuse-groups");
    IntegerAttr fuseGroupAttr = *fuseGroupArrAttr.getAsRange<IntegerAttr>().begin();

    std::deque<Operation *> stack = {outputOp};
    while (!stack.empty()) {
        Operation *op = stack.front();
        stack.pop_front();

        if (isFuseGroupPrincipalOp(op, fuseGroupAttr)) {
            return op;
        }

        for (auto operand : op->getOperands()) {
            auto srcOp = operand.getDefiningOp();
            if (auto attr = srcOp->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
                attr && llvm::is_contained(attr, fuseGroupAttr)) {
                stack.push_back(srcOp);
            }
        }
    }

    return nullptr;
}

SmallVector<OpOperand *>
getFuseGroupPrincipalOpOperandsForward(IntegerAttr fuseGroupAttr, Value result) {
    SmallVector<OpOperand *> principalOperands;

    std::deque<OpOperand *> queue;
    for (auto &next : result.getUses())
        queue.push_back(&next);

    while (!queue.empty()) {
        OpOperand *operand = queue.front();
        queue.pop_front();

        Operation *op = operand->getOwner();

        if (auto arrayAttr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
            arrayAttr && llvm::is_contained(arrayAttr, fuseGroupAttr)) {

            if (auto idAttr = op->getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
                idAttr && idAttr == fuseGroupAttr) {

                principalOperands.push_back(operand);
                continue;
            }

            for (auto &next : op->getUses())
                queue.push_back(&next);
        }
    }

    return principalOperands;
}

bool isMarkedFuseGroup(Operation *op) {
    // We don't check if the array attribute is not empty, because we always add
    // something to it when we created it.
    return op->hasAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
}

bool checkShareFuseGroup(Operation *op1, Operation *op2) {
    ArrayAttr fuseGroupAttr1 = op1->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
    ArrayAttr fuseGroupAttr2 = op2->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);

    if (!fuseGroupAttr1 || !fuseGroupAttr2)
        return false;

    for (auto attr1 : fuseGroupAttr1) {
        if (llvm::is_contained(fuseGroupAttr2, attr1)) {
            return true;
        }
    }

    return false;
}

std::optional<int64_t> isFuseGroupOutput(Operation *op) {
    auto fuseGroupAttr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
    if (!fuseGroupAttr) {
        return std::nullopt;
    }

    for (auto attr : fuseGroupAttr) {
        auto intAttr = cast<IntegerAttr>(attr);

        bool foundUser = false;
        for (auto user : op->getUsers()) {
            if (auto userFuseGroupAttr = user->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
                userFuseGroupAttr && llvm::is_contained(userFuseGroupAttr, intAttr)) {
                foundUser = true;
                break;
            }
        }

        if (foundUser) {
            continue;
        }

        return getConstantIntValue(intAttr);
    }

    return std::nullopt;
}

Operation *getFuseGroupOutputOp(Operation *op, IntegerAttr fuseGroupAttr) {
    Operation *currentOp = op;
    bool inGroup = true;
    while (inGroup) {
        inGroup = false;
        for (auto user : currentOp->getUsers()) {
            auto userFuseGroupAttr = user->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
            if (!userFuseGroupAttr)
                continue;

            if (llvm::is_contained(userFuseGroupAttr, fuseGroupAttr)) {
                inGroup = true;
                currentOp = user;
                break;
            }
        }
    }
    return currentOp;
}

// Extract and return the tosa multiplier and shift values from a tosa::ApplyScaleOp
MultiplierShiftInfo getMultiplierAndShift(
    linalg::GenericOp genericOp, tosa::ApplyScaleOp applyScaleOp, int scaleValuesCount
) {
    auto tosaMultiplier = applyScaleOp.getMultiplier();
    auto tosaShift = applyScaleOp.getShift();

    auto tosaMultiplierArg = dyn_cast<BlockArgument>(tosaMultiplier);
    auto tosaShiftArg = dyn_cast<BlockArgument>(tosaShift);

    Value tosaMultiplierValue =
        tosaMultiplierArg ? genericOp.getMatchingOpOperand(tosaMultiplierArg)->get() : nullptr;
    Value tosaShiftValue =
        tosaShiftArg ? genericOp.getMatchingOpOperand(tosaShiftArg)->get() : nullptr;

    auto tosaMultiplierType =
        tosaMultiplierValue ? cast<RankedTensorType>(tosaMultiplierValue.getType()) : nullptr;
    auto tosaShiftType =
        tosaShiftValue ? cast<RankedTensorType>(tosaShiftValue.getType()) : nullptr;

    if (tosaMultiplierType && tosaShiftType &&
        tosaMultiplierType.getShape() != tosaShiftType.getShape()) {
        genericOp.emitError() << "matching error multiplier and shift have different shapes!\n";
        return {};
    }

    std::vector<int32_t> multiplier;
    if (tosaMultiplierValue) {
        auto mulConstV = computeArithConst(tosaMultiplierValue, true, {});
        if (failed(mulConstV)) {
            genericOp.emitError() << "multiplier must be a constant";
            return {};
        }

        multiplier = attrValuesAsVec<int32_t>(*mulConstV);
        if (multiplier.size() != scaleValuesCount) {
            genericOp.emitError() << "multiplier unexpected size:" << multiplier.size() << "\n";
            return {};
        }
    }
    else {
        std::optional<int64_t> scalarMultiplier = getConstIntValue(tosaMultiplier);
        if (!scalarMultiplier.has_value()) {
            genericOp.emitError() << "matching error multiplier must be a scalar constant!\n";
            return {};
        }
        multiplier = std::vector<int32_t>(scaleValuesCount, scalarMultiplier.value());
    }

    std::vector<int8_t> shift;
    if (tosaShiftValue) {
        auto shiftConstV = computeArithConst(tosaShiftValue, true, {});
        if (failed(shiftConstV)) {
            genericOp.emitError() << "shift must be a constant";
            return {};
        }

        shift = attrValuesAsVec<int8_t>(*shiftConstV);
        if (shift.size() != scaleValuesCount) {
            genericOp.emitError() << "shift vect has unexpected size: " << shift.size() << "\n";
            return {};
        }
    }
    else {
        std::optional<int64_t> scalarShift = getConstIntValue(tosaShift);
        if (!scalarShift.has_value()) {
            genericOp.emitError() << "matching error shift must be a scalar constant!\n";
            return {};
        }
        shift = std::vector<int8_t>(scaleValuesCount, scalarShift.value());
    }
    return {std::move(multiplier), std::move(shift)};
}

// Input scaling is normally expressed as a linalg generic with 1 input and the following body:
// ^bb0(%in: i8, %out: i32):
//   %10 = arith.extsi %in : i8 to i32
//   %11 = arith.subi %10, %c-2_i32 : i32
//   %12 = tosa.apply_scale %11, %c107_i32, %c10_i8 {double_round = false} : (i32, i32, i8) -> i32
//   linalg.yield %12 : i32
// Sometimes followed by another linalg generic with the following body:
// ^bb0(%in: i32, %out: i32):
//   %10 = tosa.apply_scale %in, %c151_i32, %c32_i8 {double_round = true} : (i32, i32, i8) -> i32
//   linalg.yield %10 : i32
// It also support an alternative pattern used for input rescaling from i16:
// A sequence of linalg.generic ops where:
//   1. %in is first extended from i16 to i32,
//   2. shifted left by a constant,
//   3. and finally scaled via tosa.apply_scale.
//
// Example pattern:
// ^bb0(%in: i16, %out: i32):                     // linalg.generic (extsi)
//   %0 = arith.extsi %in : i16 to i32
//   linalg.yield %0 : i32
// ^bb0(%in: i32, %out: i32):                     // linalg.generic (shli)
//   %1 = arith.shli %in, %c15_i32 : i32
//   linalg.yield %1 : i32
// ^bb0(%in: i32, %out: i32):                     // linalg.generic (apply_scale)
//   %2 = tosa.apply_scale %in, %cM_i32, %cS_i8 {double_round = true} : (i32, i32, i8) -> i32
//   linalg.yield %2 : i32
bool foldBackwardRescale(Value &value, ScaleInfo &scaleInfo) {
    linalg::GenericOp rescaleOp = value.getDefiningOp<linalg::GenericOp>();
    if (!rescaleOp || rescaleOp.getInputs().size() != 1) {
        return false;
    }
    if (rescaleOp.getNumReductionLoops() > 0) {
        // This is not an element-wise operation
        return false;
    }
    auto yieldOp = dyn_cast<linalg::YieldOp>(rescaleOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getValues().size() != 1) {
        rescaleOp.emitError() << "matching error yieldOp is not a 1-valued yield!\n";
        return {};
    }

    auto applyScaleOp = yieldOp.getValues()[0].getDefiningOp<tosa::ApplyScaleOp>();
    if (!applyScaleOp) {
        LLVM_DEBUG({ llvm::dbgs() << "yield not an ApplyScale\n"; });
        return {};
    }

    // TODO: check double round?

    int zp = 0;
    Value in = applyScaleOp.getValue();
    if (auto ext = in.getDefiningOp<arith::ExtSIOp>()) {
        // direct extsi→apply_scale
        in = ext.getIn();
    }

    if (auto subOp = in.getDefiningOp<arith::SubIOp>()) {
        in = subOp.getLhs();
        if (auto ext = in.getDefiningOp<arith::ExtSIOp>()) {
            in = ext.getIn();
        }
        auto maybeZeroPoint = getConstIntValue(subOp.getRhs());
        if (!maybeZeroPoint) {
            LLVM_DEBUG({ llvm::dbgs() << "addOp.getRhs() is not a constant\n"; });
            return {};
        }
        zp = *maybeZeroPoint;
    }

    if (!isa<BlockArgument>(in)) {
        LLVM_DEBUG({ llvm::dbgs() << "applyScaleOp in is not a BlockArgument\n"; });
        return {};
    }

    Value input = rescaleOp.getInputs()[0];

    int64_t shiftAmount = 0;
    if (auto shiftGen = dyn_cast_or_null<linalg::GenericOp>(input.getDefiningOp())) {
        Operation *maybeBinary = getElementwiseBinaryOp(shiftGen, /*allowConstants*/ true);
        auto shliOp = dyn_cast_or_null<arith::ShLIOp>(maybeBinary);
        if (shliOp) {
            // The shift constant may be an arith.constant directly, or it may
            // be a BlockArgument of the enclosing linalg.generic (when the
            // constant tensor is passed as an input operand).  Resolve through
            // the block argument mapping so we can extract the value.
            Value shiftVal = shliOp.getRhs();
            if (auto blockArg = dyn_cast<BlockArgument>(shiftVal)) {
                shiftVal = shiftGen.getDpsInputOperand(blockArg.getArgNumber())->get();
            }
            // Try scalar constant first, then dense splat tensor constant
            // (e.g. arith.constant dense<15> : tensor<1x1x1x1xi32>).
            std::optional<int64_t> maybeC = getConstIntValue(shiftVal);
            if (!maybeC) {
                if (auto denseAttr = returnDenseElementAttr(shiftVal)) {
                    if (denseAttr.isSplat())
                        maybeC = denseAttr.getSplatValue<APInt>().getSExtValue();
                }
            }
            if (maybeC) {
                shiftAmount = *maybeC;
                input = shiftGen.getInputs()[0];
                // Peel off its input tensor (the output of the extsi gen):
                Value extTensor = shiftGen.getInputs()[0];
                if (auto extGen = dyn_cast_or_null<linalg::GenericOp>(extTensor.getDefiningOp())) {
                    Operation *maybeUnaryExtOp = getElementwiseUnaryOp(extGen);
                    auto extOp = dyn_cast_or_null<arith::ExtSIOp>(maybeUnaryExtOp);
                    if (extOp) {
                        input = extGen.getInputs()[0];
                    }
                }
            }
        }
    }

    auto ms = getMultiplierAndShift(rescaleOp, applyScaleOp, 1);
    if (!ms) {
        LLVM_DEBUG({ llvm::dbgs() << "applyScaleOp multiplier and shift not found\n"; });
        return {};
    }
    // Apply shiftLeft to modify the effective scale
    int64_t effectiveShift = ms.shift[0] - shiftAmount;
    double scale = computeScaleDouble(ms.multiplier[0], effectiveShift);

    // Update scaling information
    scaleInfo.zp += zp;
    scaleInfo.scale *= scale;

    // Fold the operation(s)
    value = input;
    return true;
}

// If user of output is a linalg.generic whose body contains
// an arith.truncf, if its output is BF16 truncation pattern and
// forward `output` to that genericOp's result.
static void foldForwardTruncFOp(Value &value) {
    if (value.hasOneUse()) {
        auto *userOp = *value.getUsers().begin();
        if (auto genericOp = dyn_cast<linalg::GenericOp>(userOp)) {
            Block *body = genericOp.getBody();

            // Strict check: body has a truncf
            if (isa<arith::TruncFOp>(body->front())) {
                if (dyn_cast<arith::TruncFOp>(body->front()).getOut().getType().isBF16()) {
                    value = genericOp.getResult(0);
                }
                else {
                    LLVM_DEBUG({ llvm::dbgs() << "truncFOp output type request bf16\n"; });
                }
            }
        }
    }
}

// ScaleClamp is normally expressed as a linalg generic with the following body for integer:
// ^bb0(%in: i32, %in_3: i32, %in_4: i8, %out: i8):
//  [%16 = arith.subi %in, %c196_i32 : i32] optional
//   %17 = tosa.apply_scale %16, %in_3, %in_4 {double_round = true} : (i32, i32, i8) -> i32
//   %18 = arith.addi %17, %c-128_i32 : i32
//   %19 = arith.maxsi %18, %c-128_i32 : i32
//   %20 = arith.minsi %19, %c127_i32 : i32
//   %21 = arith.trunci %20 : i32 to i8
//   linalg.yield %21 : i8
// The 2nd and 3rd args to apply_scale can be vectors or scalars
// The addi is used to add the zero point and is optional

static ScaleClampInfo foldForwardFloatTruncClamp(Value &value) {
    ScaleClampInfo sci;
    auto valueType = dyn_cast<ShapedType>(value.getType());
    if (!valueType) {
        LLVM_DEBUG({ llvm::errs() << "matching error value is not a ShapedType!\n"; });
        return {};
    }

    if (!llvm::isa<FloatType>(valueType.getElementType())) {
        return {};
    }

    // For floating point tensors the scale operation is not mandatory so we initialize
    // it with meaningful default values
    float max = std::numeric_limits<float>::max();
    sci.max = reinterpret_cast<int32_t &>(max);
    float min = std::numeric_limits<float>::lowest();
    sci.min = reinterpret_cast<int32_t &>(min);

    foldForwardTruncFOp(value);

    // check output clamp
    valueType = dyn_cast<ShapedType>(value.getType());
    if (!valueType) {
        LLVM_DEBUG({ llvm::errs() << "matching error value is not a ShapedType!\n"; });
        return {};
    }

    if (valueType.getElementType().isBF16()) {

        linalg::GenericOp maybeClampOp = getSingleUser<linalg::GenericOp>(value);
        if (!maybeClampOp) {
            LLVM_DEBUG({
                llvm::dbgs() << "must have a single GenericOp of cmp and select for clamp\n";
            });
            return sci;
        }

        if (maybeClampOp.getNumDpsInits() != 1 || maybeClampOp.getNumDpsInputs() != 1) {
            LLVM_DEBUG({
                llvm::dbgs() << "must have a single GenericOp of cmp and select for clamp\n";
            });
            return sci;
        }

        auto yieldOp = dyn_cast<linalg::YieldOp>(maybeClampOp.getBody()->getTerminator());
        if (!yieldOp) {
            LLVM_DEBUG({ llvm::dbgs() << "matching error yieldOp is not a yield!\n"; });
            return sci;
        }

        // relax clamp check: iterate genericOp to find two selectOp otherwise it is not clampOp
        Region &bodyRegion = maybeClampOp->getRegion(0);
        Block &entryBlock = bodyRegion.front();
        SmallVector<float, 2> vals;

        // TODO: if there are more than 2 selectOp for some algo, need to add more check
        int cnt = 2;

        for (Operation &op : entryBlock) {

            if (op.hasTrait<OpTrait::IsTerminator>())
                continue;

            if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
                if (cnt <= 0) {
                    llvm::errs() << "NOTE: there are more than 2 selectOp for clamp op\n";
                }
                cnt--;

                auto trueValue = selectOp.getTrueValue();
                auto val = getFloatValue(trueValue);
                if (!val) {
                    continue;
                }
                vals.push_back(*val);
            }
        }

        if (!vals.empty() && vals.size() == 2 && vals[0] != vals[1]) {
            sci.max = llvm::bit_cast<uint32_t>(vals[0] > vals[1] ? vals[0] : vals[1]);
            sci.min = llvm::bit_cast<uint32_t>(vals[0] < vals[1] ? vals[0] : vals[1]);

            value = maybeClampOp.getResult(0);
        }
        LLVM_DEBUG({ llvm::dbgs() << "sci.max: " << sci.max << " sci.min: " << sci.min << "\n"; });
    }

    return sci;
}

// Checks that the operation is an element-wise operation with the given arity
// An element-wise operation is defined as a linalg.generic op with:
//  - all parallel loops
// - arity inputs and one output
// - all indexing maps are identity
// - the init tensor is not used to compute the output
static bool isElementwiseOperation(linalg::GenericOp op, int arity) {

    // no reductions
    if (!op.isAllParallelLoops() || op.getNumLoops() < 1)
        return false;

    // aryity inputs and one output and all indexing maps are identity
    if (op.getNumDpsInputs() != arity || op.getNumDpsInits() != 1 ||
        !llvm::all_of(op.getIndexingMapsArray(), [](AffineMap map) { return map.isIdentity(); }))
        return false;

    // the init tensor is not used to compute the output
    if (op.payloadUsesValueFromOperand(op.getDpsInitOperand(0)))
        return false;

    return true;
}

//
// Find a fused per-channel add followed by clamp like this
//
//  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
//  affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2,
//  d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%in, %bias :
//  tensor<1x32x112x112xf32>, tensor<1x32x112x112xf32>) outs(%8 : tensor<1x32x112x112xbf16>) {
//   ^bb0(%in: f32, %in_4: f32, %out: bf16):
//     %10 = arith.addf %in, %in_4 : f32
//     %11 = arith.truncf %10 : f32 to bf16
//     %12 = arith.cmpf ult, %11, %cst_1 : bf16
//     %13 = arith.select %12, %cst_1, %11 : bf16
//     %14 = arith.cmpf ugt, %13, %cst_3 : bf16
//     %15 = arith.select %14, %cst_3, %13 : bf16
//     linalg.yield %15 : bf16
//   } -> tensor<1x32x112x112xbf16>
//
//
struct FuseBiasMatcher {

    Value bias;     // this corresponds to %bias
    Value clampMin; // this corresponds to %cst_1
    Value clampMax; // this corresponds to %cst_3

    static std::optional<FuseBiasMatcher> match(Operation *op, std::string &failureReason) {

        auto genericOp = dyn_cast<linalg::GenericOp>(op);

        if (!genericOp) {
            failureReason = "not a linalg.generic op";
            return std::nullopt;
        }

        if (!isElementwiseOperation(genericOp, 2)) {
            failureReason = "operation is not elementwise with 2 inputs";
            return std::nullopt;
        }

        auto input0Type = dyn_cast<ShapedType>(genericOp.getDpsInputOperand(0)->get().getType());
        auto input1Type = dyn_cast<ShapedType>(genericOp.getDpsInputOperand(1)->get().getType());
        auto outputType = dyn_cast<ShapedType>(genericOp.getDpsInitOperand(0)->get().getType());

        // ensure expected types
        if (!input0Type || !input1Type || !outputType || !input0Type.getElementType().isF32() ||
            !input1Type.getElementType().isF32() || !outputType.getElementType().isBF16()) {
            failureReason = "expected F32 inputs and BF16 output, got different types";
            return std::nullopt;
        }

        // linalg.yield %15 : bf16

        auto returnOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());

        if (!returnOp) {
            failureReason = "expected linalg.yield terminator";
            return std::nullopt;
        }

        // %15 = arith.select %14, %cst_3, %13 : bf16

        auto select2Op = returnOp.getValues()[0].getDefiningOp<arith::SelectOp>();

        if (!select2Op) {
            failureReason = "expected arith.select operation as final result";
            return std::nullopt;
        }

        auto maxVal = select2Op.getTrueValue();

        // ensure maxVal is not defined in the genericOp
        if (maxVal.getParentRegion() == &genericOp.getBodyRegion()) {
            failureReason = "maxVal should not be defined inside genericOp body";
            return std::nullopt;
        }

        // %14 = arith.cmpf ugt, %13, %cst_3 : bf16

        auto cmp2Op = select2Op.getCondition().getDefiningOp<arith::CmpFOp>();
        if (!cmp2Op) {
            failureReason = "expected arith.cmpf operation as select condition";
            return std::nullopt;
        }

        if (cmp2Op.getPredicate() != arith::CmpFPredicate::UGT) {
            failureReason = "expected UGT predicate in cmp2";
            return std::nullopt;
        }

        if (cmp2Op.getLhs() != select2Op.getFalseValue()) {
            failureReason = "cmp2 lhs must match select2 false value";
            return std::nullopt;
        }

        if (cmp2Op.getRhs() != select2Op.getTrueValue()) {
            failureReason = "cmp2 rhs must match select2 true value";
            return std::nullopt;
        }

        // %13 = arith.select %12, %cst_1, %11 : bf16

        auto select1Op = cmp2Op.getLhs().getDefiningOp<arith::SelectOp>();

        if (!select1Op) {
            failureReason = "expected arith.select operation for cmp2 lhs";
            return std::nullopt;
        }

        auto minVal = select1Op.getTrueValue();

        // ensure minVal is not defined inside genericOp
        if (minVal.getParentRegion() == &genericOp.getBodyRegion()) {
            failureReason = "minVal should not be defined inside genericOp body";
            return std::nullopt;
        }

        // %12 = arith.cmpf ult, %11, %cst_1 : bf16

        auto cmp1Op = select1Op.getCondition().getDefiningOp<arith::CmpFOp>();
        if (!cmp1Op) {
            failureReason = "expected arith.cmpf operation as select1 condition";
            return std::nullopt;
        }

        if (cmp1Op.getPredicate() != arith::CmpFPredicate::ULT) {
            failureReason = "expected ULT predicate in cmp1";
            return std::nullopt;
        }

        if (cmp1Op.getLhs() != select1Op.getFalseValue()) {
            failureReason = "cmp1 lhs must match select1 false value";
            return std::nullopt;
        }

        if (cmp1Op.getRhs() != select1Op.getTrueValue()) {
            failureReason = "cmp1 rhs must match select1 true value";
            return std::nullopt;
        }

        // %11 = arith.truncf %10 : f32 to bf16

        auto truncOp = cmp1Op.getLhs().getDefiningOp<arith::TruncFOp>();

        if (!truncOp) {
            failureReason = "expected arith.truncf operation";
            return std::nullopt;
        }

        // %10 = arith.addf %in, %in_4 : f32

        auto addOp = truncOp.getIn().getDefiningOp<arith::AddFOp>();

        if (!addOp) {
            failureReason = "expected arith.addf operation as truncf input";
            return std::nullopt;
        }

        auto in1 = dyn_cast<BlockArgument>(addOp.getLhs());

        if (!in1) {
            failureReason = "addf lhs must be a block argument";
            return std::nullopt;
        }

        if (in1.getArgNumber() != 0) {
            failureReason = "addf lhs must be the first block argument (arg 0)";
            return std::nullopt;
        }

        auto in2 = dyn_cast<BlockArgument>(addOp.getRhs());

        if (!in2) {
            failureReason = "addf rhs must be a block argument";
            return std::nullopt;
        }

        auto bias = genericOp.getDpsInputOperand(1)->get();

        return FuseBiasMatcher{bias, minVal, maxVal};
    }
};

ScaleClampInfo foldForwardScaleClamp(
    Value &value, int scaleValuesCount, int shift8b, int shift16b, bool isElementWiseOp
) {
    ScaleClampInfo sci;
    auto valueType = dyn_cast<ShapedType>(value.getType());
    if (!valueType) {
        LLVM_DEBUG({ llvm::errs() << "matching error value is not a ShapedType!\n"; });
        return {};
    }

    if (llvm::isa<FloatType>(valueType.getElementType())) {
        return foldForwardFloatTruncClamp(value);
    }

    linalg::GenericOp genericOp = getSingleUser<linalg::GenericOp>(value);
    if (!genericOp) {
        // close to scaleclamp generic op to check expandshapeop for integer dtype
        if (value.hasOneUse() && isa<tensor::ExpandShapeOp>(*value.getUsers().begin())) {
            value = value.getUsers().begin()->getResult(0);
            genericOp = getSingleUser<linalg::GenericOp>(value);
        }
        if (!genericOp) {
            LLVM_DEBUG({
                llvm::dbgs(
                ) << "must have a single GenericOp scaleClampOp, we return empty object for "
                     "caller to further check\n";
            });
            return {};
        }
    }

    // expects one init, the value of p
    if (genericOp.getNumDpsInits() != 1) {
        LLVM_DEBUG({ llvm::dbgs() << "matching error init is not 1!\n"; });
        return {};
    }

    // expects three inputs: data, multiplier and shift
    // multiplier and shift can be inputs or scalar constants
    const int dpsInputCount = genericOp.getNumDpsInputs();
    if (dpsInputCount < 1 || dpsInputCount > 3) {
        LLVM_DEBUG({ llvm::dbgs() << "matching error inputs count: " << dpsInputCount << "\n"; });
        return {};
    }

    // this is an element-wise operation
    if (genericOp.getNumReductionLoops() > 0) {
        LLVM_DEBUG({ llvm::dbgs() << "matching error reduction loops > 0!\n"; });
        return {};
    }

    auto initOpOperand = genericOp.getDpsInitOperand(0);
    auto initType = dyn_cast<ShapedType>(initOpOperand->get().getType());
    if (!initType) {
        LLVM_DEBUG({ llvm::dbgs() << "matching error init type is not a ShapedType!\n"; });
        return {};
    }

    // match the body
    auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp) {
        LLVM_DEBUG({ llvm::dbgs() << "matching error yieldOp is not a yield!\n"; });
        return {};
    }

    auto truncOp = yieldOp.getValues()[0].getDefiningOp<arith::TruncIOp>();
    if (!truncOp) {
        LLVM_DEBUG({ llvm::dbgs() << "matching error yieldOp is not a trunc!\n"; });
        return {};
    }
    auto minOp = truncOp.getIn().getDefiningOp<arith::MinSIOp>();
    if (!minOp) {
        LLVM_DEBUG({ llvm::dbgs() << "matching error yieldOp is not a min!\n"; });
        return {};
    }
    // The max constant is used in the min operation
    auto maybeMaxConst = getConstIntValue(minOp.getRhs());
    if (!maybeMaxConst) {
        LLVM_DEBUG({ llvm::dbgs() << "matching error minOp.getRhs() is not a constant!\n"; });
        return {};
    }
    sci.max = *maybeMaxConst;

    auto maxOp = minOp.getLhs().getDefiningOp<arith::MaxSIOp>();
    if (!maxOp) {
        LLVM_DEBUG({ llvm::dbgs() << "matching error minOp.getLhs() is not a max!\n"; });
        return {};
    }
    // The min constant is used in the max operation
    auto maybeMinConst = getConstIntValue(maxOp.getRhs());
    if (!maybeMinConst) {
        LLVM_DEBUG({ llvm::dbgs() << "matching error maxOp.getRhs() is not a constant!\n"; });
        return {};
    }
    sci.min = *maybeMinConst;

    // Adding zero-point is optional.
    auto addOp = maxOp.getLhs().getDefiningOp<arith::AddIOp>();
    auto applyScaleOp = maxOp.getLhs().getDefiningOp<tosa::ApplyScaleOp>();
    if (addOp) {
        auto maybeZeroPoint = getConstIntValue(addOp.getRhs());
        if (!maybeZeroPoint) {
            LLVM_DEBUG({ llvm::dbgs() << "matching error addOp.getRhs() is not a constant!\n"; });
            return {};
        }
        sci.zp = *maybeZeroPoint;
        applyScaleOp = addOp.getLhs().getDefiningOp<tosa::ApplyScaleOp>();
    }

    if (!applyScaleOp) {
        LLVM_DEBUG({ llvm::dbgs() << "matching error addOp.getLhs() is not an ApplyScale!\n"; });
        return {};
    }

    auto inputData = applyScaleOp.getValue();

    if (auto subOp = inputData.getDefiningOp<arith::SubIOp>()) {
        inputData = subOp.getLhs();
        auto maybeBias = getConstIntValue(subOp.getRhs());
        if (!maybeBias) {
            LLVM_DEBUG({ llvm::dbgs() << "matching error subOp.getRhs() is not a constant!\n"; });
            return {};
        }
        // This is an additional bias applied before scaling
        sci.bias = *maybeBias;
    }

    auto dataArg = dyn_cast<BlockArgument>(inputData /*applyScaleOp.getValue()*/);
    if (!dataArg) {
        LLVM_DEBUG({
            llvm::dbgs() << "matching error applyScaleOp.getValue() is not a BlockArgument!\n";
        });
        return {};
    }

    // Extract TOSA multiplier and shift values (these are used in scale computation)
    auto ms = getMultiplierAndShift(genericOp, applyScaleOp, scaleValuesCount);
    if (!ms) {
        LLVM_DEBUG({
            llvm::dbgs() << "matching error applyScaleOp multiplier and shift not found!\n";
        });
        return {};
    }

    // Use operation-specific shift value strategy
    auto elementType = initType.getElementType();
    if (elementType.isInteger(8) || elementType.isInteger(16)) {
        if (isElementWiseOp) {
            // For element-wise operations (add/sub), use hardcoded shift values for stability
            if (elementType.isInteger(8)) {
                sci.scaleShift = shift8b;
            }
            else { // 16-bit
                sci.scaleShift = shift16b;
            }
        }
        else {
            // For other operations (convolution), use the smallest TOSA shift value
            // to minimize overflow risk while preserving original quantization parameters
            sci.scaleShift = *std::min_element(ms.shift.begin(), ms.shift.end());
        }
    }
    else {
        LLVM_DEBUG({
            llvm::dbgs() << "matching error output type is not supported\n";
            elementType.dump();
        });
        return {};
    }
    sci.scaleNpu = compute_scale(ms.multiplier, ms.shift, sci.scaleShift);
    sci.scaleDouble = computeScaleDouble(ms.multiplier, ms.shift);

    value = genericOp.getResults()[0];

    // for some int16 conv2d case, there is a following extra clamp generic op for relu-like
    // operation
    linalg::GenericOp extraClampOp = dyn_cast_or_null<linalg::GenericOp>(getSingleUser(value));
    if (extraClampOp) {
        // in general, compare with constant, so input number is 1
        if (extraClampOp.getNumDpsInputs() == 1 && extraClampOp.getNumDpsInits() == 1) {
            auto yieldOp =
                dyn_cast_or_null<linalg::YieldOp>(extraClampOp.getBody()->getTerminator());
            if (yieldOp) {
                if (auto maxOp = yieldOp.getValues()[0].getDefiningOp<arith::MaxSIOp>()) {
                    // The min constant is used in the max operation
                    if (auto maybeMinConst = getConstIntValue(maxOp.getRhs())) {
                        sci.min = *maybeMinConst;
                    }

                    // if maxOp lhs is input, means block finished, we will change output value and
                    // return, if not, there is other op, maybe config clamp max value we dont' have
                    // this kind of case now, need to do it later
                    auto lhs = dyn_cast_or_null<BlockArgument>(maxOp.getLhs());
                    if (lhs) {
                        value = extraClampOp.getResults()[0];
                    }
                    else {
                        LLVM_DEBUG({
                            llvm::errs()
                                << "REMINDER: there are more op for clamp, please check!\n";
                        });
                    }
                }
                else {
                    LLVM_DEBUG({
                        llvm::errs() << "REMINDER: there is other op for clamp, please check!\n";
                    });
                }
            }
        }
    }

    return sci;
}

// Zero point for convolution weights is normally expressed as a linalg generic taking in input
// the conv output tensor and a tensor computed by pooling_nhwc_sum which is multiplied by the wzp
// Example:
/*
  %5 = linalg.depthwise_conv_2d_nhwc_hwc ins(%padded)
  %7 = linalg.fill ins(%c0_i32 : i32) -> tensor<32xi32>
  %8 = linalg.generic
  ^bb0(%in: i8, %out: i32):
    %17 = arith.extsi %in : i8 to i32
    %18 = arith.addi %17, %out : i32
    linalg.yield %18 : i32
  } -> tensor<32xi32>
  %9 = linalg.generic ins(%5, %8)
  ^bb0(%in: i32, %in_3: i32, %out: i32):
    %17 = arith.muli %in_3, %c-128_i32 : i32
    %18 = arith.subi %in, %17 : i32
    linalg.yield %18 : i32
  } -> tensor<1x112x112x32xi32>
  %10 = tensor.empty() : tensor<3x3xi32>
  %11 = linalg.pooling_nhwc_sum  ins(%padded, %10)
  %12 = linalg.generic { ins(%9, %11) {
  ^bb0(%in: i32, %in_3: i32, %out: i32):
    %17 = arith.muli %in_3, %c33_i32 : i32
    %18 = arith.subi %in, %17 : i32
    linalg.yield %18 : i32
  } -> tensor<1x112x112x32xi32>
*/
int foldForwardWeightZp(Value &value) {
    linalg::GenericOp genericOp = getSingleUser<linalg::GenericOp>(value);
    if (!genericOp) {
        return 0;
    }
    if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumReductionLoops() > 0) {
        return 0;
    }
    // This could be the linalg generic that multiplies the pooled input by the wzp before
    // subtracting it from conv output. If this is the case we can get the wzp from the muli instr.
    auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp) {
        return 0;
    }
    auto subOp = yieldOp.getValues()[0].getDefiningOp<arith::SubIOp>();
    if (!subOp) {
        return 0;
    }
    auto muliOp = subOp.getRhs().getDefiningOp<arith::MulIOp>();
    if (!muliOp) {
        return 0;
    }
    auto maybeWzp = getConstIntValue(muliOp.getRhs());
    if (!maybeWzp) {
        return 0;
    }

    // Be sure that the other input of the muli is the result of a pooling_nhwc_sum
    BlockArgument muliLhs = dyn_cast_or_null<BlockArgument>(muliOp.getLhs());
    if (!muliLhs) {
        return 0;
    }
    auto srcOpnd = genericOp.getMatchingOpOperand(muliLhs)->get();
    if (!srcOpnd) {
        return 0;
    }
    auto muliLhsOp = srcOpnd.getDefiningOp();
    if (tensor::CollapseShapeOp collapsed = dyn_cast_or_null<tensor::CollapseShapeOp>(muliLhsOp)) {
        muliLhsOp = collapsed.getSrc().getDefiningOp();
    }
    auto poolOp = dyn_cast_or_null<linalg::PoolingNhwcSumOp>(muliLhsOp);
    if (!poolOp) {
        return 0;
    }

    // TODO: be sure that the other input of the subi is the conv output

    value = genericOp.getResults()[0];
    return *maybeWzp;
}

static LogicalResult foldLinalgFillTensorInsert(
    Value &value, SmallVector<int64_t> &padOffsetsAbove, SmallVector<int64_t> &padOffsetsBelow,
    Value &fillValue
) {

    auto insertSliceOp = value.getDefiningOp<tensor::InsertSliceOp>();

    if (!insertSliceOp) {
        return failure();
    }

    auto padDestTensor = insertSliceOp.getDest();

    auto inputFillOp = padDestTensor.getDefiningOp<linalg::FillOp>();

    if (!inputFillOp) {
        return failure();
    }

    if (llvm::any_of(
            insertSliceOp.getStaticOffsets(),
            [](int64_t offs) { return ShapedType::isDynamic(offs); }
        ) ||
        llvm::any_of(insertSliceOp.getStaticSizes(), [](int64_t size) {
            return ShapedType::isDynamic(size);
        })) {
        LLVM_DEBUG({ llvm::dbgs() << "Dynamic pad offsets sizes, or strides not supported\n"; });
        return failure();
    }

    if (llvm::any_of(insertSliceOp.getStaticStrides(), [](int64_t stride) {
            return stride != 1;
        })) {
        return failure();
    }

    auto destShape = cast<RankedTensorType>(padDestTensor.getType()).getShape();

    auto staticOffsets = insertSliceOp.getStaticOffsets();

    for (auto [i, size] : llvm::enumerate(insertSliceOp.getStaticSizes())) {
        padOffsetsAbove.push_back(staticOffsets[i]);
        padOffsetsBelow.push_back(destShape[i] - size - staticOffsets[i]);
    }

    if (inputFillOp.getInputs().size() != 1) {
        return failure();
    }

    fillValue = inputFillOp.getInputs()[0];

    value = insertSliceOp.getSource();

    return success();
}

static LogicalResult foldTensorPad(
    Value &value, SmallVector<int64_t> &padOffsetsAbove, SmallVector<int64_t> &padOffsetsBelow,
    Value &fillValue
) {

    auto padOp = value.getDefiningOp<tensor::PadOp>();

    if (!padOp) {
        return failure();
    }

    // find the value of low/hi, they may be static or constant values
    for (int i = 0; i < padOp.getSource().getType().getRank(); i++) {

        auto maybeLow = getConstantIntValue(padOp.getMixedLowPad()[i]);
        auto maybeHigh = getConstantIntValue(padOp.getMixedHighPad()[i]);

        if (!maybeLow || !maybeHigh) {
            // Reset the input arguments to their original state before failing
            padOffsetsAbove.pop_back_n(i);
            padOffsetsBelow.pop_back_n(i);

            return failure();
        }

        padOffsetsAbove.push_back(*maybeLow);
        padOffsetsBelow.push_back(*maybeHigh);
    }

    auto yieldOp = cast<tensor::YieldOp>(padOp.getBody()->getTerminator());

    fillValue = yieldOp.getValue();

    value = padOp.getSource();

    return success();
}

PaddingInfo foldBackwardPadding(Value &value, PatternRewriter &rewriter, bool nchw) {
    // Process any extract_slice op and check there is no dynamic slice extraction
    Value val = value;
    SmallVector<tensor::ExtractSliceOp> extractSliceOps;
    // TODO: remove and check if anything breaks
    while (auto extractSliceOp = val.getDefiningOp<tensor::ExtractSliceOp>()) {
        // TODO: can abort marking here
        extractSliceOps.push_back(extractSliceOp);
        auto offsets = extractSliceOp.getStaticOffsets();
        auto sizes = extractSliceOp.getStaticSizes();
        auto strides = extractSliceOp.getStaticStrides();
        if (llvm::any_of(offsets, [](int64_t offset) { return ShapedType::isDynamic(offset); }) ||
            llvm::any_of(sizes, [](int64_t size) { return ShapedType::isDynamic(size); }) ||
            llvm::any_of(strides, [](int64_t stride) { return stride != 1; })) {
            LLVM_DEBUG({ llvm::dbgs() << "Dynamic offsets, sizes, or strides not supported\n"; });
            break;
        }
        val = extractSliceOp.getSource();
    }

    // The padding is implemented with a tensor::InsertSliceOp
    SmallVector<int64_t> padOffsetsAbove;
    SmallVector<int64_t> padOffsetsBelow;
    Value fillValue;

    if (failed(foldTensorPad(val, padOffsetsAbove, padOffsetsBelow, fillValue))) {
        if (failed(foldLinalgFillTensorInsert(val, padOffsetsAbove, padOffsetsBelow, fillValue))) {
            return {};
        }
    }

    if (padOffsetsBelow.size() != 4) {
        LLVM_DEBUG({ llvm::dbgs() << "Padding supported only for 4-D tensors\n"; });
        return {};
    }

    // Check there is no padding except on the H and W dimensions
    int hDim = 1, wDim = 2;
    if (nchw) {
        hDim = 2;
        wDim = 3;
    }

    for (size_t i = 0; i < padOffsetsBelow.size(); ++i) {
        if (i == hDim || i == wDim) {
            continue;
        }
        if (padOffsetsBelow[i] != 0 || padOffsetsAbove[i] != 0) {
            // We are only able to handle padding on the H and W dimensions of 4D tensors
            LLVM_DEBUG({ llvm::dbgs() << "Padding in an unsupported dimension\n"; });
            return {};
        }
    }

    // Check the padding border comes from a constant FillOp
    std::optional<int64_t> maybeFillValue = getConstIntValue(fillValue);

    if (!maybeFillValue) {
        auto maybeFloatFillValue = getFloatValue(fillValue);

        if (!maybeFloatFillValue) {
            LLVM_DEBUG({ llvm::dbgs() << "Cannot find supported fill value\n"; });
            return {};
        }

        float floatFillValue = *maybeFloatFillValue;
        maybeFillValue = *reinterpret_cast<int32_t *>(&floatFillValue);
    }

    // Compute the padding applied to the source tensor
    int32_t top = padOffsetsAbove[hDim];
    int32_t left = padOffsetsAbove[wDim];
    int32_t bottom = padOffsetsBelow[hDim];
    int32_t right = padOffsetsBelow[wDim];

    // TODO: see above; this should be removed.
    // If we found extract_slice ops we have to update them.
    // We currently need to support only horizontal slice extraction (used for tiling)
    if (!extractSliceOps.empty()) {
        for (tensor::ExtractSliceOp extractOp : llvm::reverse(extractSliceOps)) {
            // Adjust the offsets and sizes to account for the padding
            SmallVector<int64_t> offsets{extractOp.getStaticOffsets()};
            SmallVector<int64_t> sizes{extractOp.getStaticSizes()};
            auto strides = extractOp.getStaticStrides();
            auto extractSrcShape = cast<RankedTensorType>(extractOp.getSourceType()).getShape();

            // Adjust top/bottom padding
            if (offsets[hDim] + sizes[hDim] <= extractSrcShape[hDim] - bottom) {
                // The bottom of the y-tile is above the padded area: disable bottom pad in conv
                bottom = 0;
            }
            else if (offsets[hDim] + sizes[hDim] == extractSrcShape[hDim]) {
                // The bottom of the y-tile includes the padded area: remove padding from the tile
                // since bottom padding will be done by the kernel
                sizes[hDim] -= bottom;
            }
            else {
                assert(false && "Unexpected padding bottom");
            }

            if (offsets[hDim] >= top) {
                // The start of the y-tile is after the padded area: disable top pad in conv
                offsets[hDim] -= top; // TBV: do we need this?
                top = 0;
            }
            else if (offsets[hDim] == 0) {
                // The start of the y-tile includes the padded area: remove padding from the tile
                // since top padding will be done by the kernel
                sizes[hDim] -= top;
            }
            else {
                assert(false && "Unexpected padding top");
            }

            assert(
                offsets[wDim] == 0 && offsets[wDim] + sizes[wDim] == extractSrcShape[wDim] &&
                "Unexpected offsets or sizes for conv extract on w dimension"
            );
            sizes[wDim] -= left;
            sizes[wDim] -= right;

            tensor::ExtractSliceOp newExtractSliceOp = tensor::ExtractSliceOp::create(
                rewriter, value.getLoc(), val, createVector(offsets, rewriter),
                createVector(sizes, rewriter), createVector(strides, rewriter)
            );

            val = newExtractSliceOp.getResult();
        }
    }

    // Update value to fold the padding
    value = val;

    return PaddingInfo{{left, right, top, bottom}, maybeFillValue.value()};
}

std::vector<int32_t> computePerChannelValuesInt(Value constV, int channelDim) {
    auto constAttr = returnDenseElementAttr(constV);
    auto shape = constAttr.getType().getShape();
    const int64_t numChannels = shape[channelDim];
    std::vector<int32_t> perChannelValues(numChannels);
    std::vector<uint8_t> hasValue(numChannels);
    int64_t channelSize = 1;
    for (int64_t i = channelDim + 1; i < shape.size(); ++i) {
        channelSize *= shape[i];
    }

    int64_t i = 0;
    for (auto value : constAttr.getValues<int32_t>()) {
        auto channel = (i++ / channelSize) % numChannels;
        if (!hasValue[channel]) {
            perChannelValues[channel] = value;
            hasValue[channel] = true;
        }
        else if (perChannelValues[channel] != value) {
            // The constant tensor has different values for the same channel
            return {};
        }
    }
    return perChannelValues;
}

std::vector<float> computePerChannelValuesFloat(Value constV, int channelDim) {
    auto constAttr = returnDenseElementAttr(constV);
    auto shape = constAttr.getType().getShape();
    const int64_t numChannels = shape[channelDim];
    std::vector<float> perChannelValues(numChannels);
    std::vector<uint8_t> hasValue(numChannels);
    int64_t channelSize = 1;
    for (int64_t i = channelDim + 1; i < shape.size(); ++i) {
        channelSize *= shape[i];
    }

    int64_t i = 0;
    for (auto apValue : constAttr.getValues<APFloat>()) {
        float value = apValue.convertToFloat();
        auto channel = (i++ / channelSize) % numChannels;
        if (!hasValue[channel]) {
            perChannelValues[channel] = value;
            hasValue[channel] = true;
        }
        else if (perChannelValues[channel] != value) {
            // TODO: maybe we should use a tolerance to the comparison
            // The constant tensor has different values for the same channel
            return {};
        }
    }
    return perChannelValues;
}

template <typename T> bool elementwiseAdd(std::vector<T> &result, const std::vector<T> &v1) {
    if (result.size() != v1.size()) {
        return false;
    }
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] += v1[i];
    }
    return true;
}

std::optional<ScaleClampInfo> foldFusedForwardPerChannelAddClamp(
    Value &value, int channelDim, VectorIntOrFloat &bias, std::string &failReason
) {

    ScaleClampInfo sci;

    auto genericOp = getSingleUser<linalg::GenericOp>(value);

    if (!genericOp) {
        failReason = "no single linalg.generic user";
        return std::nullopt;
    }

    auto matcher = FuseBiasMatcher::match(genericOp.getOperation(), failReason);

    if (!matcher) {
        return std::nullopt;
    }

    auto max = getFloatValue(matcher->clampMax);

    if (!max) {
        failReason = "no max value";
        return std::nullopt;
    }

    auto min = getFloatValue(matcher->clampMin);

    if (!min) {
        failReason = "no min value";
        return std::nullopt;
    }

    sci.max = llvm::bit_cast<uint32_t>(*max);
    sci.min = llvm::bit_cast<uint32_t>(*min);

    if (!bias.floats.size()) {
        failReason = "bias floats size is zero";
        return std::nullopt;
    }

    matcher->clampMax.dump();
    matcher->clampMin.dump();

    auto broadcastedBiasConst = computeArithConst(matcher->bias, true);

    if (failed(broadcastedBiasConst)) {
        failReason = "no broadcasted bias constant attribute";
        return std::nullopt;
    }

    const auto &perChannelBiasValues =
        computePerChannelValuesFloat(broadcastedBiasConst.value(), channelDim);

    if (!elementwiseAdd(bias.floats, perChannelBiasValues)) {
        failReason = "elementwise add failed";
        return std::nullopt;
    }

    value = genericOp.getResults()[0];

    return sci;
}

bool foldForwardPerChannelAdd(
    Value &value, int channelDim, VectorIntOrFloat &bias, int32_t *input_zp, Value inValue,
    int32_t *w_zp
) {
    auto valueType = dyn_cast<ShapedType>(value.getType());
    if (!valueType) {
        LLVM_DEBUG({ llvm::errs() << "matching error value is not a ShapedType!\n"; });
        return {};
    }

    // bf16 there is truncfOp for fp32 to bf16, fold truncfOp
    if (llvm::isa<FloatType>(valueType.getElementType())) {
        foldForwardTruncFOp(value);
    }

    linalg::GenericOp addOp = getSingleUser<linalg::GenericOp>(value);
    if (!addOp || addOp.getNumDpsInputs() > (inValue ? 3 : 2)) {
        // Normal case is two inputs, one for the value to add to and one for the bias
        // in some cases the bias is a constant scalar, in this case we have only one input
        // in some other cases when the weights have a zero point we have a third input
        return false;
    }

    // Check that addOp represents an Add or Sub operation
    auto yieldOp = dyn_cast<linalg::YieldOp>(addOp.getBody()->getTerminator());
    if (!yieldOp) {
        LLVM_DEBUG({ llvm::dbgs() << "foldForwardPerChannelAdd No yield op\n"; });
        return false;
    }
    auto lastOp = yieldOp.getValues()[0].getDefiningOp();
    if (!lastOp) {
        return false;
    }

    // in case the addition has the form:
    //
    //     %10 = arith.addf %in, %in_3 : f32
    //     %11 = arith.truncf %10 : f32 to bf16
    //     linalg.yield %11 : bf16
    //
    // we find the addf op behind the truncf op
    if (auto truncfOp = dyn_cast<arith::TruncFOp>(lastOp)) {
        if (truncfOp.getIn().getType().isF32() && truncfOp.getType().isBF16()) {
            lastOp = truncfOp.getIn().getDefiningOp();
        }
    }

    if (bias.ints.size() && !dyn_cast<arith::AddIOp>(lastOp) && !dyn_cast<arith::SubIOp>(lastOp)) {
        LLVM_DEBUG({ llvm::dbgs() << "foldForwardPerChannelAdd No AddIOp or SubIOp\n"; });
        return false;
    }
    if (bias.floats.size() && !dyn_cast<arith::AddFOp>(lastOp) &&
        !dyn_cast<arith::SubFOp>(lastOp)) {
        LLVM_DEBUG({ llvm::dbgs() << "foldForwardPerChannelAdd No AddFOp or SubFOp\n"; });
        return false;
    }
    // TODO: check that one of the operand is coming from value
    // TODO: check that one of the operand is coming from inValue if specified

    // Compute the result of the add operation assuming value is 0.
    // Also zero-out the DPS init (outs) operands: for a pure elementwise op
    // they are completely overwritten, so their provenance is irrelevant, but
    // outlineAndReturnOps would otherwise try to trace them back and fail when
    // they originate from a dispatch input load.
    SmallVector<Value> zeros = {inValue, value};
    zeros.append(addOp.getDpsInits().begin(), addOp.getDpsInits().end());
    auto maybeConstV = computeArithConst(addOp, true, zeros);
    if (failed(maybeConstV)) {
        LLVM_DEBUG({
            llvm::dbgs() << "foldForwardPerChannelAdd computeArithConst in:"
                         << addOp.getNumDpsInputs() << " FAILED\n";
        });
        return false;
    }

    // Check that all the values in each channel have the same value
    // and update the per-channel bias values
    if (bias.ints.size()) {
        const auto &perChannel = computePerChannelValuesInt(*maybeConstV, channelDim);
        if (!elementwiseAdd(bias.ints, perChannel)) {
            // The constant tensor has unexpected size
            LLVM_DEBUG({ llvm::dbgs() << "foldForwardPerChannelAdd int tensor bad size\n"; });
            return false;
        }
    }
    else if (bias.floats.size()) {
        const auto &perChannel = computePerChannelValuesFloat(*maybeConstV, channelDim);
        if (!elementwiseAdd(bias.floats, perChannel)) {
            // The constant tensor has unexpected size
            LLVM_DEBUG({ llvm::dbgs() << "foldForwardPerChannelAdd float tensor bad size\n"; });
            return false;
        }
    }
    else {
        LLVM_DEBUG({ llvm::dbgs() << "foldForwardPerChannelAdd tensor no int no float\n"; });
        return false;
    }

    if (inValue && addOp.getNumDpsInputs() == 3 && w_zp != nullptr) {
        // When there are 3 inputs it means the weights have a zero point
        auto addWZpByInZp = dyn_cast_or_null<arith::AddIOp>(lastOp);
        if (!addWZpByInZp) {
            // Last op is supposed to be adding the wzp*input_zp to the result
            assert(false && "Last op is supposed to be adding the wzp*input_zp to the result");
            return false;
        }
        // Ignore this add and go back to its operand (which should subtract the w*input_zp)
        lastOp = addWZpByInZp.getLhs().getDefiningOp();
    }

    // extract input_zp
    // input zp only valid for int8, so we only check muli op in pattern
    // %12 = linalg.generic {...} ins(%8, %11 : tensor<1x1000xi32>, tensor<1000xi32>) outs(%4 :
    // tensor<1x1000xi32>) { ^bb0(%in: i32, %in_1: i32, %out: i32):
    //     %15 = arith.muli %in_1, %c-128_i32 : i32
    //     %16 = arith.subi %in, %15 : i32
    //     linalg.yield %16 : i32
    // } -> tensor<1x1000xi32>
    // Note: the arith.muli op can be omitted if the corresponding zeroPoint is 1
    if (input_zp != nullptr) {
        *input_zp = 0;
        auto subiOp = dyn_cast_or_null<arith::SubIOp>(lastOp);
        if (subiOp) {
            // First op is supposed to be subtracting the weight*input_zp to the result
            auto opnd1 = subiOp.getOperand(1);
            if (auto op1 = opnd1.getDefiningOp()) {
                auto muliOp = dyn_cast_or_null<arith::MulIOp>(op1);
                auto maybeInputZp = getConstIntValue(muliOp.getRhs());
                if (maybeInputZp.has_value()) {
                    *input_zp = *maybeInputZp;
                }
                lastOp = subiOp.getLhs().getDefiningOp();
            }
            else {
                // No arith.muli, zero point must be 1
                *w_zp = 1;
            }
        }
        if (inValue && addOp.getNumDpsInputs() == 3 && w_zp != nullptr) {
            // When there are 3 inputs it means the weights have a zero point
            // We can extract the wzp from the first muli operation in the linalg.generic body:
            // %18 = arith.muli %in_1, %c12_i32 : i32 // %in_1: perChAdd(in), %c12: weights
            // zero-point %19 = arith.subi %in, %18 : i32 %20 = arith.muli %in_2, %c-128_i32 : i32
            // // %in_2: perChAdd(w), %c-128: in zero-point %21 = arith.subi %19, %20 : i32 %22 =
            // arith.addi %21, %c-49152_i32 : i32 // %c-49152: weights zero-point * in zero-point
            // linalg.yield %22 : i32
            // Note: the arith.muli ops can be omitted if the corresponding zeroPoint is 1

            auto subInByWZp = dyn_cast_or_null<arith::SubIOp>(lastOp);
            if (!subInByWZp) {
                // First op is supposed to be subtracting the input*weight_zp to the result
                return false;
            }
            auto opnd1 = subInByWZp.getOperand(1);
            if (auto op1 = opnd1.getDefiningOp()) {
                auto muliOp = dyn_cast_or_null<arith::MulIOp>(op1);
                auto maybeWeightZp = getConstIntValue(muliOp.getRhs());
                if (maybeWeightZp.has_value()) {
                    *w_zp = *maybeWeightZp;
                }
            }
            else {
                // No arith.muli, zero point must be 1
                *w_zp = 1;
            }
        }
    }

    // Update value to fold the operation
    value = addOp.getResultTensors()[0];
    return true;
}

using RegionComputationFn = std::function<APIntOrFloat(const APIntOrFloatArray &)>;

static RegionComputationFn getValueComputeFn(LinalgOp linalgOp, Value value) {

    if (auto blockArg = dyn_cast<BlockArgument>(value)) {

        if (blockArg.getOwner() != linalgOp.getBlock()) {
            // the value is an argument to another block, we don't support this
            return nullptr;
        }

        return [blockArg](const APIntOrFloatArray &inputs) {
            if (!blockArg.getType().isInteger()) {
                return APIntOrFloat{std::nullopt, inputs.apFloats[blockArg.getArgNumber()]};
            }
            else {
                return APIntOrFloat{inputs.apInts[blockArg.getArgNumber()], std::nullopt};
            }
        };
    }
    else if (!value.getDefiningOp()) {
        // the value is an argument to another block, we don't support this
        return nullptr;
    }
    else {

        auto op = value.getDefiningOp();

        if (auto yieldOp = dyn_cast<linalg::YieldOp>(op)) {

            // support only one yield value
            if (yieldOp.getValues().size() != 1) {
                return nullptr;
            }

            RegionComputationFn valueFn = getValueComputeFn(linalgOp, yieldOp.getValues()[0]);

            return [valueFn](const APIntOrFloatArray &inputs) { return valueFn(inputs); };
        }
        else if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {

            if (auto fVal = dyn_cast<FloatAttr>(constOp.getValueAttr())) {
                return [fVal](const APIntOrFloatArray &inputs) {
                    return APIntOrFloat{std::nullopt, fVal.getValue()};
                };
            }
            else if (auto iVal = dyn_cast<IntegerAttr>(constOp.getValueAttr())) {
                return [iVal](const APIntOrFloatArray &inputs) {
                    return APIntOrFloat{iVal.getValue(), std::nullopt};
                };
            }
            else {
                return nullptr;
            }
        }
        else if (auto sitofpOp = dyn_cast<arith::SIToFPOp>(op)) {

            RegionComputationFn inFn = getValueComputeFn(linalgOp, sitofpOp.getIn());

            if (!inFn) {
                return nullptr;
            }

            FloatType floatTy = cast<FloatType>(sitofpOp.getType());

            const llvm::fltSemantics &floatSemantic = floatTy.getFloatSemantics();
            auto width = floatTy.getWidth();

            return [inFn, &floatSemantic, width](const APIntOrFloatArray &inputs) {
                APFloat apf(floatSemantic, APInt::getZero(width));
                apf.convertFromAPInt(
                    inFn(inputs).apInt.value(), /*IsSigned=*/true, APFloat::rmNearestTiesToEven
                );
                return APIntOrFloat{std::nullopt, apf};
            };
        }
        else if (auto extSiOp = dyn_cast<arith::ExtSIOp>(op)) {

            RegionComputationFn inFn = getValueComputeFn(linalgOp, extSiOp.getIn());

            if (!inFn) {
                return nullptr;
            }

            int width = extSiOp.getType().getIntOrFloatBitWidth();

            return [inFn, width](const APIntOrFloatArray &inputs) {
                return APIntOrFloat{inFn(inputs).apInt.value().sext(width), std::nullopt};
            };
        }
        else if (auto shliOp = dyn_cast<arith::ShLIOp>(op)) {

            RegionComputationFn lhsFn = getValueComputeFn(linalgOp, shliOp.getLhs());
            RegionComputationFn rhsFn = getValueComputeFn(linalgOp, shliOp.getRhs());

            if (!rhsFn || !lhsFn) {
                return nullptr;
            }

            return [lhsFn, rhsFn](const APIntOrFloatArray &inputs) {
                return APIntOrFloat{
                    lhsFn(inputs).apInt.value().shl(rhsFn(inputs).apInt.value()), std::nullopt
                };
            };
        }
        else if (auto divfOp = dyn_cast<arith::DivFOp>(op)) {

            RegionComputationFn lhsFn = getValueComputeFn(linalgOp, divfOp.getLhs());
            RegionComputationFn rhsFn = getValueComputeFn(linalgOp, divfOp.getRhs());

            if (!rhsFn || !lhsFn) {
                return nullptr;
            }

            return [lhsFn, rhsFn](const APIntOrFloatArray &inputs) {
                return APIntOrFloat{
                    std::nullopt, lhsFn(inputs).apFloat.value() / rhsFn(inputs).apFloat.value()
                };
            };
        }
        else if (auto mulfOp = dyn_cast<arith::AddIOp>(op)) {

            RegionComputationFn lhsFn = getValueComputeFn(linalgOp, mulfOp.getLhs());
            RegionComputationFn rhsFn = getValueComputeFn(linalgOp, mulfOp.getRhs());

            if (!rhsFn || !lhsFn) {
                return nullptr;
            }

            return [lhsFn, rhsFn](const APIntOrFloatArray &inputs) {
                return APIntOrFloat{
                    lhsFn(inputs).apInt.value() + rhsFn(inputs).apInt.value(), std::nullopt
                };
            };
        }
        else if (auto mulfOp = dyn_cast<arith::MulIOp>(op)) {

            RegionComputationFn lhsFn = getValueComputeFn(linalgOp, mulfOp.getLhs());
            RegionComputationFn rhsFn = getValueComputeFn(linalgOp, mulfOp.getRhs());

            if (!rhsFn || !lhsFn) {
                return nullptr;
            }

            return [lhsFn, rhsFn](const APIntOrFloatArray &inputs) {
                return APIntOrFloat{
                    lhsFn(inputs).apInt.value() * rhsFn(inputs).apInt.value(), std::nullopt
                };
            };
        }
        else if (auto subiOp = dyn_cast<arith::SubIOp>(op)) {

            RegionComputationFn lhsFn = getValueComputeFn(linalgOp, subiOp.getLhs());
            RegionComputationFn rhsFn = getValueComputeFn(linalgOp, subiOp.getRhs());

            if (!rhsFn || !lhsFn) {
                return nullptr;
            }

            return [lhsFn, rhsFn](const APIntOrFloatArray &inputs) {
                return APIntOrFloat{
                    lhsFn(inputs).apInt.value() - rhsFn(inputs).apInt.value(), std::nullopt
                };
            };
        }
        else if (auto mulfOp = dyn_cast<arith::MulFOp>(op)) {

            RegionComputationFn lhsFn = getValueComputeFn(linalgOp, mulfOp.getLhs());
            RegionComputationFn rhsFn = getValueComputeFn(linalgOp, mulfOp.getRhs());

            if (!rhsFn || !lhsFn) {
                return nullptr;
            }

            return [lhsFn, rhsFn](const APIntOrFloatArray &inputs) {
                return APIntOrFloat{
                    std::nullopt, lhsFn(inputs).apFloat.value() * rhsFn(inputs).apFloat.value()
                };
            };
        }
        else if (auto mulfOp = dyn_cast<arith::AddFOp>(op)) {
            RegionComputationFn lhsFn = getValueComputeFn(linalgOp, mulfOp.getLhs());
            RegionComputationFn rhsFn = getValueComputeFn(linalgOp, mulfOp.getRhs());
            if (!rhsFn || !lhsFn) {
                return nullptr;
            }
            return [lhsFn, rhsFn](const APIntOrFloatArray &inputs) {
                return APIntOrFloat{
                    std::nullopt, lhsFn(inputs).apFloat.value() + rhsFn(inputs).apFloat.value()
                };
            };
        }
        else if (auto fptosiOp = dyn_cast<arith::FPToSIOp>(op)) {

            RegionComputationFn inFn = getValueComputeFn(linalgOp, fptosiOp.getIn());

            if (!inFn) {
                return nullptr;
            }

            unsigned bitWidth = llvm::cast<IntegerType>(fptosiOp.getType()).getWidth();

            return [inFn, bitWidth](const APIntOrFloatArray &inputs) {
                bool ignored;
                APSInt api(bitWidth, /*isUnsigned=*/false);
                inFn(inputs).apFloat.value().convertToInteger(api, APFloat::rmTowardZero, &ignored);
                return APIntOrFloat{api, std::nullopt};
            };
        }
        else {
            LLVM_DEBUG({ llvm::dbgs() << "unsupported operation: " << op->getName() << "\n"; });
            return nullptr;
        }
    }
}

ScaleClampInfo getDefaultScaleClampInfo(Type outElemType, Operation *srcOp) {
    ScaleClampInfo scInfo;
    scInfo.zp = 0;
    scInfo.scaleNpu = {};
    scInfo.scaleDouble = {};

    bool isInt = outElemType.isInteger();

    if (isInt) {
        unsigned bitWidth = outElemType.getIntOrFloatBitWidth();

        if (bitWidth == 8) {
            scInfo.min = -128;
            scInfo.max = 127;
            scInfo.scaleShift = 28;
        }
        else if (bitWidth == 16) {
            scInfo.min = -32768;
            scInfo.max = 32767;
            scInfo.scaleShift = 12;
        }
        else {
            llvm::report_fatal_error("Unhandled integer bit width for default scale/clamp");
        }
    }
    else {
        // float / bf16 path
        scInfo.min = 0xff800000; // -inf
        scInfo.max = 0x7f800000; // +inf
        scInfo.scaleShift = 0;
    }

    return scInfo;
}

Operation *getElementwiseTernaryOp(linalg::GenericOp op, bool allowConstants) {

    Value output = op.getResultTensors()[0];
    auto rank = cast<RankedTensorType>(output.getType()).getRank();

    // if rank == 0 (scalar) there is no loop
    if (rank > 0 && op.getNumLoops() < 1) {
        LLVM_DEBUG({ llvm::dbgs() << "elementwise ternary op loop number < 1\n"; });
        return {};
    }

    if (op.getNumParallelLoops() != op.getNumLoops()) {
        LLVM_DEBUG({ llvm::dbgs() << "elementwiseTernaryOp expect all loops are parallel\n"; });
        return {};
    }

    // We expect 3 inputs but must also accept 1 or 2 if the same is used for multiple operands
    if (op.getNumDpsInputs() != 3 && op.getNumDpsInputs() != 2 && op.getNumDpsInputs() != 1) {
        LLVM_DEBUG({ llvm::dbgs() << "elementwiseTernaryOp expect 3 inputs (or 2/1)\n"; });
        return {};
    }

    if (op.getNumDpsInits() != 1) {
        LLVM_DEBUG({ llvm::dbgs() << "elementwiseTernaryOp expect 1 init\n"; });
        return {};
    }

    // All inputs are used
    for (int i = 0; i < op.getNumDpsInputs(); i++) {
        if (!op.payloadUsesValueFromOperand(op.getDpsInputOperand(i))) {
            LLVM_DEBUG({ llvm::dbgs() << "elementwiseTernaryOp input " << i << " is not used\n"; });
            return {};
        }
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        LLVM_DEBUG({ llvm::dbgs() << "elementwiseTernaryOp expect 1 yield\n"; });
        return {};
    }

    // Check the yielded result comes from an op with three operands
    Operation *ternaryOp = yieldOp.getOperand(0).getDefiningOp();
    if (!ternaryOp || ternaryOp->getNumOperands() != 3) {
        LLVM_DEBUG({ llvm::dbgs() << "elementwiseTernaryOp expect 3 operands\n"; });
        return {};
    }

    // Check the three operands are coming directly from the block args or constants
    for (int i = 0; i < 3; ++i) {
        auto operand = ternaryOp->getOperand(i);
        if (auto extsi = operand.getDefiningOp<arith::ExtSIOp>()) {
            operand = extsi.getIn();
        }
        bool ok = isa<BlockArgument>(operand) ||
                  (allowConstants && isa<arith::ConstantOp>(operand.getDefiningOp()));
        if (!ok) {
            LLVM_DEBUG({
                llvm::dbgs() << "elementwiseTernaryOp operand " << i
                             << " is not block arg (or constant if allowed)\n";
            });
            return {};
        }
    }

    return ternaryOp;
}

Operation *getElementwiseBinaryOp(linalg::GenericOp op, bool allowConstants) {

    Value output = op.getResultTensors()[0];
    auto rank = cast<RankedTensorType>(output.getType()).getRank();

    // if rank == 0 (scalar) there is no loop
    if (rank > 0 && op.getNumLoops() < 1) {
        LLVM_DEBUG({ llvm::dbgs() << "elementwise binary op loop number < 1\n"; });
        return {};
    }

    if (op.getNumParallelLoops() != op.getNumLoops()) {
        LLVM_DEBUG({ llvm::dbgs() << "elementwiseBinaryOp expect all loops are parallel\n"; });
        return {};
    }

    // We expect 2 inputs but we must also accept 1 in case the same is used for both operands
    if (op.getNumDpsInputs() != 2 && op.getNumDpsInputs() != 1) {
        LLVM_DEBUG({ llvm::dbgs() << "elementwiseBinaryOp expect 2 inputs (or 1)\n"; });
        return {};
    }

    if (op.getNumDpsInits() != 1) {
        LLVM_DEBUG({ llvm::dbgs() << "elementwiseBinaryOp expect 1 init\n"; });
        return {};
    }

    // Both inputs are used
    for (int i = 0; i < op.getNumDpsInputs(); i++) {
        if (!op.payloadUsesValueFromOperand(op.getDpsInputOperand(i))) {
            LLVM_DEBUG({ llvm::dbgs() << "elementwiseBinaryOp input " << i << " is not used\n"; });
            return {};
        }
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        LLVM_DEBUG({ llvm::dbgs() << "elementwiseBinaryOp expect 1 yield\n"; });
        return {};
    }

    // Check the yielded result comes from an op with two operands
    // Handle case where yield operand is an extsi wrapping the binary op
    Value yieldValue = yieldOp.getOperand(0);
    if (auto extsi = yieldValue.getDefiningOp<arith::ExtSIOp>()) {
        yieldValue = extsi.getIn();
    }

    Operation *bynaryOp = yieldValue.getDefiningOp();
    if (!bynaryOp || bynaryOp->getNumOperands() != 2) {
        LLVM_DEBUG({ llvm::dbgs() << "elementwiseBinaryOp expect 2 operands\n"; });
        return {};
    }
    // Check the two operands are coming directly from the block args with optional sign extension
    auto lhs = bynaryOp->getOperand(0);
    if (auto extsi = lhs.getDefiningOp<arith::ExtSIOp>()) {
        lhs = extsi.getIn();
    }
    auto rhs = bynaryOp->getOperand(1);
    if (auto extsi = rhs.getDefiningOp<arith::ExtSIOp>()) {
        rhs = extsi.getIn();
    }

    bool lhsOK =
        isa<BlockArgument>(lhs) || (allowConstants && isa<arith::ConstantOp>(lhs.getDefiningOp()));
    bool rhsOK =
        isa<BlockArgument>(rhs) || (allowConstants && isa<arith::ConstantOp>(rhs.getDefiningOp()));
    if (!(lhsOK && rhsOK)) {
        LLVM_DEBUG({
            llvm::dbgs() << "lhs is block arg: " << lhsOK << ", rhsOK is block arg: " << rhsOK
                         << ", constant allowed: " << allowConstants << "\n";
        });
        return {};
    }

    return bynaryOp;
}

Operation *getElementwiseUnaryOp(linalg::GenericOp op) {
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
        return {};

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return {};

    Operation *opInBody = yieldOp.getOperand(0).getDefiningOp();
    if (!opInBody || opInBody->getNumOperands() != 1)
        return {};

    auto input = opInBody->getOperand(0);
    if (!(isa<BlockArgument>(input) || isa<arith::ConstantOp>(input.getDefiningOp())))
        return {};

    return opInBody;
}

LogicalResult foldForwardDepthToSpace(
    linalg::TransposeOp transposeOp, PatternRewriter &rewriter,
    const std::optional<IntegerAttr> &maybeFuseGroupAttr
) {
    auto perm_values = transposeOp.getPermutation();

    const llvm::SmallVector<int64_t> d2s_dcr_perm{
        0, 1, 3, 2, 4, 5
    }; // Specific perm for DCR mode
       // Refer https://onnx.ai/onnx/operators/onnx__DepthToSpace.html
    const llvm::SmallVector<int64_t> d2s_crd_perm{
        0, 1, 4, 2, 5, 3
    }; // Specific perm for CRD mode
       // Refer https://onnx.ai/onnx/operators/onnx__DepthToSpace.html

    int d2s_mode = 0;
    if (perm_values == llvm::ArrayRef<int64_t>(d2s_dcr_perm)) {
        d2s_mode = 1;
    }
    if (perm_values == llvm::ArrayRef<int64_t>(d2s_crd_perm)) {
        d2s_mode = 2;
    }
    // Depth2Space necessary condition, either perm matches CRD or DCR perm
    if (!d2s_mode) {
        return rewriter.notifyMatchFailure(transposeOp, "Not DepthToSpace");
    }
    auto p_op = transposeOp.getOperand(0).getDefiningOp();
    if (!p_op) {
        return rewriter.notifyMatchFailure(transposeOp, "Not DepthToSpace");
    }

    auto expandOp = mlir::dyn_cast<tensor::ExpandShapeOp>(p_op); // Reshape Op
    if (!expandOp) {
        return rewriter.notifyMatchFailure(transposeOp, "Not DepthToSpace");
    }

    auto collapseOp =
        mlir::dyn_cast<tensor::CollapseShapeOp>(*transposeOp.getResult().getUsers().begin()
        ); // Pattern for D2S CollapseShape after transpose
    if (!collapseOp) {
        return rewriter.notifyMatchFailure(transposeOp, "Not DepthToSpace");
    }
    auto childShape = mlir::cast<RankedTensorType>(collapseOp.getResult().getType()).getShape();
    auto parentShape = mlir::cast<RankedTensorType>(expandOp.getResult().getType()).getShape();
    auto blockSize = childShape[1] / parentShape[1];
    if (blockSize != 2) {
        return rewriter.notifyMatchFailure(transposeOp, "Depth2space only supports blocksize of 2");
    }

    if (markOpFuseGroup(expandOp, rewriter, maybeFuseGroupAttr) &&
        markOpFuseGroup(collapseOp, rewriter, maybeFuseGroupAttr))
        return success();

    Type elementType =
        mlir::cast<RankedTensorType>(transposeOp.getOperand(0).getType()).getElementType();
    assert(elementType.isInteger() && "Only integer type supported");
    int dtype_size = elementType.getIntOrFloatBitWidth() / 8;
    const auto wram_width = 32 / dtype_size;
    const auto num_inputs = 2;

    auto d2s_input_type = Permutation::nhwc2nchw();
    auto d2s_output_type = d2s_input_type.reverse();

    // Create weights for the d2s interleaving operation
    auto d2s_weights = genD2SWeights(wram_width);
    auto d2s_enum_mode =
        d2s_mode == 1 ? torq_hl::DepthToSpaceModeEnum::DCR : torq_hl::DepthToSpaceModeEnum::CRD;

    auto d2sOp = torq_hl::DepthToSpaceOp::create(
        rewriter, transposeOp.getLoc(),
        transposeType(collapseOp.getResult().getType(), d2s_input_type),
        createInitTensorNCHW(collapseOp, rewriter), blockSize, d2s_enum_mode,
        createI8Const(
            rewriter, transposeOp, d2s_weights, llvm::ArrayRef<int64_t>{wram_width * num_inputs}
        ),
        transposeValue(expandOp.getOperand(0), d2s_input_type, transposeOp.getLoc(), rewriter)
    );
    auto targetOp =
        transposeValue(d2sOp.getOutput(), d2s_output_type, transposeOp.getLoc(), rewriter);
    collapseOp.replaceAllUsesWith(targetOp);
    rewriter.eraseOp(expandOp);
    rewriter.eraseOp(collapseOp);
    rewriter.replaceOp(transposeOp, targetOp);
    return success();
}

bool isRoundingRightShiftOp(linalg::GenericOp op, arith::ShRSIOp &shrsiOp1) {
    if (op.getNumDpsInputs() > 2 || op.getNumDpsInits() != 1) {
        return false;
    }

    // Get the yield op and its defining op
    auto *body = op.getBody();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        return false;
    }

    Value addVal = yieldOp.getOperand(0);
    auto addOp = addVal.getDefiningOp<arith::AddIOp>();
    if (!addOp) {
        return false;
    }

    shrsiOp1 = addOp.getLhs().getDefiningOp<arith::ShRSIOp>();
    if (!shrsiOp1) {
        return false;
    }

    auto extuiOp = addOp.getRhs().getDefiningOp<arith::ExtUIOp>();
    if (!extuiOp) {
        return false;
    }

    auto trunciOp = extuiOp.getIn().getDefiningOp<arith::TruncIOp>();
    if (!trunciOp) {
        // try to find if op is arith.andi
        auto andiOp = extuiOp.getIn().getDefiningOp<arith::AndIOp>();
        if (andiOp) {
            // if andiOp is found, get trunciOp from its rhs
            trunciOp = andiOp.getRhs().getDefiningOp<arith::TruncIOp>();
            if (!trunciOp) {
                return false;
            }
        }
        else {
            return false;
        }
    }
    // check if trunciOp is actually a truncation to i1
    if (!trunciOp.getType().isInteger(1)) {
        return false;
    }

    auto shrsiOp2 = trunciOp.getIn().getDefiningOp<arith::ShRSIOp>();
    if (!shrsiOp2) {
        return false;
    }

    // check if shrisiOp1 and shrisiOp2 input is argument of the op
    auto shrsiOp1Input = shrsiOp1.getOperand(0);
    auto shrsiOp2input = shrsiOp2.getOperand(0);
    if (!isa<BlockArgument>(shrsiOp1Input) || !isa<BlockArgument>(shrsiOp2input)) {
        return false;
    }
    return true;
}

// DEADCODE:
bool isCollapseOrExpandShapeGeneric(Operation *op) {
    if (!isa<linalg::GenericOp>(op))
        return false;

    auto genericOp = dyn_cast<linalg::GenericOp>(op);

    if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
        return false;

    auto inType = cast<RankedTensorType>(genericOp.getDpsInputs()[0].getType());
    auto outType = cast<RankedTensorType>(genericOp.getDpsInits()[0].getType());
    if (!inType || !outType)
        return false;

    if (inType.getNumElements() != outType.getNumElements())
        return false;

    if (inType.getElementType() != outType.getElementType())
        return false;

    Region &region = genericOp.getRegion();

    if (!region.hasOneBlock())
        return false;

    Block &block = region.front();

    return block.getOperations().size() == 1 && mlir::isa<mlir::linalg::YieldOp>(block.front());
}

StringRef getCastOpName(Value input, Value output) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    auto inputElementType = inputType.getElementType();
    auto outputElementType = outputType.getElementType();

    if ((inputElementType.isF32() || inputElementType.isBF16()) && outputElementType.isInteger()) {
        return "f2i";
    }
    else if ((inputElementType.isF32() || inputElementType.isBF16()) &&
             (outputElementType.isF32() || outputElementType.isBF16())) {
        return "f2f";
    }
    else if (inputElementType.isInteger() && outputElementType.isInteger()) {
        return "i2i";
    }
    else if (inputElementType.isInteger() &&
             (outputElementType.isF32() || outputElementType.isBF16())) {
        return "i2f";
    }

    return "";
}

bool getIntegerConstantValue(arith::ConstantOp constOp, int32_t *value) {
    if (!constOp || !value) {
        return false;
    }

    auto attr = constOp.getValue();
    if (!attr) {
        LLVM_DEBUG({ llvm::errs() << "constantOp has no attribute \n"; });
        return false;
    }

    int32_t data = 0;

    if (auto intAttr = dyn_cast_if_present<IntegerAttr>(attr)) {
        data = intAttr.getInt();
    }
    else if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(attr)) {
        if (denseAttr.isSplat() || denseAttr.getNumElements() == 1) {
            data = (*denseAttr.begin()).getSExtValue();
        }
        else {
            LLVM_DEBUG({ llvm::errs() << "constant is not scalar \n"; });
            return false;
        }
    }
    else {
        LLVM_DEBUG({ llvm::errs() << "unsupported constant type \n"; });
        return false;
    }

    *value = data;

    return true;
}

// rescale op cannot work for floating point data type
// if create something from rescale scalar which means only works for integer type
Value create1DimTensorFromRescaleScalar(
    linalg::GenericOp srcOp, tosa::ApplyScaleOp applyScaleOp, ScaleInfo &scaleInfo,
    const Type &elementType, PatternRewriter &rewriter
) {
    Value input = applyScaleOp.getValue();
    if (!input) {
        LLVM_DEBUG({ llvm::errs() << "applyScaleOp cannot get value\n"; });
        return nullptr;
    }

    auto ms = getMultiplierAndShift(srcOp, applyScaleOp, 1);
    if (!ms) {
        return nullptr;
    }

    double scaleFactor = static_cast<double>(ms.multiplier[0]) / (1l << ms.shift[0]);

    if (scaleInfo.scale > 0) {
        scaleFactor = scaleFactor * scaleInfo.scale;
    }

    int32_t data = 0;
    auto constOp = dyn_cast_or_null<arith::ConstantOp>(input.getDefiningOp());
    if (constOp) {
        if (!getIntegerConstantValue(constOp, &data)) {
            LLVM_DEBUG({ llvm::errs() << "cannot get integer constant value\n"; });
            return input;
        }
        data = static_cast<int32_t>(std::round(data * scaleFactor));
    }
    else {
        // Handle the case where a constant tensor is passed as a DPS input to the
        // linalg.generic (dpsInputCount == 1). Inside the body the constant appears
        // as a block argument, not as an arith.constant, so we trace backwards:
        //
        //   linalg.generic ins(%cst : tensor<i16>) outs(...) {
        //   ^bb0(%in: i16, %out: i32):
        //     %ext  = arith.extsi %in : i16 to i32
        //     %sub  = arith.subi %ext, %zp : i32          // optional zero-point
        //     %res  = tosa.apply_scale %sub, %mult, %shift
        //     linalg.yield %res
        //   }
        //
        // We walk: apply_scale input -> subi (extract zp) -> extsi -> block_arg
        //          -> DPS input operand -> arith.constant
        // Then fold: result = round((constantValue - zeroPoint) * scaleFactor)
        int32_t inputZp = 0;
        Value traceVal = input;

        // Check for subi (zero-point subtraction)
        if (auto subOp = traceVal.getDefiningOp<arith::SubIOp>()) {
            auto maybeZp = getConstIntValue(subOp.getRhs());
            if (maybeZp) {
                inputZp = *maybeZp;
            }
            traceVal = subOp.getLhs();
        }

        // Check for extsi
        if (auto extOp = traceVal.getDefiningOp<arith::ExtSIOp>()) {
            traceVal = extOp.getIn();
        }

        // Should now be a block argument
        auto blockArg = dyn_cast<BlockArgument>(traceVal);
        if (!blockArg) {
            return nullptr;
        }

        // Resolve block arg to DPS input of the linalg.generic
        unsigned argIdx = blockArg.getArgNumber();
        if (argIdx >= srcOp.getNumDpsInputs()) {
            return nullptr;
        }

        Value dpsInput = srcOp.getInputs()[argIdx];
        auto dpsConstOp = dpsInput.getDefiningOp<arith::ConstantOp>();
        if (!dpsConstOp) {
            return nullptr;
        }

        int32_t inputData = 0;
        if (!getIntegerConstantValue(dpsConstOp, &inputData)) {
            LLVM_DEBUG({ llvm::errs() << "cannot get DPS input constant value\n"; });
            return nullptr;
        }

        // Compute the rescale: (inputData - inputZp) * scaleFactor
        data = static_cast<int32_t>(std::round((inputData - inputZp) * scaleFactor));
        constOp = dpsConstOp;
    }

    auto outputType = cast<RankedTensorType>(srcOp.getResults()[0].getType());
    RankedTensorType constType = RankedTensorType::get(outputType.getShape(), elementType);
    DenseElementsAttr value;

    if (elementType.isInteger(16)) {
        value = DenseIntElementsAttr::get(constType, static_cast<int16_t>(data));
    }
    else if (elementType.isInteger(32)) {
        value = DenseIntElementsAttr::get(constType, data);
    }
    else if (elementType.isInteger(8)) {
        value = DenseIntElementsAttr::get(constType, static_cast<int8_t>(data));
    }
    else {
        LLVM_DEBUG({ llvm::errs() << "only support 8/16/32 bit integer\n"; });
        return input;
    }
    auto output = arith::ConstantOp::create(rewriter, constOp.getLoc(), constType, value);
    return output.getResult();
}

bool foldScalarRescale(
    Value &input, ScaleInfo &scaleInfo, const Type &elementType, PatternRewriter &rewriter
) {

    linalg::GenericOp rescaleOp = input.getDefiningOp<linalg::GenericOp>();

    if (!rescaleOp) {
        LLVM_DEBUG({ llvm::errs() << "Value input definingOp is not linalg.generic op\n"; });
        return false;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(rescaleOp.getBody()->getTerminator());
    if (!yieldOp) {
        LLVM_DEBUG({ llvm::errs() << "There is no yield in linalg.generic body\n"; });
        return false;
    }

    auto yieldValues = yieldOp.getValues();
    if (yieldValues.size() != 1) {
        LLVM_DEBUG({ llvm::errs() << "Linalg.yield operand is not 1 \n"; });
        return false;
    }

    tosa::ApplyScaleOp applyScaleOp;
    applyScaleOp = yieldValues[0].getDefiningOp<tosa::ApplyScaleOp>();
    if (!applyScaleOp) {
        LLVM_DEBUG({ llvm::errs() << "apply scale op does not exist\n"; });
        return false;
    }
    auto output = create1DimTensorFromRescaleScalar(
        rescaleOp, applyScaleOp, scaleInfo, elementType, rewriter
    );

    if (output) {
        input = output;
        return true;
    }
    return false;
}

// Conv2DMatmulOpConversion weight conversion function
Value convertWeights(mlir::Value weights, PatternRewriter &rewriter) {
    // Reorder weights to OIHW
    auto weightTy = dyn_cast<RankedTensorType>(weights.getType());
    auto weightElemType = weightTy.getElementType();
    auto weightShape = weightTy.getShape();

    // Validate expected shape: [OC, IC] for matmul-style
    assert(weightShape.size() == 2);

    // Assume shape was originally [OC, IC] from matmul-style
    int on = weightShape[0]; // OC
    int in = weightShape[1]; // IC
    int hn = 1;
    int wn = 1;
    std::vector<int64_t> weight_shape{on, in, hn, wn};

    auto srcOp = weights.getDefiningOp();

    if (weightElemType.isBF16()) {
        auto bfVec = attrValuesAsVec<APFloat>(weights);
        std::vector<APFloat> reordered = get_weights_OIHW<APFloat>(bfVec, on, hn, wn, in);
        return createFConst(rewriter, *srcOp, reordered, weight_shape);
    }
    else if (weightElemType.isInteger(8)) {
        auto reordered = attrValuesAsVec<int8_t>(weights);
        reordered = get_weights_OIHW<int8_t>(reordered, on, hn, wn, in);
        return createI8Const(rewriter, *srcOp, reordered, weight_shape);
    }
    else {
        assert(false && "Unsupported weight type");
    }
}

Value makeBitcast(
    linalg::GenericOp srcOp, PatternRewriter &rewriter, RankedTensorType resultType, Value input
) {
    return torq_hl::IdentityOp::create(
               rewriter, srcOp.getLoc(), resultType, createInitTensor(srcOp, rewriter, resultType),
               input
    )
        .getOutput();
}

Value makeRescale16(
    linalg::GenericOp srcOp, PatternRewriter &rewriter, Value input, int32_t scaleFactor,
    int shiftFactor, int32_t inputZp, int32_t outputZp
) {

    auto outputType = dyn_cast<RankedTensorType>(input.getType());
    int8_t weight_data = 1;
    int32_t bias_data = -inputZp;
    std::vector<int8_t> weights = {weight_data};
    const std::vector<int32_t> bias = {bias_data};
    const std::vector<int32_t> scale = {scaleFactor};

    // make the rescale
    return torq_hl::FMAOp::create(
               rewriter, srcOp.getLoc(), outputType, createInitTensor(srcOp, rewriter, outputType),
               outputZp, std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max(),
               shiftFactor, createI8Const(rewriter, srcOp, weights, llvm::ArrayRef<int64_t>{1}),
               createI32Const(rewriter, srcOp, interleave(bias, scale)), input
    )
        .getResult(0);
}

Value makeI16LUTFromVals(
    linalg::GenericOp srcOp, PatternRewriter &rewriter, Value input, SmallVector<int32_t> values
) {
    const std::vector<APInt> bias = {APInt(32, 0, /*isSigned=*/true)};
    const std::vector<APInt> scale = {APInt(32, 1, /*isSigned=*/true)};
    return syna::torq_hl::TableOp::create(
               rewriter, srcOp.getLoc(), dyn_cast<RankedTensorType>(input.getType()),
               createInitTensor(srcOp, rewriter, dyn_cast<RankedTensorType>(input.getType())),
               createIConst(rewriter, srcOp, interleave(bias, scale)), input,
               DenseI32ArrayAttr::get(rewriter.getContext(), values)
    )
        .getResult(0);
}

// 'a', 'b', 'c' correspond to those in scripts/bf16luts.py comments/prints
Value makeScaledLut(
    linalg::GenericOp srcOp, PatternRewriter &rewriter, Value input, int32_t aScaleFactor,
    int aShiftFactor, int32_t aInputZp, int32_t aOutputZp, SmallVector<int32_t> bValues,
    int32_t cScaleFactor, int cShiftFactor, int32_t cInputZp, int32_t cOutputZp
) {
    auto scaledInput =
        makeRescale16(srcOp, rewriter, input, cScaleFactor, cShiftFactor, cInputZp, cOutputZp);
    auto lookuped = makeI16LUTFromVals(srcOp, rewriter, scaledInput, bValues);
    return makeRescale16(
        srcOp, rewriter, lookuped, aScaleFactor, aShiftFactor, aInputZp, aOutputZp
    );
}

Value makeSelect(
    linalg::GenericOp srcOp, PatternRewriter &rewriter, Value pred, Value ifTrue, Value ifFalse
) {
    auto resultType = dyn_cast<RankedTensorType>(ifTrue.getType());
    return torq_hl::SelectOp::create(
               rewriter, srcOp.getLoc(), resultType, createInitTensor(srcOp, rewriter, resultType),
               pred, ifTrue, ifFalse
    )
        .getOutput();
}

Value makeElementWiseBinary(
    linalg::GenericOp srcOp, PatternRewriter &rewriter, Value input0, Value input1,
    torq_hl::ElementwiseOpEnum opType
) {
    auto resultType = dyn_cast<RankedTensorType>(input0.getType());
    if (opType == torq_hl::ElementwiseOpEnum::GREATER)
        resultType = RankedTensorType::get(resultType.getShape(), rewriter.getI1Type());

    return torq_hl::ElementWiseBinaryOp::create(
               rewriter, srcOp.getLoc(), resultType, createInitTensor(srcOp, rewriter, resultType),
               opType, input0, input1, /*isUnsigned=*/false
    )
        .getOutput();
}

FailureOr<Value> pickGroupResultInt8(Value value) {
    // Follow single-use chain forward until we hit an int rescale/clamp boundary,
    // which defines the terminal value for fusion planning.
    while (true) {
        if (!value.hasOneUse()) {
            // Ambiguous fanout: stop to avoid planning across multiple consumers.
            return failure();
        }
        auto userOp = value.getUsers().begin();
        if (auto genericOp = dyn_cast<linalg::GenericOp>(*userOp)) {
            value = genericOp.getResult(0);
            for (auto &op : genericOp.getRegion().getOps()) {
                if (isa<linalg::YieldOp>(op)) {
                    continue;
                }
                if (isa<tosa::ApplyScaleOp>(op) || isa<arith::TruncIOp>(op) ||
                    isa<arith::MaxSIOp>(op) || isa<arith::MinSIOp>(op)) {
                    // Found quantized rescale tail (apply_scale + clamp/trunc).
                    return value;
                }
            }
        }
        else if (isa<tensor::ExpandShapeOp>(*userOp)) {
            // Shape-only op: keep walking through the transformed value.
            value = (*userOp)->getResult(0);
        }
        else {
            // Non-target user kind: current value is the best terminal point.
            return value;
        }
    }
    return value;
}

FailureOr<Value> pickGroupResultFloat(Value value) {
    // Float path mirrors int path but uses float rescale/clamp markers.
    while (true) {
        if (!value.hasOneUse()) {
            return failure();
        }
        auto userOp = value.getUsers().begin();
        if (auto genericOp = dyn_cast<linalg::GenericOp>(*userOp)) {
            value = genericOp.getResult(0);
            for (auto &op : genericOp.getRegion().getOps()) {
                if (isa<linalg::YieldOp>(op)) {
                    continue;
                }
                if (isa<arith::TruncFOp>(op) || isa<arith::MaximumFOp>(op) ||
                    isa<arith::MinimumFOp>(op)) {
                    // Found float rescale tail (clamp/trunc boundary).
                    return value;
                }
            }
        }
        else if (isa<tensor::ExpandShapeOp>(*userOp)) {
            value = (*userOp)->getResult(0);
        }
        else {
            return value;
        }
    }
    return value;
}

FailureOr<Value> pickGroupResult(Value value) {
    // Dispatch terminal-value selection by element type family.
    auto valueType = dyn_cast<ShapedType>(value.getType()).getElementType();
    if (valueType.isInteger()) {
        return pickGroupResultInt8(value);
    }
    if (valueType.isBF16() || valueType.isF32()) {
        return pickGroupResultFloat(value);
    }
    return failure();
}

bool isSingleTensorReductionOp(linalg::LinalgOp linalgOp) {
    // Keep only simple reductions that are safe to include in fusion clone:
    // one input, one init/output, projected-permutation indexing.
    if (linalgOp.getNumReductionLoops() == 0) {
        return false;
    }
    if (linalgOp.getNumDpsInits() != 1) {
        return false;
    }
    if (linalgOp.getNumDpsInputs() != 1) {
        return false;
    }
    if (!linalgOp.hasOnlyProjectedPermutations()) {
        return false;
    }

    return true;
}

bool shouldInclude(Operation *op, Value value) {
    // Allow-list of ops that are considered fusible/supporting for bias/scale
    // extraction. Anything else becomes a traversal boundary.
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        if (linalg::isElementwise(linalgOp)) {
            return true;
        }

        if (isSingleTensorReductionOp(linalgOp)) {
            return true;
        }

        if (isa<linalg::TransposeOp, linalg::FillOp>(op)) {
            return true;
        }

        return false;
    }

    if (isa<tensor::ExtractSliceOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp,
            tensor::EmptyOp>(op)) {
        return true;
    }

    if (isa<arith::ConstantOp>(op)) {
        return true;
    }

    if (isa<affine::AffineApplyOp>(op)) {
        return true;
    }

    return false;
}

// Support functions for fusion plan
LogicalResult computeGroup(Value value, Value groupResult, SmallVector<Operation *> &opsToGroup);

FailureOr<FusionPlan> buildFusionPlanAndRebindOutput(Value &value) {
    auto plan = createFusionPlan(value);
    if (failed(plan)) {
        return failure();
    }
    // Update the output to be the selected group result for downstream use.
    value = plan->getFusedOutput();
    return *plan;
}

FailureOr<FusionPlan> createFusionPlan(Value value) {
    // Build a backward fusion plan rooted at `value`.
    // The plan captures operations between `value` (anchor) and a selected terminal
    // result (`groupResult`) that are eligible for cloning/folding later.
    FusionPlan fusionPlan;
    auto valueType = dyn_cast<ShapedType>(value.getType());
    if (!valueType) {
        LLVM_DEBUG({ llvm::errs() << "matching error value is not a ShapedType!\n"; });
        return failure();
    }

    // Choose the final op/value of the fusible chain (typically at the end of
    // elementwise/rescale/clamp sequence).
    auto groupResult = pickGroupResult(value);
    if (failed(groupResult)) {
        LLVM_DEBUG({ llvm::dbgs() << "pickGroupResult FAILED\n"; });
        return failure();
    }

    // Walk backward from groupResult toward anchor and collect only supported ops.
    SmallVector<Operation *> opsToGroup;
    if (failed(computeGroup(value, *groupResult, opsToGroup))) {
        LLVM_DEBUG({ llvm::dbgs() << "computeGroup FAILED\n"; });
        return failure();
    }

    // Persist the computed plan for downstream bias/scale extraction utilities.
    fusionPlan.anchor = value.getDefiningOp();
    fusionPlan.neededOps.insert(fusionPlan.neededOps.end(), opsToGroup.begin(), opsToGroup.end());
    for (auto op : fusionPlan.neededOps) {
        LLVM_DEBUG({
            llvm::dbgs() << "createFusionPlan neededOp: ";
            op->print(llvm::dbgs());
            llvm::dbgs() << "\n";
        });
    }

    return fusionPlan;
}

LogicalResult computeGroup(Value anchor, Value groupResult, SmallVector<Operation *> &opsToGroup) {
    // Backward traversal from the chosen terminal result to collect the minimal
    // connected subgraph needed to compute bias/scale relative to `anchor`.
    llvm::SmallVector<Value, 8> worklist;
    worklist.push_back(groupResult);
    llvm::SmallPtrSet<Operation *, 8> visited;

    while (!worklist.empty()) {
        Value v = worklist.pop_back_val();
        Operation *defOp = v.getDefiningOp();
        if (!defOp || visited.contains(defOp)) {
            continue;
        }
        if (v.getParentRegion() != anchor.getParentRegion()) {
            // Do not cross region boundaries (e.g. nested control-flow regions).
            continue;
        }
        visited.insert(defOp);

        // Stop expansion through non-allowlisted ops (except anchor itself).
        if (v != anchor && !shouldInclude(defOp, v)) {
            continue;
        }
        if (v == anchor) {
            // Anchor is handled as external root; no need to include it in neededOps.
            continue;
        }
        // Insert at front so producer ordering is closer to topological order.
        opsToGroup.insert(opsToGroup.begin(), defOp);

        for (Value input : defOp->getOperands()) {
            worklist.push_back(input);
        }
    }
    // opsToGroup.insert(opsToGroup.begin(), anchor.getDefiningOp());
    LLVM_DEBUG({
        llvm::dbgs() << "computeGroup opsToGroup:\n";
        for (auto op : opsToGroup) {
            op->print(llvm::dbgs());
            llvm::dbgs() << "\n";
        }
    });

    // Sort the ops by their order in the block to ensure correct cloning order later
    LLVM_DEBUG({
        llvm::dbgs() << "computeGroup before sort opsToGroup:\n";
        for (auto op : opsToGroup) {
            op->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
            llvm::dbgs() << "\n";
        }
    });
    std::sort(opsToGroup.begin(), opsToGroup.end(), [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
    });
    LLVM_DEBUG({
        llvm::dbgs() << "computeGroup after sort opsToGroup:\n";
        for (auto op : opsToGroup) {
            op->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
            llvm::dbgs() << "\n";
        }
    });
    return success();
}

bool isRescaleF32(Operation *op) {
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
        for (auto &op : genericOp.getRegion().getOps()) {
            if (isa<arith::TruncFOp>(op) || isa<arith::MaximumFOp>(op) ||
                isa<arith::MinimumFOp>(op)) {
                return true;
            }
        }
    }
    return false;
}

bool isRescaleInt(Operation *op) {
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
        for (auto &op : genericOp.getRegion().getOps()) {
            if (isa<tosa::ApplyScaleOp>(op) || isa<arith::TruncIOp>(op) ||
                isa<arith::MaxSIOp>(op) || isa<arith::MinSIOp>(op)) {
                return true;
            }
        }
    }
    return false;
}

bool isRescale(Operation *op) {
    auto valueType = dyn_cast<ShapedType>(op->getResult(0).getType()).getElementType();
    if (valueType.isInteger()) {
        return isRescaleInt(op);
    }
    if (valueType.isBF16() || valueType.isF32()) {
        return isRescaleF32(op);
    }
    return false;
}

FailureOr<Value> getWeightZp(Value bias, OpBuilder &builder) {
    auto biasOp = mlir::dyn_cast<linalg::GenericOp>(bias.getDefiningOp());
    if (!biasOp) {
        return failure();
    }
    if (biasOp.getNumOperands() < 4) {
        return failure();
    }

    // Get the first Op from start of biasOp block
    auto &region = biasOp.getRegion();
    auto &block = region.front();
    auto firstOp = block.getOperations().begin();

    auto weightZpOp = firstOp->getOperand(1).getDefiningOp();
    if (auto constOp = dyn_cast<arith::ConstantOp>(weightZpOp)) {
        auto constType = dyn_cast<RankedTensorType>(constOp.getType());
        if (!constType) {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfter(constOp);
            auto ty = dyn_cast<IntegerType>(constOp.getType());
            SmallVector<int64_t> newShape{1};
            auto newTy = RankedTensorType::get(newShape, builder.getIntegerType(ty.getWidth()));
            auto emTensor =
                tensor::EmptyOp::create(
                    builder, biasOp.getLoc(), ArrayRef<int64_t>{newShape}, newTy.getElementType()
                )
                    .getResult();
            auto zeroOp =
                linalg::FillOp::create(builder, bias.getLoc(), constOp.getResult(), emTensor);
            return zeroOp.getResult(0);
        }
    }
    return weightZpOp->getResult(0);
}

Value postProcessBias(Value bias, OpBuilder &builder) {
    // This rewrite targets the specific fused-bias form that carries extra operands.
    // Leave simpler generics unchanged.
    auto biasOp = mlir::dyn_cast<linalg::GenericOp>(bias.getDefiningOp());
    if (!biasOp) {
        // Nothing to normalize when bias is not produced by linalg.generic.
        return bias;
    }
    if (biasOp->getNumOperands() < 4) {
        // Extra operands are the weight zero point tensors
        return bias;
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(biasOp);
    assert(biasOp->getNumOperands() < 5 && "Bias op with greater than 4 operands is not expected");
    // Operand#1 is the weight zero point, we explicitly neutralize to zero
    auto operand = biasOp.getOperand(1);

    auto opTy = mlir::dyn_cast<ShapedType>(operand.getType());

    auto zeroAttr = DenseElementsAttr::get(opTy, builder.getZeroAttr(opTy.getElementType()));
    auto zeroConst = arith::ConstantOp::create(builder, biasOp->getLoc(), opTy, zeroAttr);

    biasOp->setOperand(1, zeroConst);

    return biasOp->getResult(0);
}

FailureOr<Value> createNewBias(
    FusionPlan &fusionPlan, OpBuilder &builder, llvm::SmallVectorImpl<Operation *> &opsToDelete,
    std::optional<Value> &optionalWeightZp
) {
    // Builds a per-channel bias tensor from a fusion plan by cloning the relevant
    // subgraph, optionally extracting weight zero-point information, normalizing
    // shape (including inverse collapse when needed), and reducing non-channel
    // dimensions into a 1D bias vector with appropriate accumulator type handling.
    int endIdx = fusionPlan.neededOps.size() - 1;
    if (!isRescale(fusionPlan.neededOps.back())) {
        // If the tail op is not a rescale/clamp stage, include the full fusion range.
        // This usually happens in BF16 cases where there is no rescale after anchor op
        endIdx = fusionPlan.neededOps.size();
    }
    auto bias =
        createClonedBlock(builder, fusionPlan, fusionPlan.neededOps, 0, endIdx, opsToDelete);

    if (!bias) {
        // Fallback: no cloneable bias-producing chain was found.
        // Materialize a zero 1D bias over the channel dimension.
        return failure();
    }
    auto maybeWeightZp = getWeightZp(bias, builder);
    if (succeeded(maybeWeightZp)) {
        // Thread optional weight zero-point back to caller for downstream quant handling.
        optionalWeightZp = *maybeWeightZp;
    }
    // Canonicalize bias clone shape/operands before reduction. Currently weight zero-point is
    // neutralized
    bias = postProcessBias(bias, builder);

    LLVM_DEBUG({
        llvm::dbgs() << "computeBias: Ops to Delete:\n";
        for (auto op : opsToDelete) {
            op->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
            llvm::dbgs() << "\n";
        }
    });
    fusionPlan.opsToFuse.insert(fusionPlan.opsToFuse.end(), opsToDelete.begin(), opsToDelete.end());
    return bias;
}

static void reduceBodyYield(OpBuilder &b, Location loc, ValueRange args) {
    linalg::YieldOp::create(b, loc, ArrayRef<Value>{args[0]});
}

static void reduceBodyExtYield(OpBuilder &b, Location loc, ValueRange args) {
    Value y = arith::ExtSIOp::create(b, loc, b.getI32Type(), args[0]);
    linalg::YieldOp::create(b, loc, ArrayRef<Value>{y});
}

static void reduceBodyTruncYield(OpBuilder &b, Location loc, ValueRange args) {
    // Warning: the assumption is that even if the bias type is 64bits, the actual value fits 32bits
    Value y = arith::TruncIOp::create(b, loc, b.getI32Type(), args[0]);
    linalg::YieldOp::create(b, loc, ArrayRef<Value>{y});
}

static void reduceBodyExtFYield(OpBuilder &b, Location loc, ValueRange args) {
    Value y = arith::ExtFOp::create(b, loc, b.getF32Type(), args[0]);
    linalg::YieldOp::create(b, loc, ArrayRef<Value>{y});
}

FailureOr<Value> computeBias(
    FusionPlan &fusionPlan, int channelDim, std::optional<Value> &optionalWeightZp, int biasChDim
) {
    auto firstOp = fusionPlan.anchor;
    auto firstOpResult = firstOp->getResult(0);
    auto firstOpResultType = dyn_cast<ShapedType>(firstOpResult.getType());
    auto loc = firstOp->getLoc();
    biasChDim = biasChDim < 0 ? channelDim : biasChDim;
    SmallVector<int64_t> biasShape{firstOpResultType.getShape()[biasChDim]};

    if (!firstOpResultType) {
        LLVM_DEBUG({ llvm::dbgs() << "computeBias: first op result is not ShapedType\n"; });
        return failure();
    }

    auto parentRegion = fusionPlan.anchor->getParentRegion();
    auto owner = parentRegion->getParentOp();
    OpBuilder builder(owner->getContext());
    SmallVector<Operation *, 8> opsToDelete;

    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(firstOp);
    auto maybeBias = createNewBias(fusionPlan, builder, opsToDelete, optionalWeightZp);
    if (failed(maybeBias)) {
        auto biasType = RankedTensorType::get(biasShape, firstOpResultType.getElementType());
        return arith::ConstantOp::create(builder, loc, builder.getZeroAttr(biasType)).getResult();
    }

    Value bias = *maybeBias;
    auto biasTy = dyn_cast<ShapedType>(bias.getType());
    if (biasTy.getRank() == 2) {
        channelDim = biasChDim;
    }
    // Reduce across all non-channel dims to produce per-channel bias.
    SmallVector<int64_t, 4> reduceD;
    reduceD.reserve(biasTy.getRank() - 1);
    for (int i = 0; i < biasTy.getRank(); ++i) {
        if (i != channelDim)
            reduceD.push_back(i);
    }

    Type biasElTy = biasTy.getElementType();
    auto reduceBody = biasElTy.isFloat()       ? reduceBodyExtFYield
                      : biasElTy.isInteger(32) ? reduceBodyYield
                      : biasElTy.isInteger(64) ? reduceBodyTruncYield
                                               : reduceBodyExtYield;

    Type torqBiasTy = biasElTy.isFloat() ? (Type)builder.getF32Type() : (Type)builder.getI32Type();
    Value empty = tensor::EmptyOp::create(builder, loc, biasShape, torqBiasTy).getResult();
    bias = linalg::ReduceOp::create(builder, loc, bias, empty, reduceD, reduceBody).getResult(0);
    setCompileTimeConstAttr(bias.getDefiningOp());

    return bias;
}

mlir::FailureOr<Value>
modifyMulValue(Value mul, Value biasScale, Operation *lastOp, OpBuilder &builder) {
    auto bTy = mlir::dyn_cast<ShapedType>(biasScale.getType());
    auto mulTy = mul.getType();

    // Handle ShapedType mul value
    if (auto ty = mlir::dyn_cast<ShapedType>(mulTy)) {
        // If shape already matches, return as-is
        if (ty.getShape().back() == bTy.getShape().back()) {
            return mul;
        }
        // Otherwise, reshape it to match the bias scale shape
        auto newShape = bTy.getShape().back();
        auto init = tensor::EmptyOp::create(
                        builder, lastOp->getLoc(), ArrayRef<int64_t>{newShape}, ty.getElementType()
        )
                        .getResult();
        return linalg::BroadcastOp::create(
                   builder, lastOp->getLoc(), mul, init, SmallVector<int64_t>{0}
        )
            .getResult()[0];
    }

    // Handle IntegerType mul value
    if (auto ty = mlir::dyn_cast<IntegerType>(mulTy)) {
        auto newShape = bTy.getShape().back();
        auto newTy = RankedTensorType::get(newShape, builder.getIntegerType(ty.getWidth()));
        auto emTensor =
            tensor::EmptyOp::create(
                builder, lastOp->getLoc(), ArrayRef<int64_t>{newShape}, newTy.getElementType()
            )
                .getResult();
        return linalg::FillOp::create(builder, lastOp->getLoc(), mul, emTensor).getResult(0);
    }

    // If neither type, return failure
    return failure();
}

mlir::FailureOr<Value>
modifyShiftValue(Value shift, Operation *lastOp, ScaleClampInfo &scInfo, OpBuilder &builder) {
    Value minShiftV;
    Value origShiftV = shift;
    while (auto tShiftV = origShiftV.getDefiningOp<tensor::ExtractSliceOp>()) {
        origShiftV = tShiftV.getSource();
    }
    auto shiftTy = origShiftV.getType();

    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointAfter(origShiftV.getDefiningOp());
    if (auto shapedTy = mlir::dyn_cast<ShapedType>(shiftTy)) {
        auto emTensor = createI8Const(
            builder, *lastOp, ArrayRef<int8_t>{std::numeric_limits<int8_t>::max()}, {}
        );
        auto rOp = linalg::ReduceOp::create(
            builder, lastOp->getLoc(), origShiftV, emTensor.getResult(), ArrayRef<int64_t>{0},
            [](OpBuilder &b, Location loc, ValueRange args) {
                auto min = arith::MinSIOp::create(b, loc, args[0], args[1]);
                linalg::YieldOp::create(b, loc, ArrayRef<Value>{min});
            }
        );

        minShiftV =
            tensor::ExtractOp::create(builder, lastOp->getLoc(), rOp.getResult(0), ValueRange{})
                .getResult();
    }
    else if (auto intTy = mlir::dyn_cast<IntegerType>(shiftTy)) {
        minShiftV = shift;
    }
    else {
        // Unsupported shift type
        LLVM_DEBUG({ llvm::dbgs() << "modifyShiftValue: unsupported shift type\n"; });
        return failure();
    }

    auto modShiftV = tensor::EmptyOp::create(
                         builder, lastOp->getLoc(), ArrayRef<int64_t>{1}, builder.getIntegerType(8)
    )
                         .getResult();
    auto fillOp = linalg::FillOp::create(builder, lastOp->getLoc(), minShiftV, modShiftV);
    // ComputeArithConst works only on tensor so getting the shift value from ReduceOp
    // result
    auto maybeShiftFactor = computeArithConst(fillOp.getResult(0), true, {});
    if (failed(maybeShiftFactor)) {
        LLVM_DEBUG({ llvm::dbgs() << "computeRescaleInfo: failed to compute shift factor\n"; });
        return failure();
    }
    // FIXME: ScaleShift is currently calculated. May need to change this to constant later
    // Also this is a side effect, probably better to change it later
    scInfo.scaleShift = returnDenseElementAttr(*maybeShiftFactor).getValues<int8_t>()[0];

    return fillOp.getResult(0);
}

/// Ops collected from the body of an int8 quantized rescale linalg.generic.
struct Int8RescaleBodyOps {
    tosa::ApplyScaleOp applyScaleOp;
    arith::AddIOp addOp;
    arith::MaxSIOp maxOp;
    arith::MinSIOp minOp;
    arith::TruncIOp truncOp;
};

/// Scan the linalg.generic body for the canonical int8 rescale ops and
/// populate ScaleClampInfo with clamp/zero-point constants.
/// Returns true iff an apply_scale op was found.
static bool extractInt8RescaleBodyOps(
    linalg::GenericOp rescaleOp, Int8RescaleBodyOps &out, ScaleClampInfo &scInfo
) {
    for (auto &op : rescaleOp.getRegion().getOps()) {
        if (auto o = dyn_cast<tosa::ApplyScaleOp>(op))
            out.applyScaleOp = o;
        else if (auto o = dyn_cast<arith::AddIOp>(op))
            out.addOp = o;
        else if (auto o = dyn_cast<arith::MaxSIOp>(op))
            out.maxOp = o;
        else if (auto o = dyn_cast<arith::MinSIOp>(op))
            out.minOp = o;
        else if (auto o = dyn_cast<arith::TruncIOp>(op))
            out.truncOp = o;
    }
    // minsi rhs is the upper clamp bound (confusingly stored in scInfo.max).
    if (out.minOp)
        if (auto c = dyn_cast<arith::ConstantOp>(out.minOp.getRhs().getDefiningOp()))
            scInfo.max = mlir::cast<IntegerAttr>(c.getValue()).getInt();
    // maxsi rhs is the lower clamp bound (stored in scInfo.min).
    if (out.maxOp)
        if (auto c = dyn_cast<arith::ConstantOp>(out.maxOp.getRhs().getDefiningOp()))
            scInfo.min = mlir::cast<IntegerAttr>(c.getValue()).getInt();
    if (out.addOp)
        if (auto c = dyn_cast<arith::ConstantOp>(out.addOp.getRhs().getDefiningOp()))
            scInfo.zp = mlir::cast<IntegerAttr>(c.getValue()).getInt();
    return !!out.applyScaleOp;
}

/// Returns true if op is the fused f32-relu6-clamp-then-truncf linalg.generic:
///   ins(%f32_tensor) outs(%bf16_tensor)
///   body: cmpf(ult) + select + cmpf(ugt) + select + truncf
static bool isFusedF32ClampTruncf(linalg::GenericOp op) {
    if (!op || op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
        return false;
    auto inputElemTy = dyn_cast<ShapedType>(op.getInputs()[0].getType()).getElementType();
    auto outputElemTy = dyn_cast<ShapedType>(op.getResultTypes()[0]).getElementType();
    if (!inputElemTy.isF32() || !outputElemTy.isBF16())
        return false;
    bool hasCmpfUlt = false, hasCmpfUgt = false, hasTruncf = false;
    for (auto &bodyOp : op.getRegion().front().without_terminator()) {
        if (auto cmp = dyn_cast<arith::CmpFOp>(bodyOp)) {
            if (cmp.getPredicate() == arith::CmpFPredicate::ULT)
                hasCmpfUlt = true;
            if (cmp.getPredicate() == arith::CmpFPredicate::UGT)
                hasCmpfUgt = true;
        }
        else if (isa<arith::TruncFOp>(bodyOp)) {
            hasTruncf = true;
        }
    }
    return hasCmpfUlt && hasCmpfUgt && hasTruncf;
}

/// Extract the f32 clamp bounds from the fused linalg.generic body and store
/// them as int32 values in ScaleClampInfo.min / ScaleClampInfo.max.
/// Returns true on success.
static bool extractF32ClampTruncfInfo(linalg::GenericOp op, ScaleClampInfo &scInfo) {
    if (!isFusedF32ClampTruncf(op))
        return false;
    for (auto &bodyOp : op.getRegion().front().without_terminator()) {
        auto cmp = dyn_cast<arith::CmpFOp>(bodyOp);
        if (!cmp)
            continue;
        auto boundConst = cmp.getRhs().getDefiningOp<arith::ConstantOp>();
        if (!boundConst)
            continue;
        auto floatAttr = dyn_cast<FloatAttr>(boundConst.getValue());
        if (!floatAttr)
            continue;
        // Store the IEEE 754 binary representation of the f32 value as int32.
        // bitcastToAPInt() gives the raw bit pattern without any numeric conversion.
        int32_t boundBits =
            static_cast<int32_t>(floatAttr.getValue().bitcastToAPInt().getZExtValue());
        if (cmp.getPredicate() == arith::CmpFPredicate::ULT)
            scInfo.min = boundBits;
        else if (cmp.getPredicate() == arith::CmpFPredicate::UGT)
            scInfo.max = boundBits;
    }
    return true;
}

FailureOr<Value>
computeRescaleInfo(FusionPlan &fusionPlan, Value biasScale, ScaleClampInfo &scInfo) {
    // Parse the terminal rescale generic and reconstruct explicit scale/clamp metadata.
    // The goal is to materialize per-channel scale values and interleave them with biasScale.
    auto lastOp = fusionPlan.neededOps.back();
    if (!isRescale(lastOp)) {
        LLVM_DEBUG({ llvm::dbgs() << "computeRescaleInfo: last op is not a rescale op\n"; });
        return biasScale;
    }
    auto rescaleOp = dyn_cast<linalg::GenericOp>(lastOp);

    // Dispatch: fused f32 relu6-clamp+truncf path — extract float clamp bounds as int32
    // and return early (no quantized scale tensor to interleave).
    if (isFusedF32ClampTruncf(rescaleOp)) {
        LLVM_DEBUG({ llvm::dbgs() << "computeRescaleInfo: handling fused f32 clamp+truncf\n"; });
        extractF32ClampTruncfInfo(rescaleOp, scInfo);
        return biasScale;
    }

    // --- int8 quantized rescale path ---
    // Rescale Op Ex:
    //  %21 = linalg.generic ... ins(%expanded, %cst_1, %cst_2 : tensor<...xi32>, tensor<24xi32>,
    //  tensor<24xi8>) outs(%extracted_slice_8 : tensor<...xi8>) {
    //  ^bb0(%in: i32, %in_9: i32, %in_10: i8, %out: i8):
    //    %22 = tosa.apply_scale %in, %in_9, %in_10 {double_round = true} : (i32,i32,i8) -> i32
    //    %23 = arith.addi %22, %c-3_i32 : i32
    //    %24 = arith.maxsi %23, %c-128_i32 : i32
    //    %25 = arith.minsi %24, %c127_i32 : i32
    //    %26 = arith.trunci %25 : i32 to i8
    //    linalg.yield %26 : i8
    //  } -> tensor<...xi8>
    Int8RescaleBodyOps bodyOps;
    if (!extractInt8RescaleBodyOps(rescaleOp, bodyOps, scInfo)) {
        LLVM_DEBUG({ llvm::dbgs() << "computeRescaleInfo: applyScaleOp not found in rescaleOp\n"; }
        );
        return biasScale;
    }

    OpBuilder builder(lastOp);

    // Lift apply_scale multiplier / shift block-arguments back to the enclosing operands.
    auto mul = bodyOps.applyScaleOp.getMultiplier();
    if (isa<BlockArgument>(mul)) {
        auto mulBArg = mlir::cast<BlockArgument>(mul);
        // Lift block argument back to the parent generic operand so we can reuse upstream tensors.
        Operation *parentOp = mulBArg.getOwner()->getParentOp();
        if (llvm::isa_and_nonnull<linalg::GenericOp>(parentOp))
            mul = parentOp->getOperand(mulBArg.getArgNumber());
    }

    auto shift = bodyOps.applyScaleOp.getShift();
    if (isa<BlockArgument>(shift)) {
        auto shiftBArg = mlir::cast<BlockArgument>(shift);
        // Same for shift: remap region arg to enclosing op operand.
        Operation *parentOp = shiftBArg.getOwner()->getParentOp();
        if (llvm::isa_and_nonnull<linalg::GenericOp>(parentOp))
            shift = parentOp->getOperand(shiftBArg.getArgNumber());
    }

    // Returns a value with single shift value of type tensor<i32>
    auto maybeShiftV = modifyShiftValue(shift, lastOp, scInfo, builder);
    if (failed(maybeShiftV)) {
        LLVM_DEBUG({ llvm::dbgs() << "computeRescaleInfo: failed to modify shift value\n"; });
        return failure();
    }
    auto modShiftV = *maybeShiftV;

    auto shiftTy = shift.getType();
    // Cache type casts to avoid redundant dyn_cast operations
    auto shiftShapedTy = mlir::dyn_cast<ShapedType>(shiftTy);
    auto shiftIntTy = mlir::dyn_cast<IntegerType>(shiftTy);

    auto maybeMulV = modifyMulValue(mul, biasScale, lastOp, builder);
    if (failed(maybeMulV)) {
        LLVM_DEBUG({ llvm::dbgs() << "computeRescaleInfo: failed to modify mul value\n"; });
        return failure();
    }
    auto mulV = *maybeMulV;
    auto mulTy = mlir::dyn_cast<ShapedType>(mulV.getType());

    Value finalScaleV;

    auto biasScaleTy = dyn_cast<ShapedType>(biasScale.getType());
    auto newScaleV =
        tensor::EmptyOp::create(
            builder, lastOp->getLoc(), biasScaleTy.getShape().back(), builder.getI32Type()
        )
            .getResult();

    llvm::SmallVector<AffineMap, 4> rescaleMap;
    AffineMap mulVMap = builder.getDimIdentityMap();
    AffineMap shiftVMap = builder.getDimIdentityMap();
    AffineMap modScaleVMap =
        AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, builder.getAffineConstantExpr(0));

    // Build indexing maps that support scalar or length-1 multiplier/shift via broadcast.
    if (mulTy.getShape()[0] == 1) {
        mulVMap =
            AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, builder.getAffineConstantExpr(0));
    }
    if (shiftShapedTy) {
        if (shiftShapedTy.getShape()[0] == 1) {
            shiftVMap = modScaleVMap;
        }
    }
    else if (shiftIntTy) {
        // Scalar integer shift is normalized to tensor form produced by modifyShiftValue.
        shift = modShiftV;
        shiftVMap = modScaleVMap;
    }
    rescaleMap.push_back(mulVMap);
    rescaleMap.push_back(shiftVMap);
    rescaleMap.push_back(modScaleVMap);
    rescaleMap.push_back(builder.getDimIdentityMap());

    auto rescaleLinalg = linalg::GenericOp::create(
        builder, lastOp->getLoc(), newScaleV.getType(),
        llvm::ArrayRef<Value>{mulV, shift, modShiftV}, llvm::ArrayRef<Value>{newScaleV}, rescaleMap,
        llvm::SmallVector<utils::IteratorType, 4>(1, utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
            // Effective multiplier := multiplier >> (orig_shift - normalized_shift).
            // This folds shift normalization into scale tensor generation.
            auto arg1I32 = arith::ExtSIOp::create(builder, loc, builder.getI32Type(), args[1]);
            auto arg2I32 = arith::ExtSIOp::create(builder, loc, builder.getI32Type(), args[2]);
            auto shift = arith::SubIOp::create(builder, loc, arg1I32, arg2I32);
            auto multiplier = arith::ShRSIOp::create(b, loc, args[0], shift);
            linalg::YieldOp::create(b, loc, ArrayRef<Value>{multiplier});
        }
    );
    finalScaleV = rescaleLinalg.getResult(0);
    LLVM_DEBUG({
        llvm::dbgs() << "computeRescaleInfo: created rescaleLinalg:\n";
        rescaleLinalg.print(llvm::dbgs());
        llvm::dbgs() << "\n";
    });

    // Interleave [bias, scale] per channel into a single 1D tensor:
    // even indices  <- biasScale, odd indices <- finalScaleV.

    SmallVector<int64_t, 1> sh{biasScaleTy.getShape()[0] * 2};

    // HW expects biasScale tensor to be int32
    auto iTy = RankedTensorType::get(sh, builder.getIntegerType(32));

    auto init =
        tensor::EmptyOp::create(builder, lastOp->getLoc(), iTy.getShape(), iTy.getElementType())
            .getResult();

    auto iOp = tensor::InsertSliceOp::create(
                   builder, lastOp->getLoc(), biasScale, init,
                   llvm::ArrayRef<OpFoldResult>{builder.getIndexAttr(0)},
                   llvm::ArrayRef<OpFoldResult>{
                       builder.getIndexAttr(biasScaleTy.getShape()[0]),
                   },
                   llvm::ArrayRef<OpFoldResult>{builder.getIndexAttr(2)}
    )
                   .getResult();

    // Insert scales at odd offsets with stride=2 to complete interleaving.
    iOp = tensor::InsertSliceOp::create(
              builder, lastOp->getLoc(), finalScaleV, iOp,
              llvm::ArrayRef<OpFoldResult>{builder.getIndexAttr(1)},
              llvm::ArrayRef<OpFoldResult>{
                  builder.getIndexAttr(biasScaleTy.getShape()[0]),
              },
              llvm::ArrayRef<OpFoldResult>{builder.getIndexAttr(2)}
    )
              .getResult();

    biasScale = iOp;
    setCompileTimeConstAttr(biasScale.getDefiningOp());
    return biasScale;
}

FailureOr<Value> computeBiasForMatmul(
    FusionPlan &fusionPlan, int channelDim, std::optional<Value> &optionalWeightZp, bool isFC
) {
    auto anchor = fusionPlan.anchor;
    auto anchorTy = anchor ? dyn_cast<ShapedType>(anchor->getResult(0).getType()) : nullptr;
    if (isFC && anchorTy &&
        (anchorTy.getElementType().isBF16() || anchorTy.getElementType().isF32())) {
        // Floating-point matmul/fc lowering keeps explicit post-op adds in the graph.
        // Using fusion-derived bias here can accidentally capture dynamic activation
        // tensors (e.g. truncated matmul outputs) instead of static per-channel bias.
        // Use neutral bias and let the explicit add op carry the real bias.
        int64_t biasDim = anchorTy.getShape().size() > 1 ? anchorTy.getShape()[1] : 1;
        OpBuilder builder(anchor->getContext());
        builder.setInsertionPoint(anchor);
        auto zeroBiasTy = RankedTensorType::get({biasDim}, anchorTy.getElementType());
        Value bias =
            arith::ConstantOp::create(builder, anchor->getLoc(), builder.getZeroAttr(zeroBiasTy));
        optionalWeightZp.reset();
        return bias;
    }

    if (channelDim > 1) {
        return computeBias(fusionPlan, channelDim, optionalWeightZp, 1);
    }
    return computeBias(fusionPlan, channelDim, optionalWeightZp, 0);
}

FailureOr<Value> computeBiasAndRescaleInfo(
    FusionPlan &fusionPlan, int channelDim, std::optional<Value> &optionalWeightZp,
    ScaleClampInfo &scInfo
) {
    auto maybeBias = computeBias(fusionPlan, channelDim, optionalWeightZp);
    if (failed(maybeBias)) {
        return failure();
    }
    return computeRescaleInfo(fusionPlan, *maybeBias, scInfo);
}

FailureOr<Value> buildWeightWithZp(Value weights, Value weightZp, PatternRewriter &rewriter) {
    // Convert quantized weights into zero-point-adjusted signed domain:
    //   adjusted_w = sext(weight_i8) - trunc(weight_zp)
    // Result is materialized as i16 to preserve headroom for downstream arithmetic.
    auto wTy = mlir::cast<RankedTensorType>(weights.getType());
    auto rankedI16Type = RankedTensorType::get(wTy.getShape(), rewriter.getIntegerType(16));
    auto initTensor = createInitTensor(*weights.getDefiningOp(), rewriter, rankedI16Type);
    SmallVector<utils::IteratorType> iterTypes(wTy.getRank(), utils::IteratorType::parallel);
    SmallVector<AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(wTy.getRank()),
        // Treat weightZp as a scalar-like/broadcasted input across all weight elements.
        AffineMap::get(wTy.getRank(), 0, rewriter.getAffineConstantExpr(0), rewriter.getContext()),
        rewriter.getMultiDimIdentityMap(wTy.getRank())
    };
    auto rescaledWeights =
        linalg::GenericOp::create(
            rewriter, weights.getLoc(), rankedI16Type, ValueRange{weights, weightZp},
            ValueRange{initTensor}, indexingMaps, iterTypes,
            [](OpBuilder &b, Location loc, ValueRange args) {
                // Sign-extend weights to i16 first to avoid overflow/underflow on subtraction.
                auto w = arith::ExtSIOp::create(b, loc, b.getIntegerType(16), args[0]);
                // Zero-point is narrowed to i16 so both operands share arithmetic type.
                auto zp = arith::TruncIOp::create(b, loc, b.getIntegerType(16), args[1]);
                auto sub = arith::SubIOp::create(b, loc, w, zp);
                linalg::YieldOp::create(b, loc, ArrayRef<Value>{sub});
            }
        ).getResult(0);
    weights = rescaledWeights;
    // Mark as compile-time-const so later passes can fold/use it as static data.
    setCompileTimeConstAttr(weights.getDefiningOp());
    return weights;
}

} // namespace mlir::syna::torq
