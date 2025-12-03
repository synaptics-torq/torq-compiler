// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "torq/Utils/ComputeConstants.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/TargetSelect.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

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
    if (!maybeFuseGroupAttr.has_value()) {
        return false;
    }

    if (isa<arith::ConstantOp, tensor::EmptyOp,
            mlir::iree_compiler::IREE::Flow::DispatchTensorLoadOp,
            mlir::iree_compiler::IREE::HAL::InterfaceBindingSubspanOp>(op)) {
        return true;
    }

    assert(op && "Trying to mark null op!");
    if (!isa<TilingInterface, tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op)) {
        op->emitError("Trying to mark an op that does not implement TilingInterface");
        op->dump();
        assert(false && "Trying to mark an op that does not implement TilingInterface");
    }

    rewriter.modifyOpInPlace(op, [&]() {
        // op->setAttr(TORQ_FUSE_GROUP, fuseGroupAttr);
        SmallVector<Attribute> newAttr;
        if (auto oldAttr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP)) {
            if (llvm::is_contained(oldAttr, *maybeFuseGroupAttr)) {
                return;
            }

            newAttr.append(oldAttr.begin(), oldAttr.end());
        }
        newAttr.push_back(*maybeFuseGroupAttr);
        op->setAttr(TORQ_FUSE_GROUP, rewriter.getArrayAttr(newAttr));
    });

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

Operation *getFuseGroupPrincipalOpBackward(Operation *op) {
    ArrayAttr fuseGroupArrAttr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
    assert(fuseGroupArrAttr && "op is not part of a fuse-group");

    assert(fuseGroupArrAttr.size() == 1 && "op is part of multiple fuse-groups");
    IntegerAttr fuseGroupAttr = *fuseGroupArrAttr.getAsRange<IntegerAttr>().begin();

    while (true) {
        if (isFuseGroupPrincipalOp(op, fuseGroupAttr)) {
            return op;
        }

        bool foundSrc = false;
        // Because we are at the output of the principal operation, there should
        // be exactly one source from the same fuse-group.
        for (auto operand : op->getOperands()) {
            auto srcOp = operand.getDefiningOp();
            if (auto attr = srcOp->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
                attr && llvm::is_contained(attr, fuseGroupAttr)) {
                op = srcOp;
                foundSrc = true;
                break;
            }
        }

        if (!foundSrc)
            return nullptr;
    }

    llvm_unreachable("the while loop should terminate by return");
}

SmallVector<OpOperand *>
getFuseGroupPrincipalOpOperandsForward(IntegerAttr fuseGroupAttr, OpResult result) {
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
    return (bool)op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
}

std::optional<int64_t> isFuseGroupOutput(Operation *op) {
    auto fuseGroupAttr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
    if (!fuseGroupAttr) {
        return std::nullopt;
    }

    for (auto intAttr : fuseGroupAttr.getAsRange<IntegerAttr>()) {
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

// Extract and return the tosa multiplier and shift values from a tosa::ApplyScaleOp
MultiplierShiftInfo getMultiplierAndShift(
    linalg::GenericOp genericOp, tosa::ApplyScaleOp applyScaleOp, int scaleValuesCount
) {
    auto tosaMultiplier = applyScaleOp.getMultiplier();
    auto tosaShift = applyScaleOp.getShift();

    auto tosaMultiplierArg = dyn_cast<BlockArgument>(tosaMultiplier);
    auto tosaShiftArg = dyn_cast<BlockArgument>(tosaShift);

    auto tosaMultiplierOpOperand =
        tosaMultiplierArg ? genericOp.getMatchingOpOperand(tosaMultiplierArg) : nullptr;
    auto tosaShiftOpOperand = tosaShiftArg ? genericOp.getMatchingOpOperand(tosaShiftArg) : nullptr;

    // TODO is this correct ?
    auto adaptor = genericOp;

    auto tosaMultiplierValue =
        tosaMultiplierOpOperand ? adaptor.getOperands()[tosaMultiplierOpOperand->getOperandNumber()]
                                : nullptr;
    auto tosaShiftValue = tosaShiftOpOperand
                              ? adaptor.getOperands()[tosaShiftOpOperand->getOperandNumber()]
                              : nullptr;

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
        auto multiplierElementsAttr = computeConstant(tosaMultiplierValue, true, {});
        if (!multiplierElementsAttr) {
            genericOp.emitError() << "multiplier must be a constant";
            return {};
        }

        auto multiplierElements = multiplierElementsAttr.getValues<int32_t>();
        multiplier = std::vector<int32_t>(multiplierElements.begin(), multiplierElements.end());
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
        DenseIntOrFPElementsAttr shiftElementsAttr = computeConstant(tosaShiftValue, true, {});
        if (!shiftElementsAttr) {
            genericOp.emitError() << "shift must be a constant";
            return {};
        }

        auto shiftElements = shiftElementsAttr.getValues<int8_t>();
        shift = std::vector<int8_t>(shiftElements.begin(), shiftElements.end());
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
        rescaleOp.emitError() << "matching error reduction loops > 0!\n";
        return {};
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
            if (auto maybeC = getConstIntValue(shliOp.getRhs())) {
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
                float val = getFloatValue(trueValue).value();
                vals.push_back(val);
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
        LLVM_DEBUG({
            llvm::dbgs() << "must have a single GenericOp scaleClampOp, we return empty object for "
                            "caller to further check\n";
        });
        return {};
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

    if (failed(foldTensorPad(val, padOffsetsAbove, padOffsetsBelow, fillValue)) &&
        failed(foldLinalgFillTensorInsert(val, padOffsetsAbove, padOffsetsBelow, fillValue))) {
        return {};
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

        // FIXME: not clear how the fill value should be represented for bf16 computations
        float floatFillValue = *maybeFloatFillValue;
        maybeFillValue = (int32_t)floatFillValue;
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

            tensor::ExtractSliceOp newExtractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
                value.getLoc(), val, createVector(offsets, rewriter), createVector(sizes, rewriter),
                createVector(strides, rewriter)
            );

            val = newExtractSliceOp.getResult();
        }
    }

    // Update value to fold the padding
    value = val;

    return PaddingInfo{{left, right, top, bottom}, maybeFillValue.value()};
}

std::vector<int32_t> computePerChannelValuesInt(DenseElementsAttr constAttr, int channelDim) {
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

std::vector<float> computePerChannelValuesFloat(DenseElementsAttr constAttr, int channelDim) {
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

    // Compute the result of the add operation assuming value is 0
    DenseElementsAttr constAttr = computeConstant(addOp, true, {inValue, value});
    if (!constAttr) {
        LLVM_DEBUG({
            llvm::dbgs() << "foldForwardPerChannelAdd computeConstant in:"
                         << addOp.getNumDpsInputs() << " FAILED\n";
        });
        return false;
    }

    // Check that all the values in each channel have the same value
    // and update the per-channel bias values
    if (bias.ints.size()) {
        const auto &perChannel = computePerChannelValuesInt(constAttr, channelDim);
        if (!elementwiseAdd(bias.ints, perChannel)) {
            // The constant tensor has unexpected size
            LLVM_DEBUG({ llvm::dbgs() << "foldForwardPerChannelAdd int tensor bad size\n"; });
            return false;
        }
    }
    else if (bias.floats.size()) {
        const auto &perChannel = computePerChannelValuesFloat(constAttr, channelDim);
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

// Compute the constant value for the given LinalgOp.
DenseIntOrFPElementsAttr
computeConstant(LinalgOp linalgOp, bool recursive, const std::vector<Value> &assumeZero) {

    if (linalgOp->getNumResults() != 1) {
        // We only support ops with one output for now.
        return nullptr;
    }

    auto maybeAttr = computeValue(linalgOp->getResult(0), recursive, assumeZero);

    if (failed(maybeAttr)) {
        return nullptr;
    }

    return *maybeAttr;
}

DenseIntOrFPElementsAttr
computeConstant(Value value, bool recursive, const std::vector<Value> &assumeZero) {

    auto maybeAttr = computeValue(value, recursive, assumeZero);

    if (failed(maybeAttr)) {
        return nullptr;
    }

    return *maybeAttr;
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
    Operation *bynaryOp = yieldOp.getOperand(0).getDefiningOp();
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
            llvm::dbgs(
            ) << "elementwiseBinaryOp lhs/rhs is not block arg (or constant if allowed)\n";
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

    auto d2sOp = rewriter.create<torq_hl::DepthToSpaceOp>(
        transposeOp.getLoc(), transposeType(collapseOp.getResult().getType(), d2s_input_type),
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

    auto constOp = dyn_cast<arith::ConstantOp>(input.getDefiningOp());
    if (!constOp) {
        return input;
    }

    int32_t data = 0;
    if (!getIntegerConstantValue(constOp, &data)) {
        LLVM_DEBUG({ llvm::errs() << "cannot get integer constant value\n"; });
        return input;
    }

    data = static_cast<int32_t>(std::round(data * scaleFactor));

    RankedTensorType constType = RankedTensorType::get({}, elementType);
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
    auto output = rewriter.create<arith::ConstantOp>(constOp.getLoc(), constType, value);

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
Value convertWeights(
    mlir::linalg::MatmulOp srcOp, mlir::DenseIntOrFPElementsAttr weightAttr,
    PatternRewriter &rewriter
) {
    // Reorder weights to OIHW
    auto weightElemType = weightAttr.getElementType();
    auto weightShape = dyn_cast<ShapedType>(weightAttr.getType()).getShape();

    // Validate expected shape: [OC, IC] for matmul-style
    assert(weightShape.size() == 2);

    // Assume shape was originally [OC, IC] from matmul-style
    int on = weightShape[0]; // OC
    int in = weightShape[1]; // IC
    int hn = 1;
    int wn = 1;
    std::vector<int64_t> weight_shape{on, in, hn, wn};

    if (weightElemType.isBF16()) {
        auto bfVals = weightAttr.getValues<APFloat>();
        const std::vector<APFloat> bfVec(bfVals.begin(), bfVals.end());
        std::vector<APFloat> reordered = get_weights_OIHW<APFloat>(bfVec, on, hn, wn, in);
        return createFConst(rewriter, srcOp, reordered, weight_shape);
    }
    else if (weightElemType.isInteger(8)) {
        auto rawVals = weightAttr.getValues<int8_t>();
        std::vector<int8_t> reordered(rawVals.begin(), rawVals.end());
        reordered = get_weights_OIHW<int8_t>(reordered, on, hn, wn, in);
        return createI8Const(rewriter, srcOp, reordered, weight_shape);
    }
    else {
        assert(false && "Unsupported weight type");
    }
}

} // namespace mlir::syna::torq
