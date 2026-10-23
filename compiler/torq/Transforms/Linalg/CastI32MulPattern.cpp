// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-cast-i32-mul-pattern"

namespace mlir::syna::torq {

namespace {

// Check if a generic op is a mul operation (elementwise)
static bool isMulOp(linalg::GenericOp genericOp) {
    Operation *mulOp = getElementwiseBinaryOp(genericOp, /*allowConstants=*/true);
    return mulOp && isa<arith::MulIOp>(mulOp);
}

// Check if a generic op is a rescale (contains tosa.apply_scale)
static bool isRescaleOp(linalg::GenericOp genericOp) {
    auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getValues().size() != 1)
        return false;

    Value yieldValue = yieldOp.getValues()[0];
    Operation *defOp = yieldValue.getDefiningOp();

    while (defOp && defOp->getBlock() == genericOp.getBody()) {
        if (isa<tosa::ApplyScaleOp>(defOp))
            return true;

        for (auto operand : defOp->getOperands()) {
            if (operand.getDefiningOp<tosa::ApplyScaleOp>())
                return true;
        }

        if (defOp->getNumOperands() > 0)
            defOp = defOp->getOperand(0).getDefiningOp();
        else
            break;
    }
    return false;
}

// Trace through shape-changing ops (broadcasts, reshapes) to find underlying rescale
// Returns the rescale op if found, nullptr otherwise
static linalg::GenericOp traceToRescaleOp(Value input) {
    Operation *op = input.getDefiningOp();
    if (!op)
        return nullptr;

    // Direct rescale op
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
        if (isRescaleOp(genericOp))
            return genericOp;
    }

    // Trace through collapse_shape, expand_shape, reshape
    if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
        return traceToRescaleOp(collapseOp.getSrc());
    }
    if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
        return traceToRescaleOp(expandOp.getSrc());
    }
    if (auto reshapeOp = dyn_cast<tensor::ReshapeOp>(op)) {
        return traceToRescaleOp(reshapeOp.getSource());
    }

    // Trace through broadcast-like linalg.generic (just yields input)
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
        auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
        if (yieldOp && yieldOp.getNumOperands() == 1) {
            Value yieldValue = yieldOp.getOperand(0);
            // If it's just yielding a block argument, it's a broadcast/copy
            if (auto blockArg = dyn_cast<BlockArgument>(yieldValue)) {
                // Find which input this corresponds to
                unsigned argNum = blockArg.getArgNumber();
                if (argNum < genericOp.getNumDpsInputs()) {
                    return traceToRescaleOp(genericOp.getInputs()[argNum]);
                }
            }
        }
    }

    return nullptr;
}

// Check if all users of an operation (transitively through reshapes/broadcasts) are mul ops
static bool allUsersAreMuls(Operation *op, llvm::DenseSet<Operation *> &visited) {
    if (visited.contains(op))
        return true;
    visited.insert(op);

    for (auto user : op->getUsers()) {
        // Check if user is a mul
        if (auto mulOp = dyn_cast<linalg::GenericOp>(user)) {
            if (isMulOp(mulOp))
                continue;
        }

        // Check if user is an intermediate op (reshape, broadcast)
        if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::ReshapeOp>(user)) {
            if (!allUsersAreMuls(user, visited))
                return false;
            continue;
        }

        // Check if it's a broadcast-like generic
        if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
            auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
            if (yieldOp && yieldOp.getNumOperands() == 1 &&
                isa<BlockArgument>(yieldOp.getOperand(0))) {
                if (!allUsersAreMuls(user, visited))
                    return false;
                continue;
            }
        }

        // Any other user disqualifies this rescale
        return false;
    }

    return true;
}

// Check if rescale's multiplier/shift allows safe i16 conversion
// input(int8) - zeroPoint(int8) = int9
// Validates: (int9 × multiplier / shift) fits in int16
static bool isRescaleSafeForI16(linalg::GenericOp rescaleOp) {
    auto yieldOp = dyn_cast<linalg::YieldOp>(rescaleOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getValues().size() != 1) {
        LLVM_DEBUG({ llvm::dbgs() << "Yield op has wrong number of values\n"; });
        return false;
    }

    // Find apply_scale op
    tosa::ApplyScaleOp applyScaleOp = nullptr;
    for (auto &op : rescaleOp.getBody()->without_terminator()) {
        if (auto applyScale = dyn_cast<tosa::ApplyScaleOp>(op)) {
            applyScaleOp = applyScale;
            break;
        }
    }
    if (!applyScaleOp) {
        LLVM_DEBUG({ llvm::dbgs() << "No apply_scale op found in rescale op\n"; });
        return false;
    }

    auto ms = getMultiplierAndShift(rescaleOp, applyScaleOp, 1);
    if (!ms || ms.multiplier.empty() || ms.shift.empty()) {
        LLVM_DEBUG({ llvm::dbgs() << "Failed to extract multiplier and shift\n"; });
        return false;
    }

    // Check: max(int9) × |multiplier| / 2^shift <= max(int16)
    // i.e., 255 × |multiplier| <= 32767 × 2^shift
    int64_t maxInput = 255;    // max magnitude of (i8 - i8)
    int64_t maxOutput = 32767; // max magnitude for i16
    int64_t absMultiplier = std::abs(static_cast<int64_t>(ms.multiplier[0]));
    int64_t shiftValue = 1LL << ms.shift[0];

    LLVM_DEBUG({
        llvm::dbgs() << "Rescale unsafe for i16: multiplier=" << ms.multiplier[0]
                     << ", shift=" << static_cast<int>(ms.shift[0])
                     << ", effective_scale=" << (double(absMultiplier) / shiftValue) << "\n";
    });

    if (maxInput * absMultiplier > maxOutput * shiftValue) {
        LLVM_DEBUG({ llvm::dbgs() << "Effective scale exceeds i16 range\n"; });
        return false;
    }

    return true;
}

// Convert rescale op from i32 to i16
static linalg::GenericOp convertRescaleToI16(linalg::GenericOp rescaleOp, IRRewriter &rewriter) {
    auto rescaleType = cast<RankedTensorType>(rescaleOp.getResult(0).getType());

    // Create i16 result type
    auto newType = RankedTensorType::get(rescaleType.getShape(), rewriter.getIntegerType(16));

    rewriter.setInsertionPoint(rescaleOp);

    // Create new init tensor
    Value newInit = tensor::EmptyOp::create(
        rewriter, rescaleOp.getLoc(), newType.getShape(), newType.getElementType()
    );

    // Create new generic op with i16 output
    auto newRescaleOp = linalg::GenericOp::create(
        rewriter, rescaleOp.getLoc(), TypeRange{newType}, rescaleOp.getInputs(),
        ValueRange{newInit}, rescaleOp.getIndexingMapsArray(), rescaleOp.getIteratorTypesArray()
    );

    // Create block with correct argument types
    Block &oldBlock = rescaleOp.getRegion().front();
    SmallVector<Type> argTypes;
    for (auto arg : oldBlock.getArguments()) {
        if (arg.getArgNumber() == oldBlock.getNumArguments() - 1) {
            // Output argument becomes i16
            argTypes.push_back(rewriter.getIntegerType(16));
        }
        else {
            // Inputs remain as is
            argTypes.push_back(arg.getType());
        }
    }

    Block *newBlock = rewriter.createBlock(
        &newRescaleOp.getRegion(), newRescaleOp.getRegion().end(), argTypes,
        SmallVector<Location>(argTypes.size(), rescaleOp.getLoc())
    );

    // Map old block args to new block args
    IRMapping mapping;
    for (auto [oldArg, newArg] : llvm::zip(oldBlock.getArguments(), newBlock->getArguments())) {
        mapping.map(oldArg, newArg);
    }

    rewriter.setInsertionPointToStart(newBlock);

    // Clone body, modifying apply_scale to return i16
    for (auto &op : oldBlock.without_terminator()) {
        if (auto applyScale = dyn_cast<tosa::ApplyScaleOp>(op)) {
            SmallVector<Value> newOperands;
            for (auto operand : op.getOperands())
                newOperands.push_back(mapping.lookupOrDefault(operand));

            // Create apply_scale with i32 result type (we'll truncate to i16 after)
            auto newApplyScale = tosa::ApplyScaleOp::create(
                rewriter, op.getLoc(), rewriter.getIntegerType(32), newOperands, op.getAttrs()
            );
            // Truncate i32 -> i16 (safe because isRescaleSafeForI16 already validated range)
            auto truncOp = arith::TruncIOp::create(
                rewriter, op.getLoc(), rewriter.getIntegerType(16), newApplyScale.getResult()
            );
            mapping.map(op.getResult(0), truncOp.getResult());
        }
        else {
            rewriter.clone(op, mapping);
        }
    }

    // Handle yield
    auto oldYield = cast<linalg::YieldOp>(oldBlock.getTerminator());
    SmallVector<Value> newYields;
    for (auto val : oldYield.getValues())
        newYields.push_back(mapping.lookup(val));

    linalg::YieldOp::create(rewriter, oldYield.getLoc(), newYields);

    // Replace old op with new one
    rewriter.replaceOp(rescaleOp, newRescaleOp.getResult(0));
    return newRescaleOp;
}

// Update mul op to accept i16 inputs
void updateMulToI16(linalg::GenericOp mulOp, IRRewriter &rewriter) {
    rewriter.setInsertionPoint(mulOp);

    // Create new mul op
    auto newMulOp = linalg::GenericOp::create(
        rewriter, mulOp.getLoc(), mulOp.getResultTypes(), mulOp.getInputs(), mulOp.getOutputs(),
        mulOp.getIndexingMapsArray(), mulOp.getIteratorTypesArray()
    );

    // Create block with i16 input arguments (matching i16 tensors from rescale)
    Block &oldBlock = mulOp.getRegion().front();
    SmallVector<Type> argTypes;
    for (auto arg : oldBlock.getArguments()) {
        auto argIdx = arg.getArgNumber();
        if (argIdx < mulOp.getNumDpsInputs()) {
            auto inputType = dyn_cast<RankedTensorType>(mulOp.getInputs()[argIdx].getType());
            if (inputType && inputType.getElementType().isInteger(16)) {
                argTypes.push_back(rewriter.getIntegerType(16));
                continue;
            }
        }
        argTypes.push_back(arg.getType());
    }

    Block *newBlock = rewriter.createBlock(
        &newMulOp.getRegion(), newMulOp.getRegion().end(), argTypes,
        SmallVector<Location>(argTypes.size(), mulOp.getLoc())
    );

    IRMapping mapping;
    for (auto [oldArg, newArg] : llvm::zip(oldBlock.getArguments(), newBlock->getArguments())) {
        mapping.map(oldArg, newArg);
    }

    rewriter.setInsertionPointToStart(newBlock);

    // Clone operations, handling muli specially to keep i16 result
    for (auto &op : oldBlock.without_terminator()) {
        if (auto mulI = dyn_cast<arith::MulIOp>(op)) {
            Value lhs = mapping.lookupOrDefault(mulI.getLhs());
            Value rhs = mapping.lookupOrDefault(mulI.getRhs());

            // If operands are now i16, create i16*i16->i16 multiply
            if (lhs.getType().isInteger(16) && rhs.getType().isInteger(16)) {
                Value mul16 = arith::MulIOp::create(
                    rewriter, mulI.getLoc(), rewriter.getIntegerType(16), lhs, rhs,
                    mulI.getOverflowFlags()
                );

                // Extend to i32 if original result was i32
                if (mulI.getType().isInteger(32)) {
                    Value ext32 = arith::ExtSIOp::create(
                        rewriter, mulI.getLoc(), rewriter.getIntegerType(32), mul16
                    );
                    mapping.map(mulI.getResult(), ext32);
                }
                else {
                    mapping.map(mulI.getResult(), mul16);
                }
                continue;
            }
        }
        rewriter.clone(op, mapping);
    }

    // Clone yield
    auto oldYield = cast<linalg::YieldOp>(oldBlock.getTerminator());
    SmallVector<Value> mappedYields;
    for (auto val : oldYield.getValues()) {
        mappedYields.push_back(mapping.lookup(val));
    }
    linalg::YieldOp::create(rewriter, oldYield.getLoc(), mappedYields);

    rewriter.replaceOp(mulOp, newMulOp.getResults());
}

} // namespace

// Update intermediate ops (collapse_shape, broadcasts) to use i16
void updateIntermediateOps(
    Operation *op, IRRewriter &rewriter, llvm::DenseSet<Operation *> &processed
) {
    if (processed.contains(op))
        return;
    processed.insert(op);

    // Collect users first to avoid iterator invalidation when replacing ops
    SmallVector<Operation *> users(op->getUsers().begin(), op->getUsers().end());

    for (auto user : users) {
        // Update collapse_shape
        if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(user)) {
            auto srcType = cast<RankedTensorType>(collapseOp.getSrc().getType());
            if (srcType.getElementType().isInteger(16)) {
                auto resultType = cast<RankedTensorType>(collapseOp.getType());
                auto newResultType =
                    RankedTensorType::get(resultType.getShape(), rewriter.getIntegerType(16));

                rewriter.setInsertionPoint(collapseOp);
                auto newCollapseOp = tensor::CollapseShapeOp::create(
                    rewriter, collapseOp.getLoc(), newResultType, collapseOp.getSrc(),
                    collapseOp.getReassociationIndices()
                );
                rewriter.replaceOp(collapseOp, newCollapseOp.getResult());
                updateIntermediateOps(newCollapseOp.getOperation(), rewriter, processed);
            }
        }
        // Update expand_shape
        else if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(user)) {
            auto srcType = cast<RankedTensorType>(expandOp.getSrc().getType());
            if (srcType.getElementType().isInteger(16)) {
                auto resultType = cast<RankedTensorType>(expandOp.getType());
                auto newResultType =
                    RankedTensorType::get(resultType.getShape(), rewriter.getIntegerType(16));

                rewriter.setInsertionPoint(expandOp);
                auto newExpandOp = tensor::ExpandShapeOp::create(
                    rewriter, expandOp.getLoc(), newResultType, expandOp.getSrc(),
                    expandOp.getReassociationIndices()
                );
                rewriter.replaceOp(expandOp, newExpandOp.getResult());
                updateIntermediateOps(newExpandOp.getOperation(), rewriter, processed);
            }
        }
        // Update broadcast-like generic
        else if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
            auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
            if (yieldOp && yieldOp.getNumOperands() == 1 &&
                isa<BlockArgument>(yieldOp.getOperand(0))) {
                // Check if input is i16
                bool hasI16Input = false;
                for (auto input : genericOp.getInputs()) {
                    auto inputType = dyn_cast<RankedTensorType>(input.getType());
                    if (inputType && inputType.getElementType().isInteger(16)) {
                        hasI16Input = true;
                        break;
                    }
                }

                if (hasI16Input) {
                    auto resultType = cast<RankedTensorType>(genericOp.getResult(0).getType());
                    auto newResultType =
                        RankedTensorType::get(resultType.getShape(), rewriter.getIntegerType(16));

                    rewriter.setInsertionPoint(genericOp);

                    // Create new output tensor
                    Value newInit = tensor::EmptyOp::create(
                        rewriter, genericOp.getLoc(), newResultType.getShape(),
                        newResultType.getElementType()
                    );

                    auto newGenericOp = linalg::GenericOp::create(
                        rewriter, genericOp.getLoc(), TypeRange{newResultType},
                        genericOp.getInputs(), ValueRange{newInit},
                        genericOp.getIndexingMapsArray(), genericOp.getIteratorTypesArray()
                    );

                    // Clone body with updated types
                    Block &oldBlock = genericOp.getRegion().front();
                    SmallVector<Type> argTypes;
                    for (auto arg : oldBlock.getArguments()) {
                        if (arg.getArgNumber() < genericOp.getNumDpsInputs()) {
                            auto inputType = cast<RankedTensorType>(
                                genericOp.getInputs()[arg.getArgNumber()].getType()
                            );
                            argTypes.push_back(inputType.getElementType());
                        }
                        else {
                            argTypes.push_back(rewriter.getIntegerType(16));
                        }
                    }

                    Block *newBlock = rewriter.createBlock(
                        &newGenericOp.getRegion(), newGenericOp.getRegion().end(), argTypes,
                        SmallVector<Location>(argTypes.size(), genericOp.getLoc())
                    );
                    IRMapping mapping;
                    for (auto [oldArg, newArg] :
                         llvm::zip(oldBlock.getArguments(), newBlock->getArguments())) {
                        mapping.map(oldArg, newArg);
                    }

                    rewriter.setInsertionPointToStart(newBlock);
                    auto oldYield = cast<linalg::YieldOp>(oldBlock.getTerminator());
                    SmallVector<Value> newYields;
                    for (auto val : oldYield.getValues()) {
                        newYields.push_back(mapping.lookup(val));
                    }
                    linalg::YieldOp::create(rewriter, oldYield.getLoc(), newYields);

                    rewriter.replaceOp(genericOp, newGenericOp.getResult(0));
                    updateIntermediateOps(newGenericOp.getOperation(), rewriter, processed);
                }
            }
        }
    }
}

// Convert rescale operations from i32 to i16 when they feed into mul operations
//  %189 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
//  affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel",
//  "parallel", "parallel"]} ins(%188 : tensor<1x28x28x240xi8>) outs(%176 : tensor<1x28x28x240xi32>)
//  {
//   ^bb0(%in: i8, %out: i32):
//     %583 = arith.extsi %in : i8 to i32
//     %584 = arith.subi %583, %c-6_i32 : i32
//     %585 = tosa.apply_scale %584, %c1073741824_i32, %c30_i8 {double_round = false} : (i32, i32,
//     i8) -> i32 linalg.yield %585 : i32
//   } -> tensor<1x28x28x240xi32>
//   %190 = tensor.empty() : tensor<1x1x1x240xi32>
//   %191 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
//   iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%190 :
//   tensor<1x1x1x240xi32>) { ^bb0(%out: i32):
//     %583 = tosa.apply_scale %c128_i32, %c1073741824_i32, %c30_i8 {double_round = false} : (i32,
//     i32, i8) -> i32 linalg.yield %583 : i32
//   } -> tensor<1x1x1x240xi32>
//   %192 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>,
//   affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2,
//   d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%189, %191 :
//   tensor<1x28x28x240xi32>, tensor<1x1x1x240xi32>) outs(%176 : tensor<1x28x28x240xi32>) {
//   ^bb0(%in: i32, %in_234: i32, %out: i32):
//     %583 = arith.muli %in, %in_234 : i32
//     linalg.yield %583 : i32
//   } -> tensor<1x28x28x240xi32>
//               |
//               v
// %227 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
// affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel",
// "parallel", "parallel"]} ins(%225 : tensor<1x28x28x240xi8>) outs(%226 : tensor<1x28x28x240xi16>)
// {
//   ^bb0(%in: i8, %out: i16):
//     %707 = arith.extsi %in : i8 to i32
//     %708 = arith.subi %707, %c-6_i32 : i32
//     %709 = tosa.apply_scale %708, %c1073741824_i32, %c30_i8 {double_round = false} : (i32, i32,
//     i8) -> i16 linalg.yield %709 : i16
//   } -> tensor<1x28x28x240xi16>
//   %228 = tensor.empty() : tensor<1x1x1x240xi32>
//   %229 = tensor.empty() : tensor<1x1x1x240xi16>
//   %230 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
//   iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%229 :
//   tensor<1x1x1x240xi16>) { ^bb0(%out: i16):
//     %707 = tosa.apply_scale %c128_i32, %c1073741824_i32, %c30_i8 {double_round = false} : (i32,
//     i32, i8) -> i16 linalg.yield %707 : i16
//   } -> tensor<1x1x1x240xi16>
//   %collapsed_155 = tensor.collapse_shape %230 [[0], [1, 2, 3]] : tensor<1x1x1x240xi16> into
//   tensor<1x240xi16> %231 = tensor.empty() : tensor<1x28x28x240xi32> %232 = tensor.empty() :
//   tensor<1x28x28x240xi16> %233 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) ->
//   (d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel",
//   "parallel", "parallel", "parallel"]} ins(%collapsed_155 : tensor<1x240xi16>) outs(%232 :
//   tensor<1x28x28x240xi16>) { ^bb0(%in: i16, %out: i16):
//     linalg.yield %in : i16
//   } -> tensor<1x28x28x240xi16>
//   %234 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
//   affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2,
//   d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%227, %233 :
//   tensor<1x28x28x240xi16>, tensor<1x28x28x240xi16>) outs(%210 : tensor<1x28x28x240xi32>) {
//   ^bb0(%in: i16, %in_251: i16, %out: i32):
//     %707 = arith.muli %in, %in_251 : i16
//     %708 = arith.extsi %707 : i16 to i32
//     linalg.yield %708 : i32
//   } -> tensor<1x28x28x240xi32>
void convertRescaleI32ToI16ForMul(FunctionOpInterface funcOp, IRRewriter &rewriter) {
    // Collect rescale ops that ONLY feed into elementwise mul operations
    llvm::DenseSet<linalg::GenericOp> rescaleOpsToConvert;
    llvm::DenseSet<linalg::GenericOp> mulOpsToUpdate;

    // Find all mul operations and check their inputs
    funcOp.walk([&](linalg::GenericOp mulOp) {
        // Must be an elementwise mul
        if (!isMulOp(mulOp)) {
            return;
        }

        llvm::SmallVector<linalg::GenericOp> inputRescaleOps;

        // Check each input of the mul operation
        for (auto input : mulOp.getInputs()) {
            auto inputType = dyn_cast<RankedTensorType>(input.getType());

            // Input must be i32
            if (!inputType || !inputType.getElementType().isInteger(32)) {
                LLVM_DEBUG({ llvm::dbgs() << "Input is not i32\n"; });
                return;
            }

            // Trace through broadcasts/reshapes to find underlying rescale operation
            auto rescaleOp = traceToRescaleOp(input);
            if (!rescaleOp) {
                LLVM_DEBUG({ llvm::dbgs() << "Input is not a rescale op\n"; });
                return;
            }

            inputRescaleOps.push_back(rescaleOp);
        }

        for (auto rescaleOp : inputRescaleOps) {
            llvm::DenseSet<Operation *> visited;
            // Check that all these rescale ops are ONLY used by elementwise mul ops
            // (possibly through intermediate reshape/broadcast operations)
            if (!allUsersAreMuls(rescaleOp.getOperation(), visited)) {
                LLVM_DEBUG({ llvm::dbgs() << "Rescale op has non-mul users\n"; });
                return;
            }
            // Verify rescale outputs fit in i16: (int9 × multiplier/shift) <= int16
            if (!isRescaleSafeForI16(rescaleOp)) {
                LLVM_DEBUG({ llvm::dbgs() << "Rescale produces values exceeding i16 range\n"; });
                return;
            }
        }

        // Convert only if all conditions are satisfied
        LLVM_DEBUG({ llvm::dbgs() << "Found mul with rescale inputs to convert\n"; });
        for (auto rescaleOp : inputRescaleOps) {
            rescaleOpsToConvert.insert(rescaleOp);
        }
        mulOpsToUpdate.insert(mulOp);
    });

    // Convert the identified rescale operations to i16
    llvm::DenseSet<Operation *> processedOps;
    for (auto rescaleOp : rescaleOpsToConvert) {
        auto newRescaleOp = convertRescaleToI16(rescaleOp, rewriter);
        // Update all intermediate reshape/broadcast ops
        updateIntermediateOps(newRescaleOp.getOperation(), rewriter, processedOps);
    }

    // Update mul operations to use i16 inputs
    for (auto mulOp : mulOpsToUpdate) {
        updateMulToI16(mulOp, rewriter);
    }
}

static llvm::cl::opt<bool> clFoldMulInputZp(
    "torq-fold-mul-input-zp",
    llvm::cl::desc("Fold the zero-point subtraction of a mul input into the mul"),
    llvm::cl::init(true)
);

static bool hasIdentityMaps(linalg::GenericOp op) {
    return llvm::all_of(op.getIndexingMapsArray(), [](AffineMap map) { return map.isIdentity(); });
}

// The i8 input and the zero point of a rescale that convertRescaleI32ToI16ForMul made and that
// only subtracts the zero point:
//   ^bb0(%in: i8, %out: i16):
//     %0 = arith.extsi %in : i8 to i32
//     %1 = arith.subi %0, %c_zp : i32    (no subi for zero point 0)
//     %2 = tosa.apply_scale %1, %c_2^s, %c_s : (i32, i32, i8) -> i32
//     %3 = arith.trunci %2 : i32 to i16
//     linalg.yield %3 : i16
struct ZeroPointRescale {
    Value input;
    int32_t zp;
};

static std::optional<ZeroPointRescale> matchZeroPointRescale(Value value) {
    auto op = value.getDefiningOp<linalg::GenericOp>();
    if (!op || op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 || !hasIdentityMaps(op) ||
        !getElementTypeOrSelf(op.getInputs()[0].getType()).isInteger(8) ||
        !getElementTypeOrSelf(value.getType()).isInteger(16))
        return std::nullopt;

    auto yieldOp = cast<linalg::YieldOp>(op.getBody()->getTerminator());
    auto truncOp = yieldOp.getOperand(0).getDefiningOp<arith::TruncIOp>();
    if (!truncOp)
        return std::nullopt;
    auto applyScaleOp = truncOp.getIn().getDefiningOp<tosa::ApplyScaleOp>();
    if (!applyScaleOp)
        return std::nullopt;
    auto multiplier = getConstIntValue(applyScaleOp.getMultiplier());
    auto shift = getConstIntValue(applyScaleOp.getShift());
    if (!multiplier || !shift || *shift < 0 || *shift > 30 || *multiplier != (1LL << *shift))
        return std::nullopt;

    Value scaled = applyScaleOp.getValue();
    int32_t zp = 0;
    if (auto subiOp = scaled.getDefiningOp<arith::SubIOp>()) {
        auto maybeZp = getConstIntValue(subiOp.getRhs());
        if (!maybeZp)
            return std::nullopt;
        zp = static_cast<int32_t>(*maybeZp);
        scaled = subiOp.getLhs();
    }
    auto extOp = scaled.getDefiningOp<arith::ExtSIOp>();
    if (!extOp || extOp.getIn() != op.getBody()->getArgument(0))
        return std::nullopt;
    return ZeroPointRescale{op.getInputs()[0], zp};
}

// A two-input i16 mul with identity maps and a ranked i32 result, as
// convertRescaleI32ToI16ForMul leaves it. The input1 offset is a vector along the last output
// dimension, so a rank-0 mul does not fold.
static bool isFoldableMul(linalg::GenericOp op) {
    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    return isMulOp(op) && op.getNumDpsInputs() == 2 && hasIdentityMaps(op) &&
           op.getInputs()[0] != op.getInputs()[1] && resultType && resultType.getRank() > 0 &&
           resultType.getElementType().isInteger(32);
}

// The offset -zp is stored as i8, so the zero point of an input1 with an offset must be in
// [-127, 127]
static bool fitsI8Offset(int32_t zp) { return zp >= -127 && zp <= 127; }

// The new inputs of a mul after the fold, and the zero point that stays on input1
struct MulZeroPointFold {
    Value input1;
    Value input2;
    int32_t zp;
};

// Choose which mul inputs to fold. The kernel can add an offset only to input1, and only one
// that fits in i8, so:
// - an input with a zero point in [-127, 127] becomes input1; a non-zero zero point is
//   preferred, so that the other input can still fold if its zero point is 0;
// - the other input is folded too when its zero point is 0 (a plain sign extension), else it
//   stays the i16 rescale.
static std::optional<MulZeroPointFold> chooseMulZeroPointFold(linalg::GenericOp mulOp) {
    std::optional<ZeroPointRescale> rescales[2] = {
        matchZeroPointRescale(mulOp.getInputs()[0]), matchZeroPointRescale(mulOp.getInputs()[1])
    };
    int first = -1;
    for (int i : {0, 1}) {
        if (rescales[i] && fitsI8Offset(rescales[i]->zp) && (first < 0 || rescales[first]->zp == 0))
            first = i;
    }
    if (first < 0)
        return std::nullopt;

    int second = 1 - first;
    Value input2 = rescales[second] && rescales[second]->zp == 0 ? rescales[second]->input
                                                                 : mulOp.getInputs()[second];
    return MulZeroPointFold{rescales[first]->input, input2, rescales[first]->zp};
}

// Body of the folded mul: (extsi(input1) - zp) * extsi(input2) in i32, with no subi for zp 0
static void buildMulWithInput1ZpBody(OpBuilder &b, Location loc, ValueRange args, int32_t zp) {
    Type i32 = b.getI32Type();
    Value lhs = arith::ExtSIOp::create(b, loc, i32, args[0]);
    if (zp != 0) {
        Value zpValue = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(zp));
        lhs = arith::SubIOp::create(b, loc, lhs, zpValue);
    }
    Value rhs = arith::ExtSIOp::create(b, loc, i32, args[1]);
    linalg::YieldOp::create(b, loc, ValueRange{arith::MulIOp::create(b, loc, lhs, rhs)});
}

// Replace the mul with one that reads the folded inputs
static void
rebuildMulWithFold(linalg::GenericOp mulOp, const MulZeroPointFold &fold, IRRewriter &rewriter) {
    LLVM_DEBUG(
        llvm::dbgs() << "Folding mul input zero point " << fold.zp << " into " << mulOp << "\n"
    );
    rewriter.setInsertionPoint(mulOp);
    auto newMulOp = linalg::GenericOp::create(
        rewriter, mulOp.getLoc(), mulOp.getResultTypes(), ValueRange{fold.input1, fold.input2},
        mulOp.getOutputs(), mulOp.getIndexingMapsArray(), mulOp.getIteratorTypesArray(),
        [&](OpBuilder &b, Location loc, ValueRange args) {
            buildMulWithInput1ZpBody(b, loc, args, fold.zp);
        }
    );
    rewriter.replaceOp(mulOp, newMulOp.getResults());
}

// Move the zero-point rescales of an i16 mul (from convertRescaleI32ToI16ForMul) into the mul,
// so they need no slice task of their own:
//   %a16 = rescale(%a8, zp_a)  %b16 = rescale(%b8, zp_b)  %m = muli(%a16, %b16)
//               |
//               v
//   ^bb0(%a: i8, %b: i8 or i16, %out: i32):
//     %m = arith.muli (arith.subi (arith.extsi %a), zp_a), (arith.extsi %b)
static void foldMulInputZeroPoints(FunctionOpInterface funcOp, IRRewriter &rewriter) {
    SmallVector<linalg::GenericOp> mulOps;
    funcOp.walk([&](linalg::GenericOp op) {
        if (isFoldableMul(op))
            mulOps.push_back(op);
    });

    for (linalg::GenericOp mulOp : mulOps) {
        if (std::optional<MulZeroPointFold> fold = chooseMulZeroPointFold(mulOp))
            rebuildMulWithFold(mulOp, *fold, rewriter);
    }
}

void populateCastI32MulPatterns(FunctionOpInterface funcOp, IRRewriter &rewriter) {
    convertRescaleI32ToI16ForMul(funcOp, rewriter);
    if (clFoldMulInputZp)
        foldMulInputZeroPoints(funcOp, rewriter);
}

} // namespace mlir::syna::torq
