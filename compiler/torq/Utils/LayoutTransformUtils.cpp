// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Utils/LayoutTransformUtils.h"

#include "torq/Utils/ConversionUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::syna::torq {

//===----------------------------------------------------------------------===//
// Permutation Utilities
//===----------------------------------------------------------------------===//

SmallVector<int64_t> nhwcToNchwShape(ArrayRef<int64_t> shape) {
    auto size = shape.size();
    if (size == 4)
        return SmallVector<int64_t>{shape[0], shape[3], shape[1], shape[2]};
    if (size == 3)
        return SmallVector<int64_t>{shape[2], shape[0], shape[1]};
    return SmallVector<int64_t>(shape.begin(), shape.end());
}

SmallVector<int64_t> adaptPermToRank(ArrayRef<int64_t> perm4D, int64_t rank) {
    assert(perm4D.size() == 4 && "expected 4D permutation");
    int64_t nLeading = rank - 3;
    SmallVector<int64_t> result;
    for (int64_t i = 0; i < nLeading; ++i)
        result.push_back(i);
    for (int64_t i = 1; i < 4; ++i)
        result.push_back(perm4D[i] - 1 + nLeading);
    return result;
}

bool isLayoutConversionTranspose(linalg::TransposeOp transposeOp) {
    auto perm = transposeOp.getPermutation();
    return isNchwToNhwcTranspose(perm) || isNhwcToNchwTranspose(perm);
}

//===----------------------------------------------------------------------===//
// Tensor Utilities
//===----------------------------------------------------------------------===//

Value createZeroFilledTensor(
    OpBuilder &rewriter, Location loc, ArrayRef<int64_t> shape, Type elemType
) {
    auto zero = createZeroConstant(rewriter, loc, elemType);
    auto init = tensor::EmptyOp::create(rewriter, loc, shape, elemType);
    return linalg::FillOp::create(rewriter, loc, ValueRange{zero}, ValueRange{init}).getResult(0);
}

Value createMinFilledTensor(
    OpBuilder &rewriter, Location loc, ArrayRef<int64_t> shape, Type elemType
) {
    TypedAttr attr;
    if (auto floatTy = dyn_cast<FloatType>(elemType)) {
        llvm::APFloat minFloat =
            llvm::APFloat::getInf(floatTy.getFloatSemantics(), /*isNegative=*/true);
        attr = FloatAttr::get(floatTy, minFloat);
    }
    else if (auto intTy = dyn_cast<IntegerType>(elemType)) {
        llvm::APInt minInt = intTy.isUnsignedInteger()
                                 ? llvm::APInt::getMinValue(intTy.getWidth())
                                 : llvm::APInt::getSignedMinValue(intTy.getWidth());
        attr = rewriter.getIntegerAttr(intTy, minInt);
    }
    else {
        llvm_unreachable("unsupported element type for min constant");
    }
    auto minConst = arith::ConstantOp::create(rewriter, loc, attr).getResult();
    auto init = tensor::EmptyOp::create(rewriter, loc, shape, elemType);
    return linalg::FillOp::create(rewriter, loc, ValueRange{minConst}, ValueRange{init})
        .getResult(0);
}

Value rebuildGenericWithNewLayout(
    OpBuilder &rewriter, linalg::GenericOp origOp, ValueRange newInputs, Value newOutputInit,
    ArrayRef<AffineMap> newMaps
) {
    Location loc = origOp.getLoc();
    Type newOutType = cast<RankedTensorType>(newOutputInit.getType());

    auto newOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{newOutType}, newInputs, ValueRange{newOutputInit}, newMaps,
        origOp.getIteratorTypesArray(),
        [&](OpBuilder &b, Location bLoc, ValueRange bbArgs) {
            // Clone body: map old block args → new block args, then clone ops.
            IRMapping mapping;
            Block *origBlock = origOp.getBody();
            for (auto [oldArg, newArg] : llvm::zip(origBlock->getArguments(), bbArgs))
                mapping.map(oldArg, newArg);
            for (auto &bodyOp : origBlock->without_terminator())
                b.clone(bodyOp, mapping);
            auto *yield = origBlock->getTerminator();
            SmallVector<Value> yieldVals;
            for (auto v : cast<linalg::YieldOp>(yield).getValues())
                yieldVals.push_back(mapping.lookupOrDefault(v));
            linalg::YieldOp::create(b, yield->getLoc(), yieldVals);
        }
    );
    return newOp.getResult(0);
}

} // namespace mlir::syna::torq
