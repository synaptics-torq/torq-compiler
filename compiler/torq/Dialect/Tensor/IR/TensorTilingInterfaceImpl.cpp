#include "torq/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"

#include "torq/Dialect/Tensor/IR/Utils.h"

#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::tensor;
using namespace mlir::syna::torq::tensor;

namespace {

Value makeTiledCollapsedSrcShape(
    OpBuilder &builder, tensor::CollapseShapeOp collapseOp, ArrayRef<OpFoldResult> ivs,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<OpFoldResult> sizeBounds, bool omitPartialTileCheck
) {
    Location loc = collapseOp->getLoc();
    Value valueToTile = collapseOp.getSrc();

    std::optional<linalg::SliceParameters> sliceParams = computeCollapseSliceParameters(
        builder, collapseOp, ivs, tileSizes, sizeBounds, omitPartialTileCheck
    );

    return sliceParams.has_value() ? builder.create<tensor::ExtractSliceOp>(
                                         loc, valueToTile, sliceParams->offsets, sliceParams->sizes,
                                         sliceParams->strides
                                     )
                                   : valueToTile;
}

/// Implementation of the `TilingInterface` for `tensor::CollapseShapeOp`.
struct CollapseShapeOpTilingInterface
    : public TilingInterface::ExternalModel<
          CollapseShapeOpTilingInterface, tensor::CollapseShapeOp> {

    SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
        auto collapseOp = cast<CollapseShapeOp>(op);

        SmallVector<utils::IteratorType> iteratorTypes(
            collapseOp.getResultType().getRank(), utils::IteratorType::parallel
        );
        return iteratorTypes;
    }

    SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
        ReifiedRankedShapedTypeDims reifiedShapes;
        (void)reifyResultShapes(b, op, reifiedShapes);
        OpFoldResult zero = b.getIndexAttr(0);
        OpFoldResult one = b.getIndexAttr(1);
        // Initialize all the ranges to {zero, one, one}. All the `ub`s are
        // overwritten.
        SmallVector<Range> loopRanges(reifiedShapes[0].size(), {zero, one, one});
        for (const auto &ub : enumerate(reifiedShapes[0]))
            loopRanges[ub.index()].size = ub.value();
        return loopRanges;
    }

    FailureOr<TilingResult> getTiledImplementation(
        Operation *op, OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes
    ) const {
        tensor::CollapseShapeOp collapseOp = cast<tensor::CollapseShapeOp>(op);

        Value tiledSrc = makeTiledCollapsedSrcShape(b, collapseOp, offsets, sizes, {}, true);
        RankedTensorType tiledSrcTensor = cast<RankedTensorType>(tiledSrc.getType());

        // Infer the shape of the tiled result from the tiled source. For expand_shape (below), we
        // can't infer this shape from the source, so we do something else, much simpler. Can we do
        // the same here?
        SmallVector<int64_t, 4> tiledResultShape;
        for (auto reassocGroup : collapseOp.getReassociationIndices()) {
            int64_t sizeProd = 1;
            for (auto srcDimIdx : reassocGroup) {
                int64_t srcDimSize = tiledSrcTensor.getDimSize(srcDimIdx);
                if (ShapedType::isDynamic(srcDimSize)) {
                    sizeProd = ShapedType::kDynamic;
                    break;
                }
                sizeProd *= srcDimSize;
            }
            tiledResultShape.push_back(sizeProd);
        }

        RankedTensorType resultTensor =
            RankedTensorType::get(tiledResultShape, tiledSrcTensor.getElementType());

        Operation *tiledOp = clone(b, op, resultTensor, tiledSrc);

        return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
    }

    LogicalResult getResultTilePosition(
        Operation *op, OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
        SmallVector<OpFoldResult> &resultSizes
    ) const {
        resultOffsets.assign(offsets.begin(), offsets.end());
        resultSizes.assign(sizes.begin(), sizes.end());
        return success();
    }

    LogicalResult getIterationDomainTileFromResultTile(
        Operation *op, OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes, SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
        SmallVectorImpl<OpFoldResult> &iterDomainSizes
    ) const {
        iterDomainOffsets.assign(offsets.begin(), offsets.end());
        iterDomainSizes.assign(sizes.begin(), sizes.end());
        return success();
    }

    FailureOr<TilingResult> generateResultTileValue(
        Operation *op, OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes
    ) const {
        return getTiledImplementation(op, b, offsets, sizes);
    }
};

Value makeTiledExpandSrcShape(
    OpBuilder &builder, tensor::ExpandShapeOp expandOp, ArrayRef<OpFoldResult> ivs,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<OpFoldResult> sizeBounds, bool omitPartialTileCheck
) {
    Location loc = expandOp->getLoc();
    Value valueToTile = expandOp.getSrc();

    std::optional<linalg::SliceParameters> sliceParams = computeExpandSliceParameters(
        builder, expandOp, ivs, tileSizes, sizeBounds, omitPartialTileCheck
    );

    return sliceParams.has_value() ? builder.create<tensor::ExtractSliceOp>(
                                         loc, valueToTile, sliceParams->offsets, sliceParams->sizes,
                                         sliceParams->strides
                                     )
                                   : valueToTile;
}

/// Implementation of the `TilingInterface` for `tensor::ExpandShapeOp`.
struct ExpandShapeOpTilingInterface
    : public TilingInterface::ExternalModel<ExpandShapeOpTilingInterface, tensor::ExpandShapeOp> {

    SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
        auto expandOp = cast<ExpandShapeOp>(op);
        SmallVector<utils::IteratorType> iteratorTypes(
            expandOp.getResultType().getRank(), utils::IteratorType::parallel
        );
        return iteratorTypes;
    }

    SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
        ReifiedRankedShapedTypeDims reifiedShapes;
        (void)reifyResultShapes(b, op, reifiedShapes);
        OpFoldResult zero = b.getIndexAttr(0);
        OpFoldResult one = b.getIndexAttr(1);
        // Initialize all the ranges to {zero, one, one}. All the `ub`s are
        // overwritten.
        SmallVector<Range> loopRanges(reifiedShapes[0].size(), {zero, one, one});
        for (const auto &ub : enumerate(reifiedShapes[0]))
            loopRanges[ub.index()].size = ub.value();
        return loopRanges;
    }

    FailureOr<TilingResult> getTiledImplementation(
        Operation *op, OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes
    ) const {
        tensor::ExpandShapeOp expandOp = cast<tensor::ExpandShapeOp>(op);

        Value tiledSrc = makeTiledExpandSrcShape(b, expandOp, offsets, sizes, {}, true);

        RankedTensorType tiledSrcTensor = cast<RankedTensorType>(tiledSrc.getType());

        // To compute this shape linalg relies on the fact that, for linalg operations, one of the
        // operands (output) has the same shape as the result. Since this is not the case here, and
        // there's no way to infer this shape from the operand, I think the input argument `sizes`
        // describe the shape.
        SmallVector<int64_t, 4> tiledResultShape =
            llvm::to_vector(llvm::map_range(sizes, [&](const OpFoldResult &size) {
                std::optional<int64_t> intSize = getConstantIntValue(size);
                return intSize ? *intSize : ShapedType::kDynamic;
            }));

        RankedTensorType resultTensor =
            RankedTensorType::get(tiledResultShape, tiledSrcTensor.getElementType());

        tensor::ExpandShapeOp tiledOp = b.create<tensor::ExpandShapeOp>(
            op->getLoc(), resultTensor, tiledSrc, expandOp.getReassociationIndices()
        );
        // linalg uses clone to create the tiled op, but for ExpandShapeOp it's
        // easier to use create (i.e. build), which will infer the
        // static_output_shape attribute and the output_shape operand. To keep
        // with the (undocumented) side effects of clone, we copy some
        // attributes. As far as I know, nothing relies on this behavior.
        tiledOp->setDiscardableAttrs(op->getDiscardableAttrDictionary());

        return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
    }

    LogicalResult getResultTilePosition(
        Operation *op, OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
        SmallVector<OpFoldResult> &resultSizes
    ) const {
        resultOffsets.assign(offsets.begin(), offsets.end());
        resultSizes.assign(sizes.begin(), sizes.end());
        return success();
    }

    LogicalResult getIterationDomainTileFromResultTile(
        Operation *op, OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes, SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
        SmallVectorImpl<OpFoldResult> &iterDomainSizes
    ) const {
        iterDomainOffsets.assign(offsets.begin(), offsets.end());
        iterDomainSizes.assign(sizes.begin(), sizes.end());
        return success();
    }

    FailureOr<TilingResult> generateResultTileValue(
        Operation *op, OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes
    ) const {
        return getTiledImplementation(op, b, offsets, sizes);
    }
};

// The test tosa_ops/casr-i8-to-i32.mlir crashes when compiled with
// --iree-hal-target-backends=llvm-cpu. The reason is that the function
// getDefaultDistributedLevelTileSizes
// (iree/compiler/src/iree/compiler/Codegen/LLVMCPU/KernelDispatch.cpp) assumes
// that ops that implement TilingInterface also implement
// PartitionableLoopsInterface. Hence we provide a trivial implementation of the
// interface.
template <typename OpTy>
struct PartitionableLoopsInterfaceImpl
    : public mlir::iree_compiler::PartitionableLoopsInterface::ExternalModel<
          PartitionableLoopsInterfaceImpl<OpTy>, OpTy> {

    llvm::SmallVector<unsigned>
    getPartitionableLoops(Operation *op, std::optional<unsigned> maxNumPartitionedLoops) const {
        return {};
    }
};

} // namespace

void mlir::syna::torq::tensor::registerTilingInterfaceExternalModels(DialectRegistry &registry) {
    registry.addExtension(+[](MLIRContext *ctx, TensorDialect *dialect) {
        mlir::tensor::CollapseShapeOp::attachInterface<CollapseShapeOpTilingInterface>(*ctx);
        mlir::tensor::ExpandShapeOp::attachInterface<ExpandShapeOpTilingInterface>(*ctx);
        mlir::tensor::CollapseShapeOp::attachInterface<
            PartitionableLoopsInterfaceImpl<CollapseShapeOp>>(*ctx);
        mlir::tensor::ExpandShapeOp::attachInterface<
            PartitionableLoopsInterfaceImpl<ExpandShapeOp>>(*ctx);
    });
}
