#include "torq/Dialect/Tensor/IR/Utils.h"

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::tensor;

namespace mlir::syna::torq::tensor {

std::optional<linalg::SliceParameters> computeCollapseSliceParameters(
    OpBuilder &builder, CollapseShapeOp collapseOp, ArrayRef<OpFoldResult> ivs,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<OpFoldResult> sizeBounds, bool omitPartialTileCheck
) {
    assert(
        ivs.size() == static_cast<size_t>(llvm::count_if(
                          llvm::make_range(tileSizes.begin(), tileSizes.end()),
                          [](OpFoldResult v) { return !isZeroIndex(v); }
                      )) &&
        "expected as many ivs as non-zero sizes"
    );

    Location loc = collapseOp->getLoc();

    SmallVector<OpFoldResult> lbs = linalg::computeTileOffsets(builder, loc, ivs, tileSizes);
    SmallVector<OpFoldResult> subShapeSizes =
        linalg::computeTileSizes(builder, loc, tileSizes, sizeBounds);

    RankedTensorType resultTensor = cast<RankedTensorType>(collapseOp.getResult().getType());

    Value valueToTile = collapseOp.getSrc();
    RankedTensorType srcTensorType = cast<RankedTensorType>(valueToTile.getType());

    // Construct an AffineMap from the dimensions of resultTensor to the indexes of valueToTile. A
    // result dimension d0, that was collapsed from a reassociation group with uncollapsed sizes s0,
    // s1, ..., sn, is mapped to (d0/(sn*...*s2*s1) mod s0, ..., d0/sn mod sn-1, d0 mod sn). Note
    // that since '0 <= d0 < sn*...*s1*s0', the 'mod s0' in the left most element is redundant.
    // clang-format off
    // For example, for:
    //   tensor.collapse_shape %1 [[0, 1, 2], [3]] : tensor<10x20x30x40xi8> into tensor<6000x40xi8>
    // we will construct:
    //   (d0, d1) -> ((d0 / 20*30) mod 10, (d0 / 30) mod 20, d0 mod 30, d1 mod 40)
    //                             ^^^^^^-redundant                        ^^^^^^-redundant
    // and with the observation that the left most mod is redundant:
    //   (d0, d1) -> (d0 / 20*30, (d0 / 30) mod 20, d0 mod 30, d1)
    // clang-format on
    SmallVector<AffineExpr, 4> srcMapExprs(srcTensorType.getRank());
    for (auto [groupIndex, reassocGroup] : llvm::enumerate(collapseOp.getReassociationIndices())) {
        int64_t sizeProd = 1;

        for (int i = reassocGroup.size() - 1; i >= 0; --i) {
            int64_t srcDimIdx = reassocGroup[i];
            int64_t srcDimSize = srcTensorType.getDimSize(srcDimIdx);

            srcMapExprs[srcDimIdx] = builder.getAffineDimExpr(groupIndex).floorDiv(sizeProd);
            if (i > 0) {
                // The mod in the last index (left most) is redundant. This
                // also eliminates the mod from singletons.
                srcMapExprs[srcDimIdx] = srcMapExprs[srcDimIdx] % srcDimSize;
            }
            sizeProd *= srcDimSize;
        }
    }
    AffineMap map = AffineMap::get(resultTensor.getRank(), 0, srcMapExprs, builder.getContext());

    return linalg::computeSliceParameters(
        builder, loc, valueToTile, tileSizes, map, lbs, sizeBounds, subShapeSizes,
        omitPartialTileCheck
    );
}

std::optional<linalg::SliceParameters> computeExpandSliceParameters(
    OpBuilder &builder, ExpandShapeOp expandOp, ArrayRef<OpFoldResult> ivs,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<OpFoldResult> sizeBounds, bool omitPartialTileCheck
) {
    assert(
        ivs.size() == static_cast<size_t>(llvm::count_if(
                          llvm::make_range(tileSizes.begin(), tileSizes.end()),
                          [](OpFoldResult v) { return !isZeroIndex(v); }
                      )) &&
        "expected as many ivs as non-zero sizes"
    );

    Location loc = expandOp->getLoc();

    SmallVector<OpFoldResult> lbs = linalg::computeTileOffsets(builder, loc, ivs, tileSizes);
    SmallVector<OpFoldResult> subShapeSizes =
        linalg::computeTileSizes(builder, loc, tileSizes, sizeBounds);

    Value valueToTile = expandOp.getSrc();

    RankedTensorType resultTensor = cast<RankedTensorType>(expandOp.getResult().getType());

    // Construct an AffineMap from the dimensions of resultTensor to the indexes of valueToTile. A
    // reassociation group [d0, d1, ..., dn] with sizes s0, s1, ..., sn, is mapped to
    // d0*s1*s2*..*sn +...+ dn-1*sn + dn
    // clang-format off
    // For example, for:
    //   tensor.expand_shape %1 [[0, 1, 2], [3]] output_shape [10, 20, 30, 40] : tensor<6000x40xi8> into tensor<10x20x30x40xi8>
    // we will construct:
    //   (d0, d1, d2, d3) -> (d0*20*30 + d1*30 + d2, d3)
    // clang-format on
    SmallVector<AffineExpr, 4> inputMapExprs;
    for (auto groupAttr : expandOp.getReassociation()) {
        ArrayAttr reassocGroup = cast<ArrayAttr>(groupAttr);

        AffineExpr expr = builder.getAffineConstantExpr(0);
        int64_t dimSize = 1;
        for (size_t i = reassocGroup.size(); i > 0; --i) {
            int64_t resultDimIdx = cast<IntegerAttr>(reassocGroup[i - 1]).getInt();
            AffineExpr resultDimExpr = builder.getAffineDimExpr(resultDimIdx);

            expr = expr + (resultDimExpr * dimSize);
            dimSize *= resultTensor.getDimSize(resultDimIdx);
        }
        inputMapExprs.push_back(expr);
    }
    AffineMap map = AffineMap::get(resultTensor.getRank(), 0, inputMapExprs, builder.getContext());

    return linalg::computeSliceParameters(
        builder, loc, valueToTile, tileSizes, map, lbs, sizeBounds, subShapeSizes,
        omitPartialTileCheck
    );
}

} // namespace mlir::syna::torq::tensor
