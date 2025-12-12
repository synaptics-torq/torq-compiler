// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Dialect/Tensor/IR/Utils.h"
#include "torq/Utils/TorqHw.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include <cassert>
#include <cstdint>
#include <deque>
#include <mlir/Analysis/SliceAnalysis.h>
#include <optional>
#include <tuple>
#include <utility>

// TODO: all the TypeSwitchs should be extracted to TileAndFuseInterface

#define DEBUG_TYPE "torq-tile-and-fuse"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

enum class TileAndFuseProducersFuseMode { MaxSize, MaxProducers, OnlyPatterns, NoFuse };

static llvm::cl::opt<TileAndFuseProducersFuseMode> clTorqTileAndFuseProducersFuseMode(
    "torq-tile-and-fuse-producers-fuse-mode",
    llvm::cl::desc("Selects one of three predefined values"),
    llvm::cl::values(
        clEnumValN(
            TileAndFuseProducersFuseMode::MaxSize, "max-size",
            "Prefer bigger tile over fusing more producers: tile size is set to make the consumer "
            "fit in memory; producers are fused only if they fit in the same tile size)"
        ),
        clEnumValN(
            TileAndFuseProducersFuseMode::MaxProducers, "max-producers",
            "Prefer more producers over tile size: tile size is the biggest size that can still "
            "fit all the producers"
        ),
        clEnumValN(
            TileAndFuseProducersFuseMode::OnlyPatterns, "only-patterns",
            "Fuse producers only whene required to preserve patterns"
        ),
        clEnumValN(TileAndFuseProducersFuseMode::NoFuse, "no-fuse", "Do not fuse producers")
    ),
    llvm::cl::init(TileAndFuseProducersFuseMode::MaxSize) // Default value
);

class TileAndFusePass : public TileAndFuseBase<TileAndFusePass> {
  public:
    TileAndFusePass() = default;
    TileAndFusePass(const TileAndFusePass &pass) {}

    void runOnOperation() override;
};

static const std::string TORQ_TILING_DFS = "torq-tiling-dfs";
static const std::string TORQ_TILING_FUSED = "torq-tiling-fused";

torq_hl::Executor getExecutor(Operation *op) {
    return isMarkedFuseGroup(op) ? torq_hl::Executor::Slice : torq_hl::Executor::CSS;
}

void forwardDfs(Operation *firstOp, SmallVector<Operation *> &orderTi) {
    // auto boolAttrTrue = BoolAttr::get(firstOp->getContext(), true);
    auto boolAttrFalse = BoolAttr::get(firstOp->getContext(), false);

    std::deque<std::pair<Operation *, ResultRange::user_iterator>> stack;

    if (firstOp->getAttrOfType<BoolAttr>(TORQ_TILING_DFS) ||
        !isa_and_nonnull<TilingInterface>(firstOp))
        return;

    firstOp->setAttr(TORQ_TILING_DFS, boolAttrFalse);
    stack.push_back(std::make_pair(firstOp, firstOp->getResults().user_begin()));

    while (!stack.empty()) {
        auto &[op, user] = stack.back();

        if (user == op->getResults().user_end()) {
            // Close op
            stack.pop_back();
            orderTi.push_back(op);
            continue;
        }

        auto userOp = *user++;
        if (userOp->getAttrOfType<BoolAttr>(TORQ_TILING_DFS) ||
            !isa_and_nonnull<TilingInterface>(userOp)) {
            // userOp was already opened (and possibly closed).
            continue;
        }

        // Open userOp
        userOp->setAttr(TORQ_TILING_DFS, boolAttrFalse);
        stack.push_back(std::make_pair(userOp, userOp->getResults().user_begin()));
    }
}

void orderTiOps(FunctionOpInterface &funcOp, SmallVector<Operation *> &orderTi) {
    for (auto &block : funcOp.getBlocks()) {
        for (auto &op : block.getOperations()) {
            forwardDfs(&op, orderTi);
        }
    }
}

SmallVector<int64_t> getTilingDimOrder(TilingInterface tilingInterfaceOp) {
    auto loopIteratorTypes = tilingInterfaceOp.getLoopIteratorTypes();

    // Handler that use all dimensions for tiling (as needed), in that order.
    auto allParallelDims = [&]() {
        SmallVector<int64_t> tilingDimOrder;
        for (size_t dim = 0; dim < loopIteratorTypes.size(); ++dim) {
            if (loopIteratorTypes[dim] == utils::IteratorType::parallel)
                tilingDimOrder.push_back(dim);
        }
        return tilingDimOrder;
    };

    // Handler for nhwc operations: we tile C, followed by H.
    auto nhwcOpHandler = [&]() {
        SmallVector<int64_t> tilingDimOrder;

        assert(
            loopIteratorTypes[3] == utils::IteratorType::parallel &&
            "expected dimension 3 to be parallel"
        );
        tilingDimOrder.push_back(3); // C

        assert(
            loopIteratorTypes[1] == utils::IteratorType::parallel &&
            "expected dimension 1 to be parallel"
        );
        tilingDimOrder.push_back(1); // H

        return tilingDimOrder;
    };

    // Decide which dimensions to tile, and in what order, based on the executor and the operation.
    switch (getExecutor(tilingInterfaceOp)) {
    case torq_hl::Executor::Slice: {
        Operation *principalOp = getFuseGroupPrincipalOpBackward(tilingInterfaceOp);
        assert(principalOp != nullptr && "could not find the principal op of the fuse group");

        return TypeSwitch<Operation *, SmallVector<int64_t>>(principalOp)
            .Case<
                linalg::Conv2DNhwcHwcfOp, // NB: 4th iteration domain is actually F
                                          // (filters/output-channels) in this case
                linalg::DepthwiseConv2DNhwcHwcOp, linalg::PoolingNhwcMaxOp,
                linalg::PoolingNhwcMaxUnsignedOp, linalg::PoolingNhwcMinOp,
                linalg::PoolingNhwcMinUnsignedOp, linalg::PoolingNhwcSumOp>([&](auto) {
                return nhwcOpHandler();
            })
            .Default([&](auto) { return allParallelDims(); });
    }
    case torq_hl::Executor::CSS:
        return allParallelDims();

    default:
        llvm_unreachable("expected NSS or CSS executor");
    }
}

int64_t bytesOfSlice(Type type, ArrayRef<OpFoldResult> resultSizes, Attribute zero) {
    if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
        int64_t bytes = div_ceil(rankedType.getElementTypeBitWidth(), 8);
        for (auto sizeFoldResult : resultSizes) {
            auto constSize = getConstantIntValue(sizeFoldResult);
            if (!constSize) {
                assert(sizeFoldResult.is<Value>() && "expected Value");
                auto sizeValue = sizeFoldResult.get<Value>();

                constSize =
                    llvm::TypeSwitch<Operation *, std::optional<int64_t>>(sizeValue.getDefiningOp())
                        .Case<affine::AffineMinOp>([&](auto minOp) {
                            // Evaluate the size at 0,0,...
                            // This is a bit of a hack, what we really want is a tight upper bound
                            // of size.
                            SmallVector<Attribute> dims(minOp.getDimOperands().size(), zero);
                            return getConstantIntValue(minOp.fold({dims}));
                        })
                        // TODO: implement for other operations as needed.
                        .Default([](auto) {
                            assert(false && "unexpected operation");
                            return std::nullopt;
                        });
                assert(constSize && "failed to fold slice size to a constant");
            }
            bytes *= *constSize;
        }
        return bytes;
    }

    return div_ceil(type.getIntOrFloatBitWidth(), 8);
}

// This iteration domain is for all the loops of the op.
typedef std::tuple<
    Operation *, SmallVector<OpFoldResult> /* offsets */, SmallVector<OpFoldResult> /* sizes */>
    OpIterationDomain;
// This iteration domain is only for the result loops (the owner op might have more loops).
typedef std::tuple<
    OpResult, SmallVector<OpFoldResult> /* offsets */, SmallVector<OpFoldResult> /* sizes */>
    OpResultIterationDomain;

llvm::FailureOr<SmallVector<OpResultIterationDomain>> linalgOperandSlicesFromIterDomain(
    IRRewriter &rewriter, linalg::LinalgOp linalgOp, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes
) {
    auto allSliceParameter = linalg::computeAllSliceParameters(
        rewriter, linalgOp->getLoc(), linalgOp, linalgOp->getOperands(), offsets, sizes, {}, true
    );

    SmallVector<OpResultIterationDomain> oprandIterDomains;
    oprandIterDomains.reserve(allSliceParameter.size());
    for (auto [operand, sliceParams] : llvm::zip(linalgOp->getOperands(), allSliceParameter)) {

        OpResult operandOpResult = dyn_cast<OpResult>(operand);

        if (sliceParams) {
            oprandIterDomains.push_back(
                std::make_tuple(operandOpResult, sliceParams->offsets, sliceParams->sizes)
            );
            continue;
        }

        if (RankedTensorType rankedType = dyn_cast<RankedTensorType>(operandOpResult.getType())) {
            SmallVector<OpFoldResult> sizes, offsets;
            sizes.reserve(rankedType.getShape().size());
            offsets.reserve(rankedType.getShape().size());
            for (auto size : rankedType.getShape()) {
                sizes.push_back(rewriter.getIndexAttr(size));
                offsets.push_back(rewriter.getIndexAttr(0));
            }
            oprandIterDomains.push_back(std::make_tuple(operandOpResult, offsets, sizes));
            continue;
        }

        oprandIterDomains.push_back(std::make_tuple(
            operandOpResult, SmallVector<OpFoldResult>{rewriter.getIndexAttr(0)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)}
        ));
    }

    return oprandIterDomains;
}

llvm::FailureOr<SmallVector<OpResultIterationDomain>>
tensorCollapseShapeOperandSlicesFromIterDomain(
    IRRewriter &rewriter, mlir::tensor::CollapseShapeOp collapseOp, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes
) {
    std::optional<linalg::SliceParameters> sliceParams =
        tensor::computeCollapseSliceParameters(rewriter, collapseOp, offsets, sizes, {}, true);

    assert(sliceParams.has_value());

    SmallVector<OpResultIterationDomain> oprandIterDomains;
    auto operand = collapseOp.getOperand();
    OpResult operandOpResult = dyn_cast<OpResult>(operand);

    oprandIterDomains.push_back(
        std::make_tuple(operandOpResult, sliceParams->offsets, sliceParams->sizes)
    );

    return oprandIterDomains;
}

llvm::FailureOr<SmallVector<OpResultIterationDomain>> tensorExpandShapeOperandSlicesFromIterDomain(
    IRRewriter &rewriter, mlir::tensor::ExpandShapeOp expandOp, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes
) {
    std::optional<linalg::SliceParameters> sliceParams =
        tensor::computeExpandSliceParameters(rewriter, expandOp, offsets, sizes, {}, true);

    assert(sliceParams.has_value());

    SmallVector<OpResultIterationDomain> oprandIterDomains;
    auto operand = expandOp.getOperand(0);
    OpResult operandOpResult = dyn_cast<OpResult>(operand);

    oprandIterDomains.push_back(
        std::make_tuple(operandOpResult, sliceParams->offsets, sliceParams->sizes)
    );

    return oprandIterDomains;
}

llvm::FailureOr<SmallVector<OpResultIterationDomain>> tensorPadOperandSlicesFromIterDomain(
    IRRewriter &rewriter, mlir::tensor::PadOp padOp, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes
) {
    auto operand = padOp.getSource();
    OpResult operandOpResult = dyn_cast<OpResult>(operand);

    // TODO: I think the following is a safe over approximation, maybe do
    // something more precise.
    return SmallVector<OpResultIterationDomain>{std::make_tuple(
        operandOpResult, SmallVector<OpFoldResult>(offsets), SmallVector<OpFoldResult>(sizes)
    )};
}

llvm::FailureOr<SmallVector<OpResultIterationDomain>> tensorInsertSliceOperandSlicesFromIterDomain(
    IRRewriter &rewriter, mlir::tensor::InsertSliceOp insertSliceOp, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes
) {
    SmallVector<OpResultIterationDomain> operandIterDomains;

    // Handle source operand (the tensor being inserted)
    auto source = insertSliceOp.getSource();
    if (auto sourceOpResult = dyn_cast<OpResult>(source)) {
        auto sourceType = cast<RankedTensorType>(sourceOpResult.getType());
        SmallVector<OpFoldResult> sourceOffsets, sourceSizes;
        for (auto size : sourceType.getShape()) {
            sourceOffsets.push_back(rewriter.getIndexAttr(0));
            sourceSizes.push_back(rewriter.getIndexAttr(size));
        }
        operandIterDomains.push_back(std::make_tuple(sourceOpResult, sourceOffsets, sourceSizes));
    }

    // Handle destination operand (the tensor being inserted into)
    auto dest = insertSliceOp.getDest();
    if (auto destOpResult = dyn_cast<OpResult>(dest)) {
        // Use the full destination tensor size as a safe over-approximation
        auto destType = cast<RankedTensorType>(destOpResult.getType());
        SmallVector<OpFoldResult> destOffsets, destSizes;
        for (auto size : destType.getShape()) {
            destOffsets.push_back(rewriter.getIndexAttr(0));
            destSizes.push_back(rewriter.getIndexAttr(size));
        }
        operandIterDomains.push_back(std::make_tuple(destOpResult, destOffsets, destSizes));
    }

    return operandIterDomains;
}

// Return true iff the tile (including consumerOp and producerOps) can fit in availableMemoryBytes.
// When consumerOp is a member of a fuse group, it must be the output operation of that group. In
// that case, the entire group will be checked, even if the other members are not in producerOps.
// This gives the correct result when the function is called to check if the group needs to be
// tiled, before it was tiled.
// TODO: cache all the iteration domains so they don't need to be reallocated every time.
llvm::FailureOr<bool> checkTileFitsInMemory(
    IRRewriter &rewriter, Operation *consumerOp, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, const SetVector<Operation *> &producerOps,
    int64_t availableMemoryBytes
) {
    // How it works: we start from consumerOp, we calculate for each operand the slice needed for
    // the tile, and sum up the bytes for all the slices. If the sum is greater than
    // availableMemoryBytes, we return false; otherwise, we compute the iteration domain for each
    // operand based on its slice (it might have more loops), and add them to the queue. We process
    // elements in the queue in the same way until it is empty. For members of fuse groups, we don't
    // check the sum of their immediate operands, instead we check the sum of operands that are not
    // in the same group. We also don't count the init operands for fuse-group memebers, except for
    // the init of the group's ouput operator (this seems to match what the rewrite patterns do).

    // FIXME: we currently check that each operation/fuse-group can fit in availableMemoryBytes.
    // This does not take into account that while computing an operand, other operands already take
    // space.

    Attribute zero = getAsIndexOpFoldResult(rewriter.getContext(), 1).get<Attribute>();

    // Accumulate for each fuse-group the size of operands external to the group.
    llvm::DenseMap<IntegerAttr, int64_t> fusedGroupsBytes;

    std::deque<OpIterationDomain> queue;
    queue.push_back(std::make_tuple(
        consumerOp, SmallVector<OpFoldResult>(offsets), SmallVector<OpFoldResult>(sizes)
    ));
    while (!queue.empty()) {
        // Can't use structured bindings because we later capture some of them in a lambda (C++20
        // supports it):
        // auto [op, offsets, sizes] = queue.front();
        // so Instead we do this:
        auto op = std::get<0>(queue.front());
        auto offsets = std::get<1>(queue.front());
        auto sizes = std::get<2>(queue.front());
        queue.pop_front();

        // Compute operand slices
        llvm::FailureOr<SmallVector<OpResultIterationDomain>> operandSlices =
            TypeSwitch<Operation *, llvm::FailureOr<SmallVector<OpResultIterationDomain>>>(op)
                .Case<linalg::LinalgOp>([&](auto linalgOp) {
                    return linalgOperandSlicesFromIterDomain(rewriter, linalgOp, offsets, sizes);
                })
                .Case<mlir::tensor::PadOp>([&](auto padOp) {
                    return tensorPadOperandSlicesFromIterDomain(rewriter, padOp, offsets, sizes);
                })
                .Case<mlir::tensor::InsertSliceOp>([&](auto insertSliceOp) {
                    return tensorInsertSliceOperandSlicesFromIterDomain(
                        rewriter, insertSliceOp, offsets, sizes
                    );
                })
                .Case<mlir::tensor::CollapseShapeOp>([&](auto collapseOp) {
                    return tensorCollapseShapeOperandSlicesFromIterDomain(
                        rewriter, collapseOp, offsets, sizes
                    );
                })
                .Case<mlir::tensor::ExpandShapeOp>([&](auto expandOp) {
                    return tensorExpandShapeOperandSlicesFromIterDomain(
                        rewriter, expandOp, offsets, sizes
                    );
                })
                .Default([&](auto) {
                    LLVM_DEBUG({ llvm::dbgs() << "unknown op type: " << op->getName() << "\n"; });
                    return LogicalResult::failure();
                });

        if (failed(operandSlices)) {
            op->emitError("failed to compute operand slices");
            return LogicalResult::failure();
        }

        // Accumulate memory used by operand slices
        int64_t totalOpBytes = 0;

        // For each operand slice, compute its memory usage, and add the owner to the queue as
        // needed.
        for (auto [opResult, resultOffsets, resultSizes] : *operandSlices) {
            Operation *resultOp = opResult.getOwner();

            // Compute memory usage
            int64_t operandBytes = bytesOfSlice(opResult.getType(), resultSizes, zero);
            if ((totalOpBytes += operandBytes) > availableMemoryBytes && !isMarkedFuseGroup(op)) {
                return false;
            }

            bool shareFuseGroup = false;

            // If op is in a fuse group, check if this is an external operand, and add it to the
            // group's memory.
            if (auto fuseGroupAttr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP)) {
                auto operandFuseGroupAttr = resultOp->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);

                for (auto intAttr : fuseGroupAttr.getAsRange<IntegerAttr>()) {
                    if (operandFuseGroupAttr && llvm::is_contained(operandFuseGroupAttr, intAttr)) {
                        shareFuseGroup = true;
                        continue;
                    }

                    // Don't count tensor::EmptyOp feeding the output operand, except for the output
                    // op of the group.
                    auto opDsoi = dyn_cast<DestinationStyleOpInterface>(op);
                    if (isa<mlir::tensor::EmptyOp>(resultOp) && !isFuseGroupOutput(op) && opDsoi) {
                        auto inits = opDsoi.getDpsInits();
                        if (llvm::is_contained(inits, opResult)) {
                            continue;
                        }
                    }

                    // If opResult feeds the first input of the principal op,
                    // and that op has stride > 1, we need to double the memory,
                    // except if opResult comes from conv2d/dw/add.
                    auto principalOperands =
                        getFuseGroupPrincipalOpOperandsForward(intAttr, opResult);
                    if (llvm::any_of(principalOperands, [](OpOperand *principalOperand) {
                            if (principalOperand->getOperandNumber() == 0) {
                                auto strides =
                                    TypeSwitch<Operation *, SmallVector<int64_t>>(
                                        principalOperand->getOwner()
                                    )
                                        .Case<linalg::Conv2DNhwcHwcfOp>([](auto convOp) {
                                            return convOp.getStrides().template getValues<int64_t>(
                                            );
                                        })
                                        .Case<linalg::DepthwiseConv2DNhwcHwcOp>([](auto convOp) {
                                            return convOp.getStrides().template getValues<int64_t>(
                                            );
                                        })
                                        .Case<
                                            linalg::PoolingNhwcMaxOp,
                                            linalg::PoolingNhwcMaxUnsignedOp,
                                            linalg::PoolingNhwcMinOp,
                                            linalg::PoolingNhwcMinUnsignedOp,
                                            linalg::PoolingNhwcSumOp>([](auto convOp) {
                                            return convOp.getStrides().template getValues<int64_t>(
                                            );
                                        })
                                        .Default([&](auto) -> SmallVector<int64_t> { return {}; });
                                return llvm::any_of(strides, [](auto s) { return s > 1; });
                            }
                            return false;
                        })) {

                        if (operandFuseGroupAttr) {
                            Operation *sourcePrincipal = getFuseGroupPrincipalOpBackward(resultOp);
                            assert(
                                sourcePrincipal != nullptr &&
                                "could not find the principal op of the fuse group"
                            );
                            if (!isa<
                                    linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp,
                                    linalg::AddOp>(sourcePrincipal)) {
                                operandBytes *= 2;
                            }
                        }
                        else {
                            operandBytes *= 2;
                        }
                    }

                    if ((fusedGroupsBytes[intAttr] += operandBytes) > availableMemoryBytes) {
                        return false;
                    }
                }
            }

            // Add operand to the queue as needed
            // Skip tensor::InsertSliceOp as it's a data movement operation that doesn't need tiling
            if ((producerOps.contains(resultOp) || shareFuseGroup) &&
                !isa<mlir::tensor::InsertSliceOp>(resultOp)) {
                auto tiOp = cast<TilingInterface>(resultOp);

                // Get the iteration domain for all the loops of the operand owner
                SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
                if (failed(tiOp.getIterationDomainTileFromResultTile(
                        rewriter, opResult.getResultNumber(), resultOffsets, resultSizes,
                        mappedOffsets, mappedSizes
                    ))) {

                    tiOp->emitError("getIterationDomainTileFromResultTile failed");
                    return LogicalResult::failure();
                }

                queue.push_back(std::make_tuple(resultOp, mappedOffsets, mappedSizes));
            }
        }
    }

    return true;
}

// C++20 std::midpoint: computes average without overflow
inline int64_t midpoint(int64_t min, int64_t max) { return min + ((max - min) / 2); }

// Return true iff sizes was changed (made smaller).
// If the tile is not big enough, use binary search to find the biggest tile that fits in
// availableMemoryBytes.
llvm::FailureOr<bool> fitTileToMemory(
    IRRewriter &rewriter, Operation *consumerOp, const SetVector<Operation *> &producerOps,
    ArrayRef<int64_t> tilingDimOrder, size_t availableMemoryBytes, ArrayRef<int64_t> originalSizes,
    MutableArrayRef<OpFoldResult> offsets, MutableArrayRef<OpFoldResult> sizes
) {
    // First establish that the original tile is not big enough.
    auto tileFits = checkTileFitsInMemory(
        rewriter, consumerOp, offsets, sizes, producerOps, availableMemoryBytes
    );
    if (failed(tileFits)) {
        return LogicalResult::failure();
    }
    if (*tileFits) {
        // ops fit in memory, no need to change the tile
        return false;
    }

    auto one = getAsIndexOpFoldResult(rewriter.getContext(), 1);

    // For each dimension, find a factor such that originalSizes[dim]/factor fits in memory, or set
    // sizes[dim] to 1 and move to the next dimension. We first establish bounds for factor, and
    // then do a binary search to find the optimal factor.
    for (auto dim : tilingDimOrder) {
        int64_t size = *getConstantIntValue(sizes[dim]);
        if (size == 1) {
            // This dimension can't be split anymore, move to the next dimension.
            continue;
        }

        size_t minFactor = originalSizes[dim] / size;
        // NB: `size` is too big and `size <= div_ceil(originalSizes[dim], minFactor)`

        sizes[dim] = one;
        tileFits = checkTileFitsInMemory(
            rewriter, consumerOp, offsets, sizes, producerOps, availableMemoryBytes
        );
        if (failed(tileFits)) {
            return LogicalResult::failure();
        }
        if (!*tileFits) {
            // the tile does not fit even when sizes[dim] is set to 1, so keep it set to 1 and move
            // to the next dimension.
            continue;
        }

        int64_t maxFactor = originalSizes[dim];
        // NB: 1 fits (maybe smaller than needed) and `1 == div_ceil(originalSizes[dim], maxFactor)`

        // Invariant: consumerOp tiled to sizes (and fused with producerOps) fits in
        // availableMemoryBytes, when sizes[dim] is set to div_ceil(originalSizes[dim], maxFactor),
        // and does not fit when it is set to div_ceil(originalSizes[dim], minFactor).

        // While keeping the invariant, decrease maxFactor, and increase minFactor, until they are
        // consecutive, by splitting the difference in each step.
        while (maxFactor != minFactor + 1) {
            int64_t midFactor = midpoint(minFactor, maxFactor);
            // NB: minFactor < midFactor < maxFactor
            sizes[dim] = rewriter.getIndexAttr(div_ceil(originalSizes[dim], midFactor));

            tileFits = checkTileFitsInMemory(
                rewriter, consumerOp, offsets, sizes, producerOps, availableMemoryBytes
            );
            if (failed(tileFits)) {
                return LogicalResult::failure();
            }

            if (*tileFits) {
                maxFactor = midFactor;
            }
            else {
                minFactor = midFactor;
            }
        }
        sizes[dim] = rewriter.getIndexAttr(div_ceil(originalSizes[dim], maxFactor));

        return true;
    }

    consumerOp->emitWarning("operation can't be tiled: no more dimensions to tile");

    return LogicalResult::failure();
}

void replaceTiledOp(
    IRRewriter &rewriter, Operation *op, const scf::SCFTileAndFuseResult &tiledResults
) {
    for (OpResult res : op->getResults()) {
        if (auto replacement = tiledResults.replacements.lookup(res)) {
            rewriter.replaceAllUsesWith(res, replacement);
        }
    }
}

void applyTiledResults(
    IRRewriter &rewriter, Operation *op, scf::SCFTileAndFuseResult &tiledResults
) {
    // Replace the root with its tiled result
    replaceTiledOp(rewriter, op, tiledResults);

    // Mark root so we don't tile it's producers (as roots)
    op->setAttr(TORQ_TILING_FUSED, rewriter.getBoolAttr(true));

    for (auto prodOp : tiledResults.fusedProducers) {
        // In general, fused producers can declare that they want to be
        // yielded. Here we replace the original producers with the yielded
        // result. In practice, we use the default fusion options, so no
        // producers are yielded.
        replaceTiledOp(rewriter, prodOp, tiledResults);

        // Mark producers so we don't tile their producers (as roots)
        prodOp->setAttr(TORQ_TILING_FUSED, rewriter.getBoolAttr(true));
    }
}

std::tuple<bool, bool> fuseControlMaxSize(
    IRRewriter &rewriter, int64_t availableMemoryBytes,
    mlir::tensor::ExtractSliceOp candidateSliceOp, OpResult producerOpResult,
    bool isDestinationOperand
) {
    const auto doNotFuse = std::make_tuple(false, false);

    Operation *producerOp = producerOpResult.getOwner();

    auto producerTi = dyn_cast<TilingInterface>(producerOp);
    if (!producerTi) {
        // Producer has no TilingInterface
        return doNotFuse;
    }

    // TODO: if producer has users outside of the tiling fuse group, maybe we should yield it?
    bool yieldProducerReplacement = false;

    // FIXME: should we not fuse destination operands?
    // if (isDestinationOperand) {
    //     return doNotFuse;
    // }

    SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
    if (failed(producerTi.getIterationDomainTileFromResultTile(
            rewriter, producerOpResult.getResultNumber(), candidateSliceOp.getMixedOffsets(),
            candidateSliceOp.getMixedSizes(), mappedOffsets, mappedSizes
        ))) {
        return doNotFuse;
    }

    if (!isMarkedFuseGroup(producerOp)) {
        // Not part of a pattern-fuse-group; there are no restrictions on the dimensions.

        auto producerFits = checkTileFitsInMemory(
            rewriter, producerOp, mappedOffsets, mappedSizes, {}, availableMemoryBytes
        );
        if (failed(producerFits)) {
            return doNotFuse;
        }

        // Fuse iff producer fits in the tile
        return std::make_tuple(*producerFits, yieldProducerReplacement);
    }

    if (!isFuseGroupOutput(producerOp)) {
        // Is part of a pattern-fuse-group, but not the output operation (bottom most).
        // We only check the output operation of such group.
        return std::make_tuple(true, yieldProducerReplacement);
    }

    // The dimensions the producer can be tiled over
    auto dimOrder = getTilingDimOrder(producerTi);
    llvm::SmallSetVector<int64_t, 4> tilingDims{dimOrder.begin(), dimOrder.end()};

    auto [iterOffsets, iterSizes, iterStrides] =
        getOffsetsSizesAndStrides(producerTi.getIterationDomain(rewriter));

    assert(
        mappedSizes.size() == iterSizes.size() && "expected mapped and iter sizes to be the same"
    );

    // Don't fuse if we are tiling a dimension the producer can not be tiled over
    for (size_t index = 0; index < iterSizes.size(); ++index) {
        if (mappedSizes[index] != iterSizes[index] && !tilingDims.contains(index)) {
            return doNotFuse;
        }
    }

    auto producerFuseGroupFits = checkTileFitsInMemory(
        rewriter, producerOp, mappedOffsets, mappedSizes, {}, availableMemoryBytes
    );
    if (failed(producerFuseGroupFits)) {
        return doNotFuse;
    }

    // Fuse iff producer fits in the tile
    return std::make_tuple(*producerFuseGroupFits, yieldProducerReplacement);
}

std::tuple<bool, bool> fuseControlMaxProducers(
    IRRewriter &rewriter, mlir::tensor::ExtractSliceOp candidateSliceOp, OpResult producerOpResult,
    bool isDestinationOperand
) {
    const auto doNotFuse = std::make_tuple(false, false);

    Operation *producerOp = producerOpResult.getOwner();

    auto producerTi = dyn_cast<TilingInterface>(producerOp);
    if (!producerTi) {
        // Producer has no TilingInterface
        return doNotFuse;
    }

    // TODO: if producer has users outside of the tiling fuse group, maybe we should yield it?
    bool yieldProducerReplacement = false;

    // FIXME: should we not fuse destination operands?
    // if (isDestinationOperand) {
    //     return doNotFuse;
    // }

    // From here on we check if the producer can be tiled on the dimensions that
    // are being tiled. The first element in the return tuple indicates if the
    // producer should be fused (true) or not (false).

    if (!isMarkedFuseGroup(producerOp)) {
        // Not part of a pattern-fuse-group; there are no restrictions on the dimensions.
        return std::make_tuple(true, yieldProducerReplacement);
    }

    if (!isFuseGroupOutput(producerOp)) {
        // Is part of a pattern-fuse-group, but not the output operation (bottom most).
        // We only check the output operation of such group.
        return std::make_tuple(true, yieldProducerReplacement);
    }

    // The dimensions the producer can be tiled over
    auto dimOrder = getTilingDimOrder(producerTi);
    llvm::SmallSetVector<int64_t, 4> tilingDims{dimOrder.begin(), dimOrder.end()};

    SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
    if (failed(producerTi.getIterationDomainTileFromResultTile(
            rewriter, producerOpResult.getResultNumber(), candidateSliceOp.getMixedOffsets(),
            candidateSliceOp.getMixedSizes(), mappedOffsets, mappedSizes
        ))) {
        return doNotFuse;
    }

    auto [iterOffsets, iterSizes, iterStrides] =
        getOffsetsSizesAndStrides(producerTi.getIterationDomain(rewriter));

    assert(
        mappedSizes.size() == iterSizes.size() && "expected mapped and iter sizes to be the same"
    );

    // Don't fuse if we are tiling a dimension the producer can not be tiled over
    for (size_t index = 0; index < iterSizes.size(); ++index) {
        if (mappedSizes[index] != iterSizes[index] && !tilingDims.contains(index)) {
            return doNotFuse;
        }
    }

    // Producer can be fused
    return std::make_tuple(true, yieldProducerReplacement);
}

FailureOr<scf::SCFTileAndFuseResult> tileAndFuseToSize(
    IRRewriter &rewriter, TilingInterface tilingInterfaceOp, int64_t availableMemoryBytes,
    llvm::ArrayRef<OpFoldResult> completeIterSizes, ArrayRef<OpFoldResult> tileIterSizes
) {
    // set dimensions that are not being tiled to 0 (and convert to int64_t).
    SmallVector<int64_t> tileSizes(completeIterSizes.size(), 0);
    for (size_t i = 0; i < completeIterSizes.size(); ++i) {
        auto iterSize = getConstantIntValue(completeIterSizes[i]);
        auto tileSize = getConstantIntValue(tileIterSizes[i]);
        if (*tileSize != *iterSize) {
            tileSizes[i] = *tileSize;
        }
    }

    LLVM_DEBUG({
        llvm::dbgs() << "tile sizes: ";
        bool isFirst = true;
        for (auto d : tileSizes) {
            llvm::dbgs() << (isFirst ? "" : "x") << d;
            isFirst = false;
        }
        llvm::dbgs() << "\n";
    });

    auto maybePatternFuseGroup = isFuseGroupOutput(tilingInterfaceOp);

    scf::SCFTileAndFuseOptions options{};
    // Consider using setSCFTileSizes from iree/compiler/Codegen/Utils/Utils.h
    options.tilingOptions.setTileSizes(getAsIndexOpFoldResult(rewriter.getContext(), tileSizes));
    options.setFusionControlFn([&](mlir::tensor::ExtractSliceOp candidateSliceOp,
                                   OpResult producerOpResult, bool isDestinationOperand) {
        switch (clTorqTileAndFuseProducersFuseMode.getValue()) {
        case TileAndFuseProducersFuseMode::MaxSize:
            return fuseControlMaxSize(
                rewriter, availableMemoryBytes, candidateSliceOp, producerOpResult,
                isDestinationOperand
            );

        case TileAndFuseProducersFuseMode::MaxProducers:
            return fuseControlMaxProducers(
                rewriter, candidateSliceOp, producerOpResult, isDestinationOperand
            );

        case TileAndFuseProducersFuseMode::OnlyPatterns: {
            bool shouldFuse =
                maybePatternFuseGroup && isMarkedFuseGroup(producerOpResult.getOwner());
            return std::make_tuple(shouldFuse, false);
        }

        case TileAndFuseProducersFuseMode::NoFuse:
            return std::make_tuple(false, false);
        }
    });

    return scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp, options);
}

void tileAndFuse(MLIRContext *context, Operation *op) {
    // Check if it was already fused as a producer
    if (op->getAttrOfType<BoolAttr>(TORQ_TILING_FUSED)) {
        return;
    }

    // For fuse-groups, we only tile from the bottom most op
    if (isMarkedFuseGroup(op) && !isFuseGroupOutput(op)) {
        return;
    }

    IRRewriter rewriter(context);

    int64_t availableMemoryBytes = TorqHw::get().getAvailableMemoryForTiling();

    auto tilingInterfaceOp = cast<TilingInterface>(op);
    assert(tilingInterfaceOp && "operation not implementing TilingInterface");

    auto [iterOffsets, iterSizes, iterStrides] =
        getOffsetsSizesAndStrides(tilingInterfaceOp.getIterationDomain(rewriter));

    auto iterIntSizes = getConstantIntValues(iterSizes);
    if (!iterIntSizes) {
        tilingInterfaceOp.emitError("iteration domain sizes are not constants");
        return;
    }

    SmallVector<OpFoldResult> tileOffsets(iterOffsets), tileSizes(iterSizes);

    // Get the order in which we should tile the dimensions.
    SmallVector<int64_t> tilingDimOrder = getTilingDimOrder(tilingInterfaceOp);

    // Find a tile that can fit op, starting from tileSizes.
    auto tileChanged = fitTileToMemory(
        rewriter, op, {}, tilingDimOrder, availableMemoryBytes, *iterIntSizes, tileOffsets,
        tileSizes
    );
    if (failed(tileChanged)) {
        op->emitError("failed to find a tile size for op");
        return;
    }

    // If the original tileSizes fits, there's no need to tile.
    if (!*tileChanged) {
        return;
    }

    LLVM_DEBUG({
        llvm::dbgs() << op->getName() << " needs to be tiled\n";
        llvm::dbgs() << "  " << availableMemoryBytes << " bytes of available memory for tiling\n";

        if (auto groupId = op->getAttr(TORQ_FUSE_GROUP_ID)) {
            llvm::dbgs() << "  " << TORQ_FUSE_GROUP_ID << ": " << getConstantIntValue(groupId)
                         << "\n";
            if (isMarkedFuseGroup(op)) {
                llvm::dbgs() << "  principal op: " << getFuseGroupPrincipalOpBackward(op)->getName()
                             << "\n";
            }
        }

        llvm::dbgs() << "  iteration sizes: ";
        bool isFirst = true;
        for (auto size : iterSizes) {
            llvm::dbgs() << (isFirst ? "" : "x") << getConstantIntValue(size);
            isFirst = false;
        }
        llvm::dbgs() << "\n";

        llvm::dbgs() << "  iteration types: ";
        isFirst = true;
        for (auto iterType : tilingInterfaceOp.getLoopIteratorTypes()) {
            llvm::dbgs() << (isFirst ? "" : ", ") << iterType;
            isFirst = false;
        }
        llvm::dbgs() << "\n";
    });

    FailureOr<scf::SCFTileAndFuseResult> tiledResults =
        tileAndFuseToSize(rewriter, tilingInterfaceOp, availableMemoryBytes, iterSizes, tileSizes);
    if (failed(tiledResults)) {
        op->emitError("tile and fuse failed");
        return;
    }

    if (clTorqTileAndFuseProducersFuseMode.getValue() ==
        TileAndFuseProducersFuseMode::MaxProducers) {
        // Now that we know which producers were fused, find a tile that fits them too.
        tileChanged = fitTileToMemory(
            rewriter, op, tiledResults->fusedProducers, tilingDimOrder, availableMemoryBytes,
            *iterIntSizes, tileOffsets, tileSizes
        );
        if (failed(tileChanged)) {
            // REMOEV:
            llvm::dbgs() << "DEBUG: " << __FILE__ << ":" << __LINE__ << " fit after fuse failed\n";
            op->dump();
            assert(false);
            op->emitError("failed to find a tile size for producers");
            return;
        }

        if (*tileChanged) {
            // The second fitTileToMemory returned a smaller tile.

            tiledResults = tileAndFuseToSize(
                rewriter, tilingInterfaceOp, availableMemoryBytes, iterSizes, tileSizes
            );
            if (failed(tiledResults)) {
                op->emitError("tile and fuse failed");
                return;
            }
        }
    }

    LLVM_DEBUG({
        llvm::dbgs() << "  fused " << tiledResults->fusedProducers.size() << " producers\n";
    });

    applyTiledResults(rewriter, op, *tiledResults);
}

void TileAndFusePass::runOnOperation() {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    SmallVector<Operation *> orderTi;
    orderTiOps(funcOp, orderTi);

    for (auto *op : orderTi) {
        tileAndFuse(context, op);
    }

    LLVM_DEBUG({ llvm::dbgs() << "Tile and Fuse - DONE\n"; });
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTileAndFusePass() {
    return std::make_unique<TileAndFusePass>();
}

} // namespace mlir::syna::torq
