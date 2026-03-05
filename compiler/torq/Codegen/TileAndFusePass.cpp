// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Dialect/Tensor/IR/Utils.h"
#include "torq/Utils/TorqHw.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
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

int64_t computeMinMaxAtZero(Operation *op, Attribute zero) {
    bool isMin;
    AffineMap map;
    SmallVector<Value> operands;
    if (auto minOp = dyn_cast<affine::AffineMinOp>(op)) {
        isMin = true;
        map = minOp.getMap();
        operands = minOp.getOperands();
    }
    else if (auto maxOp = dyn_cast<affine::AffineMaxOp>(op)) {
        isMin = false;
        map = maxOp.getMap();
        operands = maxOp.getOperands();
    }
    else {
        assert(false && "expected min/max op");
    }

    affine::fullyComposeAffineMapAndOperands(&map, &operands);
    SmallVector<Attribute> zeros(map.getNumDims(), zero);
    SmallVector<Attribute> results;
    if (failed(map.constantFold(zeros, results)))
        assert(false && "failed to compute map when all inputs are zero");

    Attribute *result;
    if (isMin) {
        result = llvm::min_element(results, [](mlir::Attribute a, mlir::Attribute b) {
            return cast<mlir::IntegerAttr>(a).getInt() < cast<mlir::IntegerAttr>(b).getInt();
        });
    }
    else {
        result = llvm::max_element(results, [](mlir::Attribute a, mlir::Attribute b) {
            return cast<mlir::IntegerAttr>(a).getInt() < cast<mlir::IntegerAttr>(b).getInt();
        });
    }
    assert(result && "failed to find minimum");

    return cast<mlir::IntegerAttr>(*result).getInt();
}

int64_t bytesOfSlice(Type type, ArrayRef<OpFoldResult> resultSizes, Attribute zero) {
    auto rankedType = dyn_cast<RankedTensorType>(type);
    if (!rankedType) {
        return div_ceil(type.getIntOrFloatBitWidth(), 8);
    }

    int64_t elements = 1;
    for (auto sizeFoldResult : resultSizes) {
        std::optional<int64_t> constSize = getConstantIntValue(sizeFoldResult);
        if (constSize) {
            elements *= *constSize;
            continue;
        }

        assert(sizeFoldResult.is<Value>() && "expected Value");
        auto sizeValue = sizeFoldResult.get<Value>();

        constSize =
            llvm::TypeSwitch<Operation *, std::optional<int64_t>>(sizeValue.getDefiningOp())
                .Case<affine::AffineMinOp, affine::AffineMaxOp>([&](auto minMaxOp) {
                    // Evaluate the size at 0,0,...
                    // This is a bit of a hack, what we really want is a tight upper bound
                    // of size.
                    return computeMinMaxAtZero(minMaxOp, zero);
                })
                .Case<affine::AffineApplyOp>([&](auto applyOp) {
                    // Evaluate the size at 0,0,...
                    // This is a bit of a hack, what we really want is a tight upper bound
                    // of size.
                    AffineMap map = applyOp.getMap();
                    SmallVector<Value> operands = applyOp.getOperands();

                    affine::fullyComposeAffineMapAndOperands(&map, &operands);
                    SmallVector<Attribute> zeros(map.getNumDims(), zero);
                    SmallVector<Attribute> results;
                    if (failed(map.constantFold(zeros, results)))
                        assert(false && "failed to compute map when all inputs are zero");

                    assert(results.size() == 1 && "AffineApplyOp should have exactly one reault");

                    return cast<mlir::IntegerAttr>(results[0]).getInt();
                })
                // TODO: implement for other operations as needed.
                .Default([](auto) {
                    assert(false && "unexpected operation");
                    return std::nullopt;
                });
        assert(constSize && "failed to fold slice size to a constant");

        elements *= *constSize;
    }

    int64_t bytes = div_ceil(rankedType.getElementTypeBitWidth(), 8);

    return bytes * elements;
}

// This iteration domain is for all the loops of the op.
typedef std::tuple<
    Operation *, SmallVector<OpFoldResult> /* offsets */, SmallVector<OpFoldResult> /* sizes */>
    OpIterationDomain;
// This iteration domain is only for the result loops (the owner op might have more loops).
typedef std::tuple<
    Value, SmallVector<OpFoldResult> /* offsets */, SmallVector<OpFoldResult> /* sizes */>
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
        if (sliceParams) {
            oprandIterDomains.push_back(
                std::make_tuple(operand, sliceParams->offsets, sliceParams->sizes)
            );
            continue;
        }

        if (RankedTensorType rankedType = dyn_cast<RankedTensorType>(operand.getType())) {
            SmallVector<OpFoldResult> sizes, offsets;
            sizes.reserve(rankedType.getShape().size());
            offsets.reserve(rankedType.getShape().size());
            for (auto size : rankedType.getShape()) {
                sizes.push_back(rewriter.getIndexAttr(size));
                offsets.push_back(rewriter.getIndexAttr(0));
            }

            oprandIterDomains.push_back(std::make_tuple(operand, offsets, sizes));
            continue;
        }

        oprandIterDomains.push_back(std::make_tuple(
            operand, SmallVector<OpFoldResult>{rewriter.getIndexAttr(0)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)}
        ));
    }

    return oprandIterDomains;
}

llvm::FailureOr<SmallVector<OpResultIterationDomain>> softmaxOperandSlicesFromIterDomain(
    IRRewriter &rewriter, linalg::SoftmaxOp softmaxOp, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes
) {
    return SmallVector<OpResultIterationDomain>{
        std::make_tuple(
            softmaxOp->getOperand(0), SmallVector<OpFoldResult>(offsets),
            SmallVector<OpFoldResult>(sizes)
        ),
        std::make_tuple(
            softmaxOp->getOperand(1), SmallVector<OpFoldResult>(offsets),
            SmallVector<OpFoldResult>(sizes)
        )
    };
}

llvm::FailureOr<SmallVector<OpResultIterationDomain>>
tensorCollapseShapeOperandSlicesFromIterDomain(
    IRRewriter &rewriter, mlir::tensor::CollapseShapeOp collapseOp, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes
) {
    std::optional<linalg::SliceParameters> sliceParams =
        tensor::computeCollapseSliceParameters(rewriter, collapseOp, offsets, sizes, {}, true);

    assert(sliceParams.has_value());

    return SmallVector<OpResultIterationDomain>{
        std::make_tuple(collapseOp.getOperand(), sliceParams->offsets, sliceParams->sizes)
    };
}

llvm::FailureOr<SmallVector<OpResultIterationDomain>> tensorExpandShapeOperandSlicesFromIterDomain(
    IRRewriter &rewriter, mlir::tensor::ExpandShapeOp expandOp, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes
) {
    std::optional<linalg::SliceParameters> sliceParams =
        tensor::computeExpandSliceParameters(rewriter, expandOp, offsets, sizes, {}, true);

    assert(sliceParams.has_value());

    return SmallVector<OpResultIterationDomain>{
        std::make_tuple(expandOp.getOperand(0), sliceParams->offsets, sliceParams->sizes)
    };
}

llvm::FailureOr<SmallVector<OpResultIterationDomain>> tensorPadOperandSlicesFromIterDomain(
    IRRewriter &rewriter, mlir::tensor::PadOp padOp, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes
) {
    // TODO: I think the following is a safe over approximation, maybe do
    // something more precise.
    return SmallVector<OpResultIterationDomain>{std::make_tuple(
        padOp.getSource(), SmallVector<OpFoldResult>(offsets), SmallVector<OpFoldResult>(sizes)
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

llvm::FailureOr<SmallVector<OpResultIterationDomain>> operandSlicesFromIterDomain(
    IRRewriter &rewriter, Operation *op, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes
) {
    return TypeSwitch<Operation *, llvm::FailureOr<SmallVector<OpResultIterationDomain>>>(op)
        .Case<linalg::LinalgOp>([&](auto linalgOp) {
            return linalgOperandSlicesFromIterDomain(rewriter, linalgOp, offsets, sizes);
        })
        .Case<linalg::SoftmaxOp>([&](linalg::SoftmaxOp softmaxOp) {
            return softmaxOperandSlicesFromIterDomain(rewriter, softmaxOp, offsets, sizes);
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
            return tensorExpandShapeOperandSlicesFromIterDomain(rewriter, expandOp, offsets, sizes);
        })
        .Default([&](auto) {
            LLVM_DEBUG({ llvm::dbgs() << "unknown op type: " << op->getName() << "\n"; });
            assert(false);
            return LogicalResult::failure();
        });
}

bool isFirstOperandWithStride(OpOperand *operand) {
    if (operand->getOperandNumber() != 0) {
        return false;
    }

    auto strides =
        TypeSwitch<Operation *, SmallVector<int64_t>>(operand->getOwner())
            .Case<linalg::Conv2DNhwcHwcfOp>([](auto convOp) {
                return convOp.getStrides().template getValues<int64_t>();
            })
            .Case<linalg::DepthwiseConv2DNhwcHwcOp>([](auto convOp) {
                return convOp.getStrides().template getValues<int64_t>();
            })
            .Case<
                linalg::PoolingNchwMaxOp, linalg::PoolingNchwSumOp, linalg::PoolingNcwMaxOp,
                linalg::PoolingNcwSumOp, linalg::PoolingNhwcMaxOp, linalg::PoolingNhwcMaxUnsignedOp,
                linalg::PoolingNhwcMinOp, linalg::PoolingNhwcMinUnsignedOp,
                linalg::PoolingNhwcSumOp>([](auto convOp) {
                return convOp.getStrides().template getValues<int64_t>();
            })
            .Default([&](auto) -> SmallVector<int64_t> { return {}; });

    return llvm::any_of(strides, [](auto s) { return s > 1; });
}

void bytesOfFusedOpTile(
    Operation *op, ArrayRef<OpResultIterationDomain> operandSlices,
    llvm::DenseMap<IntegerAttr, int64_t> &fusedGroupsBytes,
    llvm::SetVector<Operation *> &shareFuseGroup, Attribute zero
) {
    for (auto [idx, slice] : llvm::enumerate(operandSlices)) {
        auto [resultVal, resultOffsets, resultSizes] = slice;
        Operation *resultOp = nullptr;
        ArrayAttr operandFuseGroupAttr;

        if (auto opResult = dyn_cast<OpResult>(resultVal)) {
            resultOp = opResult.getOwner();
            operandFuseGroupAttr = resultOp->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
        }

        int64_t operandBytes = bytesOfSlice(resultVal.getType(), resultSizes, zero);

        // Account for weight packing on matmul RHS (operand 1 = B in [A, B, init]):
        // kernel selection pads the N dimension to a multiple of parallel_outs.
        // TODO: this adjustment is skipped when the RHS owner is in the same fuse
        // group (e.g. a transpose), because operandBytes is not used for internal
        // operands. The packing overhead should be tracked at the group level.
        if (idx == 1 && isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
            auto outType = cast<RankedTensorType>(
                cast<DestinationStyleOpInterface>(op).getDpsInits()[0].getType()
            );
            int64_t packFactor = outType.getElementTypeBitWidth() <= 8 ? 64 : 32;
            if (auto nSize = getConstantIntValue(resultSizes.back()); nSize && *nSize > 0) {
                int64_t padded = div_ceil(*nSize, packFactor) * packFactor;
                operandBytes = (operandBytes / *nSize) * padded;
            }
        }

        auto fuseGroupAttr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
        assert(fuseGroupAttr);

        // Check if this is an external operand, and add it to the group's
        // memory.
        for (auto intAttr : fuseGroupAttr.getAsRange<IntegerAttr>()) {
            if (operandFuseGroupAttr && llvm::is_contained(operandFuseGroupAttr, intAttr)) {
                // Not an external operand
                shareFuseGroup.insert(resultOp);
                continue;
            }

            // Don't count tensor::EmptyOp feeding the output operand, except for the output
            // op of the group.
            auto opDsoi = dyn_cast<DestinationStyleOpInterface>(op);
            if (llvm::isa_and_nonnull<mlir::tensor::EmptyOp>(resultOp) && !isFuseGroupOutput(op) &&
                opDsoi) {
                auto inits = opDsoi.getDpsInits();
                if (llvm::is_contained(inits, resultVal)) {
                    continue;
                }
            }

            // If opResult feeds the first input of the principal op,
            // and that op has stride > 1, we need to double the memory,
            // except if opResult comes from conv2d/dw/pooling.
            auto principalOperands = getFuseGroupPrincipalOpOperandsForward(intAttr, resultVal);
            if (llvm::any_of(principalOperands, isFirstOperandWithStride)) {

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

            fusedGroupsBytes[intAttr] += operandBytes;
        }
    }
}

int64_t
bytesOfOpTile(Operation *op, ArrayRef<OpResultIterationDomain> operandSlices, Attribute zero) {
    return TypeSwitch<Operation *, int64_t>(op)
        .Case<linalg::SoftmaxOp>([&](auto softmaxOp) {
            // Memory requirements for SoftmaxOp: 2*input + output When
            // decomposing the softmax on NSS, most of the ops follow the same
            // input->output pattern, but two of them are torq_hl::select and
            // torq_hl::ElementwiseBinary which have 2 inputs and 1 output.
            // Until we convert these torq_hl ops to linalg, we have to tile the
            // whole softmax accordingly.

            int64_t totalBytes = 0;

            assert(operandSlices.size() == 2 && "linalg.softmax should have exactly two inputs");

            for (int i = 0; i < 2; ++i) {
                auto [opResult, resultOffsets, resultSizes] = operandSlices[i];
                int64_t bytes = bytesOfSlice(opResult.getType(), resultSizes, zero);
                totalBytes += bytes;
                if (i == 0) {
                    // 2*input
                    totalBytes += bytes;
                }
            }

            return totalBytes;
        })
        .Case<linalg::MatmulOp, linalg::BatchMatmulOp>([&](auto) {
            // Account for weight packing on RHS (operand 1 = B in [A, B, init]):
            // kernel selection pads the N dimension to a multiple of parallel_outs.
            auto outType = cast<RankedTensorType>(
                cast<DestinationStyleOpInterface>(op).getDpsInits()[0].getType()
            );
            int64_t packFactor = outType.getElementTypeBitWidth() <= 8 ? 64 : 32;

            int64_t totalBytes = 0;
            for (auto [idx, slice] : llvm::enumerate(operandSlices)) {
                auto [opResult, resultOffsets, resultSizes] = slice;
                int64_t bytes = bytesOfSlice(opResult.getType(), resultSizes, zero);
                if (idx == 1) {
                    if (auto nSize = getConstantIntValue(resultSizes.back()); nSize && *nSize > 0) {
                        int64_t padded = div_ceil(*nSize, packFactor) * packFactor;
                        bytes = (bytes / *nSize) * padded;
                    }
                }
                totalBytes += bytes;
            }
            return totalBytes;
        })
        .Default([&](auto) {
            int64_t totalBytes = 0;
            for (auto [opResult, resultOffsets, resultSizes] : operandSlices) {
                totalBytes += bytesOfSlice(opResult.getType(), resultSizes, zero);
            }
            return totalBytes;
        });
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

    Attribute zero = getAsIndexOpFoldResult(rewriter.getContext(), 0).get<Attribute>();

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
        Operation *op = std::get<0>(queue.front());
        SmallVector<OpFoldResult> offsets = std::get<1>(queue.front());
        SmallVector<OpFoldResult> sizes = std::get<2>(queue.front());
        queue.pop_front();

        // Compute operand slices
        llvm::FailureOr<SmallVector<OpResultIterationDomain>> operandSlices =
            operandSlicesFromIterDomain(rewriter, op, offsets, sizes);
        if (failed(operandSlices)) {
            op->emitError("failed to compute operand slices");
            return LogicalResult::failure();
        }

        SetVector<Operation *> shareFuseGroup = {};

        if (auto fuseGroupAttr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP)) {
            // Update fusedGroupsBytes
            bytesOfFusedOpTile(op, *operandSlices, fusedGroupsBytes, shareFuseGroup, zero);

            // Check if any fuse group exceeds the available memory
            for (auto intAttr : fuseGroupAttr.getAsRange<IntegerAttr>()) {
                if (fusedGroupsBytes[intAttr] > availableMemoryBytes) {
                    return false;
                }
            }
        }
        else {
            // Compute memory usage
            int64_t requiredBytes = bytesOfOpTile(op, *operandSlices, zero);
            if (requiredBytes > availableMemoryBytes) {
                return false;
            }
        }

        // For each operand slice, add the owner to the queue as needed.
        for (auto [resultVal, resultOffsets, resultSizes] : *operandSlices) {
            auto opResult = dyn_cast<OpResult>(resultVal);
            if (!opResult)
                continue;

            Operation *resultOp = opResult.getOwner();

            if (!producerOps.contains(resultOp) && !shareFuseGroup.contains(resultOp))
                continue;

            // Skip tensor::InsertSliceOp as it's a data movement operation that doesn't need tiling
            if (isa<mlir::tensor::InsertSliceOp>(resultOp))
                continue;

            auto tiOp = cast<TilingInterface>(resultOp);

            // Get the iteration domain for all the loops of the operand owner
            SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
            if (failed(tiOp.getIterationDomainTileFromResultTile(
                    rewriter, opResult.getResultNumber(), resultOffsets, resultSizes, mappedOffsets,
                    mappedSizes
                ))) {

                tiOp->emitError("getIterationDomainTileFromResultTile failed");
                return LogicalResult::failure();
            }

            queue.push_back(std::make_tuple(resultOp, mappedOffsets, mappedSizes));
        }
    }

    return true;
}

// C++20 std::midpoint: computes average without overflow
inline int64_t midpoint(int64_t min, int64_t max) { return min + ((max - min) / 2); }

// Binary-search for the largest tile size along a single dimension that fits in memory.
// Precondition: the tile fits when sizes[dim] = div_ceil(originalSize, maxFactor),
//               and does not fit when sizes[dim] = div_ceil(originalSize, minFactor).
// Postcondition: sizes[dim] is set to the largest fitting tile size.
static LogicalResult searchTileSizeForDim(
    IRRewriter &rewriter, Operation *consumerOp, const SetVector<Operation *> &producerOps,
    size_t availableMemoryBytes, MutableArrayRef<OpFoldResult> offsets,
    MutableArrayRef<OpFoldResult> sizes, size_t dim, int64_t originalSize, int64_t minFactor,
    int64_t maxFactor
) {
    while (maxFactor != minFactor + 1) {
        int64_t midFactor = midpoint(minFactor, maxFactor);
        sizes[dim] = rewriter.getIndexAttr(div_ceil(originalSize, midFactor));
        auto tileFits = checkTileFitsInMemory(
            rewriter, consumerOp, offsets, sizes, producerOps, availableMemoryBytes
        );
        if (failed(tileFits))
            return LogicalResult::failure();
        if (*tileFits)
            maxFactor = midFactor;
        else
            minFactor = midFactor;
    }
    sizes[dim] = rewriter.getIndexAttr(div_ceil(originalSize, maxFactor));
    return LogicalResult::success();
}

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

    // Shrink pass: set dimensions to 1, one by one, until the tile fits.
    size_t dimIndex;
    for (dimIndex = 0; dimIndex < tilingDimOrder.size(); ++dimIndex) {
        auto dim = tilingDimOrder[dimIndex];
        if (*getConstantIntValue(sizes[dim]) == 1)
            continue;

        sizes[dim] = one;
        tileFits = checkTileFitsInMemory(
            rewriter, consumerOp, offsets, sizes, producerOps, availableMemoryBytes
        );
        if (failed(tileFits))
            return LogicalResult::failure();
        if (*tileFits) {
            break;
        }
    }

    if (dimIndex == tilingDimOrder.size()) {
        consumerOp->emitWarning("operation can't be tiled: no more dimensions to tile");
        return LogicalResult::failure();
    }

    // Grow-back pass: inflate dimensions forced to 1 above back to larger tiles
    // using binary search. Iterate in reverse from where the shrink pass stopped.
    do {
        auto dim = tilingDimOrder[dimIndex];
        if (originalSizes[dim] == 1)
            continue;

        sizes[dim] = rewriter.getIndexAttr(originalSizes[dim]);
        tileFits = checkTileFitsInMemory(
            rewriter, consumerOp, offsets, sizes, producerOps, availableMemoryBytes
        );
        if (failed(tileFits))
            return LogicalResult::failure();
        if (*tileFits)
            continue;

        // We know `originalSizes[dim]` is too big (tested above), hence factor
        // 1 = originalSizes[dim]/originalSizes[dim] is a good minimum, and we know
        // sizes[dim] = 1 does fit, hence factor originalSizes[dim] = originalSizes[dim]/1
        // is a good maximum.
        if (failed(searchTileSizeForDim(
                rewriter, consumerOp, producerOps, availableMemoryBytes, offsets, sizes, dim,
                originalSizes[dim], /*minFactor=*/1, /*maxFactor=*/originalSizes[dim]
            )))
            return LogicalResult::failure();
    } while (dimIndex-- > 0);

    return true;
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
        if (getConstantIntValue(mappedSizes[index]) != getConstantIntValue(iterSizes[index]) &&
            !tilingDims.contains(index)) {
            return doNotFuse;
        }
    }

    // Producer can be fused
    return std::make_tuple(true, yieldProducerReplacement);
}

FailureOr<scf::SCFTileAndFuseResult> tileAndFuseToSize(
    IRRewriter &rewriter, TilingInterface tilingInterfaceOp, int64_t availableMemoryBytes,
    llvm::ArrayRef<OpFoldResult> completeIterSizes, ArrayRef<OpFoldResult> tileIterSizes,
    TileAndFuseProducersFuseMode fuseMode
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
        llvm::ListSeparator LSCross("x");
        for (auto d : tileSizes) {
            llvm::dbgs() << LSCross << d;
        }
        llvm::dbgs() << "\n";
    });

    auto maybePatternFuseGroup = isFuseGroupOutput(tilingInterfaceOp);

    scf::SCFTileAndFuseOptions options{};
    // Consider using setSCFTileSizes from iree/compiler/Codegen/Utils/Utils.h
    options.tilingOptions.setTileSizes(getAsIndexOpFoldResult(rewriter.getContext(), tileSizes));

    scf::SCFTileAndFuseOptions::ControlFnTy fusionControlFn;
    switch (fuseMode) {
    case TileAndFuseProducersFuseMode::MaxSize:
        fusionControlFn = [&](mlir::tensor::ExtractSliceOp candidateSliceOp,
                              OpResult producerOpResult, bool isDestinationOperand) {
            return fuseControlMaxSize(
                rewriter, availableMemoryBytes, candidateSliceOp, producerOpResult,
                isDestinationOperand
            );
        };
        break;

    case TileAndFuseProducersFuseMode::MaxProducers:
        fusionControlFn = [&](mlir::tensor::ExtractSliceOp candidateSliceOp,
                              OpResult producerOpResult, bool isDestinationOperand) {
            return fuseControlMaxProducers(
                rewriter, candidateSliceOp, producerOpResult, isDestinationOperand
            );
        };
        break;

    case TileAndFuseProducersFuseMode::OnlyPatterns:
        fusionControlFn = [&](mlir::tensor::ExtractSliceOp candidateSliceOp,
                              OpResult producerOpResult, bool isDestinationOperand) {
            bool shouldFuse =
                maybePatternFuseGroup && isMarkedFuseGroup(producerOpResult.getOwner());
            return std::make_tuple(shouldFuse, false);
        };
        break;

    case TileAndFuseProducersFuseMode::NoFuse:
        fusionControlFn = [&](mlir::tensor::ExtractSliceOp candidateSliceOp,
                              OpResult producerOpResult,
                              bool isDestinationOperand) { return std::make_tuple(false, false); };
    }

    options.setFusionControlFn(fusionControlFn);

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
                auto principalOp = getFuseGroupPrincipalOpBackward(op);
                llvm::dbgs() << "  principal op: " << principalOp->getName() << "\n";
                auto fuseGroup = principalOp->getAttr(TORQ_FUSE_GROUP_ID);
                llvm::dbgs() << "  " << TORQ_FUSE_GROUP << ": " << getConstantIntValue(fuseGroup)
                             << "\n";
            }
        }

        llvm::dbgs() << "  iteration sizes: ";
        llvm::ListSeparator LSCross("x");
        for (auto size : iterSizes) {
            llvm::dbgs() << LSCross << getConstantIntValue(size);
        }
        llvm::dbgs() << "\n";

        llvm::dbgs() << "  iteration types: ";
        llvm::ListSeparator LS;
        for (auto iterType : tilingInterfaceOp.getLoopIteratorTypes()) {
            llvm::dbgs() << LS << iterType;
        }
        llvm::dbgs() << "\n";
    });

    FailureOr<scf::SCFTileAndFuseResult> tiledResults = tileAndFuseToSize(
        rewriter, tilingInterfaceOp, availableMemoryBytes, iterSizes, tileSizes,
        clTorqTileAndFuseProducersFuseMode.getValue()
    );
    if (failed(tiledResults)) {
        op->emitError("tile and fuse failed");
        return;
    }

    if (clTorqTileAndFuseProducersFuseMode.getValue() ==
        TileAndFuseProducersFuseMode::MaxProducers) {
        SmallVector<OpFoldResult> fitTileOffsets(tileOffsets), fitTileSizes(tileSizes);

        // Now that we know which producers were fused, find a tile that fits them too.
        tileChanged = fitTileToMemory(
            rewriter, op, tiledResults->fusedProducers, tilingDimOrder, availableMemoryBytes,
            *iterIntSizes, fitTileOffsets, fitTileSizes
        );
        if (failed(tileChanged)) {
            tiledResults = tileAndFuseToSize(
                rewriter, tilingInterfaceOp, availableMemoryBytes, iterSizes, tileSizes,
                TileAndFuseProducersFuseMode::MaxProducers
            );
            if (failed(tiledResults)) {
                op->emitError("tile and fuse failed");
                assert(false);
                return;
            }
        }
        else if (*tileChanged) {
            // The second fitTileToMemory returned a smaller tile.

            tiledResults = tileAndFuseToSize(
                rewriter, tilingInterfaceOp, availableMemoryBytes, iterSizes, fitTileSizes,
                clTorqTileAndFuseProducersFuseMode.getValue()
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
