// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// ConvertNhwcConvToNchw Pass
//===----------------------------------------------------------------------===//
//
// Converts linalg.conv_2d_nhwc_hwcf to linalg.conv_2d_nchw_fchw and updates
// ALL related ops in the conv cluster:
//
//   - Bias broadcast  : output shape NHWC→NCHW, channel map d3→d1
//   - Conv fill init  : shape NHWC→NCHW
//   - Conv            : input [0,3,1,2], filter [3,2,0,1], NCHW output
//   - Bias add        : inputs are now NCHW, output shape NHWC→NCHW
//   - WZP reduction   : input map (d1,d2,d3,d0)→(d0,d1,d2,d3) for FCHW filter
//   - WZP correction  : channel map d3→d1
//   - Rescale/quantize: channel maps d3→d1, output shape NHWC→NCHW
//   - Final transpose : NCHW→NHWC [0,2,3,1] at the very end
//
// EXAMPLE:
//   BEFORE:
//     %input  = ... : tensor<1x80x80x64xi8>             // NHWC
//     %filter = linalg.transpose %cst, [1,2,3,0]        // HWCF [1,1,64,64]
//     %conv   = linalg.conv_2d_nhwc_hwcf ins(%input, %filter)  // NHWC out
//     %bias   = linalg.generic ins(%conv, %bias_bc)     // bias add NHWC
//     %quant  = linalg.generic ins(%bias, %scale, %zp)  // rescale NHWC
//     store %quant                                       // NHWC
//
//   AFTER:
//     %in_nchw   = linalg.transpose %input,  [0,3,1,2]  // NCHW [1,64,80,80]
//     %flt_fchw  = linalg.transpose %filter, [3,2,0,1]  // FCHW [64,64,1,1]
//     %conv_nchw = linalg.conv_2d_nchw_fchw ins(%in_nchw, %flt_fchw)
//     %bias_nchw = linalg.generic ins(%conv_nchw, %bias_bc) // NCHW, map (d1)
//     %q_nchw    = linalg.generic ins(%bias_nchw, %scale, %zp) // NCHW, map (d1)
//     %out_nhwc  = linalg.transpose %q_nchw, [0,2,3,1]  // back to NHWC
//     store %out_nhwc
//===----------------------------------------------------------------------===//

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/LayoutTransformUtils.h"

#include "PassesDetail.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-convert-nhwc-conv-to-nchw"

namespace mlir::syna::torq {

namespace {

// Check if a value is a zero-filled tensor (linalg.fill with zero OR arith.constant dense<0>)
bool isZeroFilledTensor(Value val) {
    // Case 1: linalg.fill with zero constant
    if (auto fillOp = val.getDefiningOp<linalg::FillOp>()) {
        Value fillValue = fillOp.getInputs()[0];
        if (auto constOp = fillValue.getDefiningOp<arith::ConstantOp>()) {
            auto attr = constOp.getValue();
            if (auto intAttr = dyn_cast<IntegerAttr>(attr))
                return intAttr.getValue().isZero();
            if (auto floatAttr = dyn_cast<FloatAttr>(attr))
                return floatAttr.getValue().isZero();
        }
        return false;
    }

    // Case 2: arith.constant with dense zero tensor
    if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
        if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
            if (!denseAttr.isSplat())
                return false;
            // Check if splat value is zero
            Type elemType = denseAttr.getElementType();
            if (isa<IntegerType>(elemType))
                return denseAttr.getSplatValue<APInt>().isZero();
            if (isa<FloatType>(elemType))
                return denseAttr.getSplatValue<APFloat>().isZero();
        }
    }

    return false;
}

// Transpose an input value, handling any tensor.pad chain by inserting the
// transpose before the first pad and rebuilding pads on top of the transposed input.
Value transposeInputWithPad(
    Value input, SmallVector<int64_t> perm, Location loc, OpBuilder &builder
) {
    // Only handle a single tensor.pad op, not a chain
    Value transposeInput = input;
    tensor::PadOp padOp = nullptr;
    if (auto maybePad = transposeInput.getDefiningOp<tensor::PadOp>()) {
        padOp = maybePad;
        transposeInput = padOp.getSource();
    }

    // Transpose the source (before any pads)
    Value result = transposeValue(transposeInput, SmallVector<int64_t>(perm), loc, builder);

    // If there was a pad, rebuild it on top of the transposed input
    if (padOp) {
        auto origPadType = cast<RankedTensorType>(padOp.getResult().getType());
        SmallVector<int64_t> nchwPadShape = nhwcToNchwShape(origPadType.getShape());

        // Permute low/high padding from NHWC [N,H,W,C] to NCHW [N,C,H,W]
        auto sLow = padOp.getStaticLow();
        auto sHigh = padOp.getStaticHigh();
        SmallVector<int64_t> nchwStaticLow = {sLow[0], sLow[3], sLow[1], sLow[2]};
        SmallVector<int64_t> nchwStaticHigh = {sHigh[0], sHigh[3], sHigh[1], sHigh[2]};

        // dLow/dHigh are empty when all dims are static — only permute if present.
        auto dLow = padOp.getLow();
        auto dHigh = padOp.getHigh();
        SmallVector<Value> nchwDynLow, nchwDynHigh;
        if (!dLow.empty())
            nchwDynLow = {dLow[0], dLow[3], dLow[1], dLow[2]};
        if (!dHigh.empty())
            nchwDynHigh = {dHigh[0], dHigh[3], dHigh[1], dHigh[2]};

        auto newPadType = RankedTensorType::get(nchwPadShape, origPadType.getElementType());
        auto newPadOp = tensor::PadOp::create(
            builder, padOp.getLoc(), newPadType, result, nchwStaticLow, nchwStaticHigh, nchwDynLow,
            nchwDynHigh, padOp.getNofold()
        );

        // Clone the padding value region
        IRMapping mapping;
        padOp.getRegion().cloneInto(&newPadOp.getRegion(), mapping);
        result = newPadOp.getResult();
    }
    return result;
}

// In an indexing map, replace AffineDimExpr(3) (NHWC channel) with
// AffineDimExpr(1) (NCHW channel).  All other expressions are unchanged.
AffineMap remapChannelDim(AffineMap map, MLIRContext *ctx) {
    SmallVector<AffineExpr> results;
    for (auto expr : map.getResults()) {
        auto dim = dyn_cast<AffineDimExpr>(expr);
        if (dim && dim.getPosition() == 3)
            results.push_back(getAffineDimExpr(1, ctx));
        else
            results.push_back(expr);
    }
    return AffineMap::get(map.getNumDims(), map.getNumSymbols(), results, ctx);
}

// Build the correct NCHW indexing map for a given original NHWC map.
//
// For 4D activation tensors (feature maps):
//   Use identity map (d0,d1,d2,d3) — both NHWC and NCHW use the same indexing
//   after the data has been physically transposed.
//   Example: NHWC [1,80,80,64] → NCHW [1,64,80,80]
//            Both indexed as (d0,d1,d2,d3) but data layout differs.
//
// For 1D per-channel tensors (bias, scale, zero-point vectors):
//   Remap d3 (NHWC channel position) → d1 (NCHW channel position).
//   Example: bias shape [64], map (d3) in NHWC → map (d1) in NCHW
//            because channel dimension moved from position 3 to position 1.
AffineMap nchwMap(AffineMap origMap, MLIRContext *ctx) {
    if (origMap.getNumResults() == 4)
        return AffineMap::getMultiDimIdentityMap(4, ctx);
    return remapChannelDim(origMap, ctx);
}

// Convert a linalg.generic op from NHWC to NCHW layout.
// This function:
//   - Remaps inputs using the provided valMap (NHWC -> NCHW replacements)
//   - Converts indexing maps:
//       4D tensors  → identity (data was already transposed by the anchor op)
//       1D channel  → remap d3 (NHWC C) to d1 (NCHW C)
//       3D transposed (e.g. depthwise filter HWC→CHW) → compose map with
//           the transpose permutation so loop vars index the new layout
//   - Transforms 4D output shapes from NHWC to NCHW
//   - Rebuilds the generic op with the new layout
Value convertGenericOpToNchw(
    linalg::GenericOp genericOp, OpBuilder &builder, MLIRContext *ctx,
    const DenseMap<Value, Value> &valMap
) {
    auto origMaps = genericOp.getIndexingMapsArray();
    unsigned numInps = genericOp.getInputs().size();

    SmallVector<Value> newInputs;
    SmallVector<AffineMap> newMaps;

    for (auto [i, inp] : llvm::enumerate(genericOp.getInputs())) {
        auto it = valMap.find(inp);
        Value newInp;

        if (it != valMap.end()) {
            // Input already converted (in valMap)
            newInp = it->second;
        }
        else {
            // Input not in valMap - check if it's a 4D NHWC tensor that needs transposing
            auto inpType = dyn_cast<RankedTensorType>(inp.getType());
            if (inpType && inpType.getRank() == 4 && isZeroFilledTensor(inp)) {
                // 4D zero-filled NHWC constant/fill → transpose to NCHW
                SmallVector<int64_t> nchwShape = nhwcToNchwShape(inpType.getShape());
                newInp = createZeroFilledTensor(
                    builder, genericOp.getLoc(), nchwShape, inpType.getElementType()
                );
            }
            else {
                // Keep as-is (1D channel tensors, non-zero tensors, etc.)
                newInp = inp;
            }
        }

        newInputs.push_back(newInp);

        AffineMap origMap = origMaps[i];

        // Special case: 3D input that was replaced by a transposed value
        // (e.g. depthwise filter HWC→CHW with perm=[2,0,1]).
        // The original map was written for the HWC layout; after transposing to
        // CHW the same loop variables address different dimensions.  Compose the
        // map with the permutation so indices into the new tensor are correct.
        //   Example: origMap = (d0,d1,d2) -> (d1,d2,d0)   [HWC: H=d1,W=d2,C=d0]
        //            perm    = [2,0,1]                     (HWC→CHW)
        //            newMap  = (d0,d1,d2) -> (d0,d1,d2)   [CHW: C=d0,H=d1,W=d2] ✓
        if (it != valMap.end() && origMap.getNumResults() == 3) {
            if (auto transpOp = newInp.getDefiningOp<linalg::TransposeOp>()) {
                auto perm = transpOp.getPermutation();
                SmallVector<AffineExpr> newExprs;
                for (int64_t p : perm)
                    newExprs.push_back(origMap.getResults()[p]);
                newMaps.push_back(
                    AffineMap::get(origMap.getNumDims(), origMap.getNumSymbols(), newExprs, ctx)
                );
                continue;
            }
        }
        // Default: 4D → identity, 1D → remap channel d3→d1
        newMaps.push_back(nchwMap(origMap, ctx));
    }

    // Output map (last in origMaps)
    newMaps.push_back(nchwMap(origMaps[numInps], ctx));

    // Update 4D output-init shape to NCHW, leave 1D unchanged.
    // IMPORTANT: If the original init is zero-filled, create a new zero-filled tensor.
    // Otherwise, transpose the original init to preserve its values (e.g., for bias-add
    // or residual connections where the init is an accumulator).
    Value origOutInit = genericOp.getOutputs()[0];
    auto origOutType = cast<RankedTensorType>(origOutInit.getType());

    SmallVector<int64_t> newOutShape = nhwcToNchwShape(origOutType.getShape());
    Value newOutInit;
    if (isZeroFilledTensor(origOutInit)) {
        // Safe to create a new zero-filled tensor with NCHW shape
        newOutInit = createZeroFilledTensor(
            builder, genericOp.getLoc(), newOutShape, origOutType.getElementType()
        );
    }
    else {
        // Transpose the original init to preserve its values
        newOutInit =
            transposeValue(origOutInit, Permutation::nhwc2nchw(), genericOp.getLoc(), builder);
    }

    return rebuildGenericWithNewLayout(builder, genericOp, newInputs, newOutInit, newMaps);
}

//===----------------------------------------------------------------------===//
// Three-phase NHWC→NCHW conversion (generalized for any anchor op)
// Supports any NHWC anchor op (conv, maxpool, depthwise, ...).
// Add new op types by updating convertAnchorToNchw() and isSupportedNhwcAnchorOp().
// Phase 1: Collect all clusters (no IR mutation).
// Phase 2: Convert each cluster; shared ops are only converted once.
//===----------------------------------------------------------------------===//

// Returns true if this op is a supported NHWC anchor op.
// Add new op types here when extending support.
bool isSupportedNhwcAnchorOp(Operation *op) {
    return isa<
        linalg::Conv2DNhwcHwcfOp, linalg::PoolingNhwcMaxOp, linalg::DepthwiseConv2DNhwcHwcOp>(op);
}

// Generic cluster descriptor — works for any supported NHWC anchor op.
struct NhwcCluster {
    Operation *anchorOp;                // Conv / MaxPool / ... in NHWC layout
    SmallVector<Operation *> neededOps; // dependent generics, topological order
    Operation *outputOp;                // last op — gets the NCHW→NHWC transpose
};

// Convert the anchor op itself to its NCHW equivalent.
// inputNchw is the already-transposed [N,C,H,W] input.
// May add extra entries to valMap (e.g. filter→filterFchw for conv).
// Returns the NCHW result value, or failure for unsupported op types.
FailureOr<Value> convertAnchorToNchw(
    Operation *anchorOp, Value inputNchw, IRRewriter &rewriter, Location loc,
    DenseMap<Value, Value> &valMap
) {
    return TypeSwitch<Operation *, FailureOr<Value>>(anchorOp)

        // ── linalg.conv_2d_nhwc_hwcf → linalg.conv_2d_nchw_fchw ──────────
        .Case<linalg::Conv2DNhwcHwcfOp>([&](linalg::Conv2DNhwcHwcfOp convOp) -> FailureOr<Value> {
            Value filter = convOp.getInputs()[1];
            Value filterFchw = transposeValue(filter, Permutation::hwcf2fchw(), loc, rewriter);
            valMap[filter] = filterFchw;
            Value outInit = convOp.getOutputs()[0];
            auto outType = cast<RankedTensorType>(outInit.getType());
            Value nchwInit = createZeroFilledTensor(
                rewriter, loc, nhwcToNchwShape(outType.getShape()), outType.getElementType()
            );
            auto nchwOp = linalg::Conv2DNchwFchwOp::create(
                rewriter, loc, nchwInit.getType(), ValueRange{inputNchw, filterFchw},
                ValueRange{nchwInit}, convOp.getStrides(), convOp.getDilations()
            );
            LLVM_DEBUG(
                llvm::dbgs() << "[nhwc→nchw] Conv2D  NCHW: " << nchwOp.getResult(0).getType()
                             << "\n"
            );
            return nchwOp.getResult(0);
        })

        // ── linalg.pooling_nhwc_max → linalg.pooling_nchw_max ─────────────
        .Case<linalg::PoolingNhwcMaxOp>([&](linalg::PoolingNhwcMaxOp poolOp) -> FailureOr<Value> {
            Value kernel = poolOp.getInputs()[1];
            Value outInit = poolOp.getOutputs()[0];
            auto outType = cast<RankedTensorType>(outInit.getType());
            Value nchwInit = createMinFilledTensor(
                rewriter, loc, nhwcToNchwShape(outType.getShape()), outType.getElementType()
            );
            auto nchwOp = linalg::PoolingNchwMaxOp::create(
                rewriter, loc, nchwInit.getType(), ValueRange{inputNchw, kernel},
                ValueRange{nchwInit}, poolOp.getStrides(), poolOp.getDilations()
            );
            LLVM_DEBUG(
                llvm::dbgs() << "[nhwc→nchw] MaxPool NCHW: " << nchwOp.getResult(0).getType()
                             << "\n"
            );
            return nchwOp.getResult(0);
        })

        // ── linalg.depthwise_conv_2d_nhwc_hwc → linalg.depthwise_conv_2d_nchw_chw ──
        // Same structure as Conv2D except:
        //   filter is 3D [H,W,C] → transpose to [C,H,W] via [2,0,1]  (no F dim)
        .Case<linalg::DepthwiseConv2DNhwcHwcOp>(
            [&](linalg::DepthwiseConv2DNhwcHwcOp depthwiseOp) -> FailureOr<Value> {
                Value filter = depthwiseOp.getInputs()[1];
                Value filterChw = transposeValue(filter, Permutation::hwc2chw(), loc, rewriter);
                valMap[filter] = filterChw;
                Value outInit = depthwiseOp.getOutputs()[0];
                auto outType = cast<RankedTensorType>(outInit.getType());
                Value nchwInit = createZeroFilledTensor(
                    rewriter, loc, nhwcToNchwShape(outType.getShape()), outType.getElementType()
                );
                auto nchwOp = linalg::DepthwiseConv2DNchwChwOp::create(
                    rewriter, loc, nchwInit.getType(), ValueRange{inputNchw, filterChw},
                    ValueRange{nchwInit}, depthwiseOp.getStrides(), depthwiseOp.getDilations()
                );
                LLVM_DEBUG(
                    llvm::dbgs() << "[nhwc→nchw] DepthwiseConv2D NCHW: "
                                 << nchwOp.getResult(0).getType() << "\n"
                );
                return nchwOp.getResult(0);
            }
        )

        .Default([](Operation *op) -> FailureOr<Value> {
            LLVM_DEBUG(llvm::dbgs() << "[nhwc→nchw] Unsupported anchor: " << op->getName() << "\n");
            return failure();
        });
}

// Phase 2: convert one cluster.
// All supported NHWC anchor ops have their 4D NHWC tensor as DPS input #0,
// which the LinalgOp interface exposes uniformly.
LogicalResult convertCluster(
    const NhwcCluster &cluster, IRRewriter &rewriter, DenseMap<Value, Value> &globalValMap
) {
    MLIRContext *ctx = rewriter.getContext();
    Operation *anchorOp = cluster.anchorOp;
    Location loc = anchorOp->getLoc();

    // DPS input #0 is always the 4D NHWC tensor for every supported anchor op.
    Value input = cast<linalg::LinalgOp>(anchorOp).getDpsInputs()[0];

    //------------------------------------------------------------------
    // Step 1: Transpose 4D NHWC input → NCHW [0,3,1,2]
    //------------------------------------------------------------------
    rewriter.setInsertionPoint(anchorOp);
    Value inputNchw = transposeInputWithPad(input, Permutation::nhwc2nchw(), loc, rewriter);
    LLVM_DEBUG(llvm::dbgs() << "[nhwc→nchw] Input NCHW: " << inputNchw.getType() << "\n");

    //------------------------------------------------------------------
    // Step 2: Create the NCHW anchor op via per-type dispatch.
    //------------------------------------------------------------------
    // Local map pre-seeded with everything already converted by prior clusters.
    DenseMap<Value, Value> valMap(globalValMap.begin(), globalValMap.end());

    FailureOr<Value> anchorNchw = convertAnchorToNchw(anchorOp, inputNchw, rewriter, loc, valMap);
    if (failed(anchorNchw))
        return failure();

    // Map the original anchor result → NCHW result for downstream generics.
    valMap[anchorOp->getResult(0)] = *anchorNchw;

    // Standalone anchor (no downstream generics) — insert the NCHW→NHWC
    // transpose immediately and we are done.
    if (cluster.neededOps.empty()) {
        rewriter.setInsertionPointAfterValue(*anchorNchw);
        Value finalNhwc = transposeValue(*anchorNchw, Permutation::nchw2nhwc(), loc, rewriter);
        rewriter.replaceAllUsesWith(anchorOp->getResult(0), finalNhwc);
        LLVM_DEBUG(
            llvm::dbgs() << "[nhwc→nchw] Standalone NCHW→NHWC: " << finalNhwc.getType() << "\n"
        );
        return success();
    }

    //------------------------------------------------------------------
    // Step 3: Convert each generic op in the cluster.
    // neededOps was collected before any changes, so each pointer is the original NHWC op.
    // globalValMap makes sure we only convert each op once, even if it is shared.
    //------------------------------------------------------------------
    for (Operation *op : cluster.neededOps) {
        auto genericOp = dyn_cast_or_null<linalg::GenericOp>(op);
        if (!genericOp)
            continue;

        Value oldResult = genericOp.getResult(0);

        if (globalValMap.count(oldResult)) {
            // Already converted by an earlier cluster — reuse its NCHW result.
            valMap[oldResult] = globalValMap[oldResult];
            LLVM_DEBUG(llvm::dbgs() << "[nhwc→nchw] Shared generic reused\n");
            continue;
        }

        rewriter.setInsertionPoint(genericOp);
        Value newResult = convertGenericOpToNchw(genericOp, rewriter, ctx, valMap);

        valMap[oldResult] = newResult;
        globalValMap[oldResult] = newResult;

        LLVM_DEBUG(llvm::dbgs() << "[nhwc→nchw] Generic: " << newResult.getType() << "\n");

        // The final op in the cluster gets an NCHW→NHWC transpose so the
        // rest of the graph remains in NHWC layout.
        if (op == cluster.outputOp) {
            Value finalNhwc = transposeValue(newResult, Permutation::nchw2nhwc(), loc, rewriter);
            rewriter.replaceAllUsesWith(oldResult, finalNhwc);
            LLVM_DEBUG(
                llvm::dbgs() << "[nhwc→nchw] Output NCHW→NHWC: " << finalNhwc.getType() << "\n"
            );
        }
    }

    return success();
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public entry point
//===----------------------------------------------------------------------===//

void convertNhwcOpToNchwOp(FunctionOpInterface funcOp) {
    IRRewriter rewriter(funcOp.getContext());

    //------------------------------------------------------------------
    // Phase 1: Collect ALL fusion plans before any mutation.
    //------------------------------------------------------------------
    SmallVector<Operation *> anchorOps;
    funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        if (isSupportedNhwcAnchorOp(op)) {
            anchorOps.push_back(op);
            return WalkResult::skip();
        }
        return WalkResult::advance();
    });

    LLVM_DEBUG(llvm::dbgs() << "[nhwc→nchw] Phase 1: " << anchorOps.size() << " anchor ops\n");

    SmallVector<NhwcCluster> clusters;
    clusters.reserve(anchorOps.size());

    for (Operation *anchorOp : anchorOps) {
        Value anchorResult = anchorOp->getResult(0);
        FailureOr<FusionPlan> planOr = buildFusionPlanAndRebindOutput(anchorResult);

        SmallVector<Operation *> neededOps;
        Operation *outOp = anchorOp; // default: anchor is its own output

        if (succeeded(planOr) && !planOr->neededOps.empty()) {
            // Build filter chain exclusion set: collect all ops that produce the filter/weights
            llvm::SmallPtrSet<Operation *, 8> filterChain;
            auto convOp = llvm::cast<linalg::ConvolutionOpInterface>(anchorOp);
            if (auto filter = convOp.filter()) {
                SmallVector<Value, 8> stack;
                stack.push_back(filter);
                while (!stack.empty()) {
                    Value v = stack.pop_back_val();
                    if (!v)
                        continue;
                    if (Operation *def = v.getDefiningOp()) {
                        if (filterChain.insert(def).second) {
                            for (Value operand : def->getOperands())
                                stack.push_back(operand);
                        }
                    }
                }
            }

            // Copy neededOps, excluding any ops in the filter/weight chain.
            for (Operation *op : planOr->neededOps) {
                if (!filterChain.contains(op)) {
                    neededOps.push_back(op);
                }
            }

            if (!neededOps.empty())
                outOp = neededOps.back();

            LLVM_DEBUG(
                llvm::dbgs() << "[nhwc→nchw]   " << anchorOp->getName()
                             << " cluster: " << neededOps.size() << " ops (filtered)\n"
            );
        }
        else {
            // No fusion (e.g. standalone maxpool) — anchor IS the output.
            // convertCluster will insert the NCHW→NHWC transpose directly.
            LLVM_DEBUG(
                llvm::dbgs() << "[nhwc→nchw]   " << anchorOp->getName()
                             << " standalone (no fusion)\n"
            );
        }

        clusters.push_back({anchorOp, neededOps, outOp});
    }

    //------------------------------------------------------------------
    // Phase 2: Conversion — globalValMap deduplicates shared ops.
    //------------------------------------------------------------------
    LLVM_DEBUG(llvm::dbgs() << "[nhwc→nchw] Phase 2: " << clusters.size() << " clusters\n");

    DenseMap<Value, Value> globalValMap;
    for (auto &cluster : clusters) {
        if (failed(convertCluster(cluster, rewriter, globalValMap)))
            LLVM_DEBUG(
                llvm::dbgs() << "[nhwc→nchw] convertCluster failed: " << cluster.anchorOp->getName()
                             << "\n"
            );
    }
}

//===----------------------------------------------------------------------===//
// Pass wrapper
//===----------------------------------------------------------------------===//

namespace {

class ConvertNhwcOpToNchwPass : public impl::ConvertNhwcOpToNchwBase<ConvertNhwcOpToNchwPass> {
  public:
    void runOnOperation() override { convertNhwcOpToNchwOp(getOperation()); }
};

} // anonymous namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createConvertNhwcOpToNchwPass() {
    return std::make_unique<ConvertNhwcOpToNchwPass>();
}

} // namespace mlir::syna::torq
