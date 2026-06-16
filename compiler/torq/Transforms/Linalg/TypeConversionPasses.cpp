// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/ConversionUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include <functional>

namespace mlir::syna::torq {
namespace {

static llvm::cl::opt<bool> clConvertDtypes(
    "torq-convert-dtypes", llvm::cl::desc("Enable Torq dtype conversion passes"),
    llvm::cl::init(false)
);

static llvm::cl::opt<bool> clConvertIODType(
    "torq-convert-io-dtype",
    llvm::cl::desc("Convert model I/O dtypes instead of inserting boundary casts"),
    llvm::cl::init(false)
);

/// Rebuilds shaped types with a new element type while preserving their shape and encoding.
Type cloneTypeWithElementType(Type type, Type elementType) {
    if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
        return RankedTensorType::get(rankedType.getShape(), elementType, rankedType.getEncoding());
    }
    if (auto unrankedType = dyn_cast<UnrankedTensorType>(type)) {
        return UnrankedTensorType::get(elementType);
    }
    if (auto vectorType = dyn_cast<VectorType>(type)) {
        return VectorType::get(vectorType.getShape(), elementType);
    }
    return elementType;
}

/// Shared materialization helper for float conversions.
Value convertRankedFloat(OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
    Type elementType = getElementTypeOrSelf(type);
    Type inputElementType = getElementTypeOrSelf(inputs[0].getType());
    if (!isa<FloatType>(elementType) || !isa<FloatType>(inputElementType)) {
        return nullptr;
    }

    if (inputElementType == elementType) {
        return nullptr;
    }

    if (inputElementType.getIntOrFloatBitWidth() > elementType.getIntOrFloatBitWidth()) {
        return arith::TruncFOp::create(builder, loc, type, inputs[0]);
    }

    if (inputElementType.getIntOrFloatBitWidth() < elementType.getIntOrFloatBitWidth()) {
        return arith::ExtFOp::create(builder, loc, type, inputs[0]);
    }

    /// MLIR does not have a direct way to cast f16 <-> bf16 so we use an f32 bridge
    if ((isa<Float16Type>(inputElementType) && isa<BFloat16Type>(elementType)) ||
        (isa<BFloat16Type>(inputElementType) && isa<Float16Type>(elementType))) {
        Type f32ElementType = Float32Type::get(builder.getContext());
        Type f32Type = cloneTypeWithElementType(type, f32ElementType);
        Value extended = arith::ExtFOp::create(builder, loc, f32Type, inputs[0]);
        return arith::TruncFOp::create(builder, loc, type, extended);
    }

    return nullptr;
}

/// Shared materialization helper for integer conversions.
Value convertRankedInteger(OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
    Type elementType = getElementTypeOrSelf(type);
    Type inputElementType = getElementTypeOrSelf(inputs[0].getType());
    if (!isa<IntegerType>(elementType) || !isa<IntegerType>(inputElementType)) {
        return nullptr;
    }

    bool isUnsigned = elementType.isUnsignedInteger();
    int64_t inWidth = cast<IntegerType>(inputElementType).getWidth();
    int64_t outWidth = cast<IntegerType>(elementType).getWidth();
    if (inWidth > outWidth) {
        return arith::TruncIOp::create(builder, loc, type, inputs[0]);
    }
    if (inWidth < outWidth) {
        if (isUnsigned)
            return arith::ExtUIOp::create(builder, loc, type, inputs[0]);
        else
            return arith::ExtSIOp::create(builder, loc, type, inputs[0]);
    }

    return nullptr;
}

/// Declares an op legal iff all observable types (operands, results, regions, signatures) are
/// legal.
void markAllTypesLegal(ConversionTarget &target, TypeConverter &tc) {
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
        if (auto g = dyn_cast<iree_compiler::IREE::Util::GlobalOpInterface>(op)) {
            return tc.isLegal(g.getGlobalType());
        }
        if (auto f = dyn_cast<FunctionOpInterface>(op)) {
            for (Type t : f.getArgumentTypes()) {
                if (!tc.isLegal(t))
                    return false;
            }
            for (Type t : f.getResultTypes()) {
                if (!tc.isLegal(t))
                    return false;
            }
        }
        for (Type t : op->getResultTypes()) {
            if (!tc.isLegal(t))
                return false;
        }
        for (Type t : op->getOperandTypes()) {
            if (!tc.isLegal(t))
                return false;
        }
        for (auto &region : op->getRegions()) {
            if (!tc.isLegal(&region))
                return false;
        }
        return true;
    });
}

struct PreservedIOFuncInfo {
    StringAttr name;
    FunctionType originalType;
};

bool isTopLevelFunc(func::FuncOp funcOp, Operation *rootOp) {
    return isa_and_nonnull<ModuleOp>(rootOp) && funcOp->getParentOp() == rootOp;
}

void collectPreservedIO(Operation *op, SmallVectorImpl<PreservedIOFuncInfo> &preservedIO) {
    ModuleOp module = dyn_cast<ModuleOp>(op);
    if (!module) {
        return;
    }

    for (auto funcOp : module.getOps<func::FuncOp>()) {
        if (!isTopLevelFunc(funcOp, op))
            continue;
        // Only entry points (public funcs) get I/O preservation. Private helpers are converted
        // end-to-end; their internal call sites are rewritten to the converted dtypes, so
        // restoring a callee's original signature would leave those calls type-mismatched.
        if (!funcOp.isPublic())
            continue;
        // Defensive: a public function that is also called internally cannot have its signature
        // restored without breaking those (already-converted) call sites. Warn and let it convert
        // fully rather than emit invalid IR.
        auto uses = SymbolTable::getSymbolUses(funcOp.getOperation(), module);
        if (uses && !uses->empty()) {
            funcOp.emitWarning("I/O dtype not preserved: public function is also called internally"
            );
            continue;
        }
        preservedIO.push_back({funcOp.getSymNameAttr(), funcOp.getFunctionType()});
    }
}

/// Inserts casts on entry/exit to keep the function signature unchanged while
/// the body operates on the converted element types.
LogicalResult addBoundaryCasts(func::FuncOp funcOp, FunctionType originalType, TypeConverter &tc) {
    FunctionType convertedType = funcOp.getFunctionType();
    if (convertedType == originalType || funcOp.isExternal()) {
        return success();
    }

    if (convertedType.getNumInputs() != originalType.getNumInputs() ||
        convertedType.getNumResults() != originalType.getNumResults()) {
        funcOp.emitOpError("cannot preserve I/O dtype when the signature shape changes");
        return failure();
    }

    Block &oldEntry = funcOp.front();
    Block *newEntry = new Block();
    for (auto [idx, argType] : llvm::enumerate(originalType.getInputs())) {
        newEntry->addArgument(argType, oldEntry.getArgument(idx).getLoc());
    }

    funcOp.getBody().getBlocks().insert(funcOp.getBody().begin(), newEntry);
    OpBuilder builder(newEntry, newEntry->begin());

    SmallVector<Value> remappedArgs;
    remappedArgs.reserve(convertedType.getNumInputs());
    // Maps each materialized input-cast result back to its original (untouched-dtype) argument so
    // a value that flows unchanged to an output can bypass the lossy convert-and-convert-back
    // round-trip (e.g. an identity/passthrough tensor).
    DenseMap<Value, Value> inputCastToOriginal;
    for (auto [idx, arg] : llvm::enumerate(newEntry->getArguments())) {
        Type expectedType = convertedType.getInput(idx);
        Value mapped = arg;
        if (mapped.getType() != expectedType) {
            mapped = tc.materializeTargetConversion(builder, arg.getLoc(), expectedType, arg);
            if (!mapped) {
                funcOp.emitOpError("failed to materialize input boundary cast at index ") << idx;
                return failure();
            }
            inputCastToOriginal[mapped] = arg;
        }
        remappedArgs.push_back(mapped);
    }

    newEntry->getOperations().splice(newEntry->end(), oldEntry.getOperations());
    for (auto [idx, mapped] : llvm::enumerate(remappedArgs)) {
        Value oldArg = oldEntry.getArgument(idx);
        if (oldArg != mapped) {
            oldArg.replaceAllUsesWith(mapped);
        }
    }
    oldEntry.erase();

    SmallVector<Type> originalResults(originalType.getResults());
    LogicalResult returnStatus = success();
    funcOp.walk([&](func::ReturnOp returnOp) -> WalkResult {
        OpBuilder retBuilder(returnOp);
        SmallVector<Value> newOperands;
        newOperands.reserve(returnOp.getNumOperands());
        for (auto [idx, operand] : llvm::enumerate(returnOp.getOperands())) {
            Type expectedType = originalResults[idx];
            Value updated = operand;
            if (operand.getType() != expectedType) {
                // If the returned value is exactly a converted input we still hold in its original
                // dtype, route that original through directly instead of casting it back (avoids a
                // precision-losing round-trip).
                if (auto it = inputCastToOriginal.find(operand);
                    it != inputCastToOriginal.end() && it->second.getType() == expectedType) {
                    newOperands.push_back(it->second);
                    continue;
                }
                updated = tc.materializeSourceConversion(
                    retBuilder, returnOp.getLoc(), expectedType, operand
                );
                if (!updated) {
                    returnOp.emitError("failed to materialize output boundary cast at index ")
                        << idx;
                    returnStatus = failure();
                    return WalkResult::interrupt();
                }
            }
            newOperands.push_back(updated);
        }
        returnOp.getOperandsMutable().assign(newOperands);
        return WalkResult::advance();
    });

    if (failed(returnStatus)) {
        return failure();
    }

    funcOp.setType(originalType);
    return success();
}

/// Base type converter used for primitive element changes.
template <typename SourceType, typename TargetType>
struct PrimitiveTypeConverter : public TypeConverter {
    PrimitiveTypeConverter() {
        addConversion([](Type type) { return type; });
        addConversion([&](SourceType type) -> Type {
            if (!isSourceType(type)) {
                return type;
            }
            return getTargetType(type);
        });
        addConversion([&](ComplexType type) {
            return ComplexType::get(convertType(type.getElementType()));
        });
        addConversion([&](RankedTensorType type) {
            return RankedTensorType::get(
                type.getShape(), convertType(type.getElementType()), type.getEncoding()
            );
        });
        addConversion([&](VectorType type) {
            return VectorType::get(type.getShape(), convertType(type.getElementType()));
        });
        addConversion([&](iree_compiler::IREE::Util::PtrType ptrType) {
            return iree_compiler::IREE::Util::PtrType::get(convertType(ptrType.getTargetType()));
        });
    }

    virtual ~PrimitiveTypeConverter() = default;

    virtual bool isSourceType(SourceType type) { return true; }
    virtual Type getTargetType(SourceType type) = 0;
};

/// Warns when narrowing an integer constant drops information.
///
/// IREE's convertAttribute narrows integers with a silent modular truncation; we keep that
/// behavior (the conversion itself is still delegated to IREE) but surface a diagnostic so
/// out-of-range shape/index values are not silently corrupted. A value is representable iff it
/// fits the target width as either signed or unsigned.
void warnOnIntegerNarrowing(Location loc, TypedAttr attr, Type newType) {
    auto newShaped = dyn_cast<ShapedType>(newType);
    Type newElem = newShaped ? newShaped.getElementType() : newType;
    auto newIntTy = dyn_cast<IntegerType>(newElem);
    if (!newIntTy) {
        return;
    }
    unsigned width = newIntTy.getWidth();
    auto fits = [&](const APInt &v) {
        return v.getSignificantBits() <= width || v.getActiveBits() <= width;
    };

    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        if (!fits(intAttr.getValue())) {
            emitWarning(loc) << "narrowing integer constant to " << newIntTy
                             << " loses information";
        }
        return;
    }
    if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(attr)) {
        for (const APInt &v : denseAttr.getValues<APInt>()) {
            if (!fits(v)) {
                emitWarning(loc) << "narrowing integer constant to " << newIntTy
                                 << " loses information";
                break;
            }
        }
    }
}

/// Torq-local replacement for iree_compiler::convertAttribute.
///
/// Deviates from IREE in two intentional, Torq-specific ways; everything else falls back to
/// iree_compiler::convertAttribute unchanged:
///   1. Floating-point literals round to the target type with round-to-nearest-even, matching
///      arith.truncf and runtime behavior. IREE rounds toward zero, which biases converted
///      weights and diverges from the value a runtime truncf would produce.
///   2. Lossy integer narrowing emits a warning (the narrowing itself stays with IREE, so its
///      modular-truncation semantics are unchanged).
Attribute convertConstantAttribute(Location loc, Attribute oldAttr, const TypeConverter &tc) {
    if (auto typedAttr = dyn_cast<TypedAttr>(oldAttr)) {
        Type newType = tc.convertType(typedAttr.getType());
        if (newType && newType != typedAttr.getType()) {
            warnOnIntegerNarrowing(loc, typedAttr, newType);

            if (auto floatAttr = dyn_cast<FloatAttr>(typedAttr)) {
                auto newFloatTy = cast<FloatType>(newType);
                APFloat value = floatAttr.getValue();
                bool losesInfo = false;
                value.convert(
                    newFloatTy.getFloatSemantics(), APFloat::rmNearestTiesToEven, &losesInfo
                );
                return FloatAttr::get(newFloatTy, value);
            }
            if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(typedAttr)) {
                auto newElemTy = cast<FloatType>(cast<ShapedType>(newType).getElementType());
                const auto &sem = newElemTy.getFloatSemantics();
                return denseAttr.mapValues(newElemTy, [&](const APFloat &src) {
                    APFloat v = src;
                    bool losesInfo = false;
                    v.convert(sem, APFloat::rmNearestTiesToEven, &losesInfo);
                    return v.bitcastToAPInt();
                });
            }
        }
    }
    return iree_compiler::convertAttribute(loc, oldAttr, tc);
}

/// Generic op cloner that rewrites types, attributes, and regions.
///
/// Constants and globals rewrite attributes to avoid silently changing literal meaning;
/// all other ops keep attributes verbatim and only adjust types and region signatures.
struct GenericTypeConversionPattern : public ConversionPattern {
    GenericTypeConversionPattern(MLIRContext *context, TypeConverter &typeConverter)
        : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 0, context) {}

    LogicalResult matchAndRewrite(
        Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter
    ) const override {
        if (isa<arith::ExtFOp, arith::TruncFOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op
            )) {
            return failure();
        }

        SmallVector<NamedAttribute> newAttrs;
        if (op->hasTrait<OpTrait::ConstantLike>() ||
            isa<iree_compiler::IREE::Util::GlobalOpInterface>(op)) {
            for (auto attr : op->getAttrs()) {
                auto newAttr =
                    convertConstantAttribute(op->getLoc(), attr.getValue(), *getTypeConverter());
                newAttrs.emplace_back(attr.getName(), newAttr);
            }
        }
        else {
            newAttrs.append(op->getAttrs().begin(), op->getAttrs().end());
        }

        SmallVector<Type> newResults;
        (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);

        OperationState state(
            op->getLoc(), op->getName().getStringRef(), operands, newResults, newAttrs,
            op->getSuccessors()
        );

        for (Region &region : op->getRegions()) {
            Region *newRegion = state.addRegion();
            rewriter.inlineRegionBefore(region, *newRegion, newRegion->begin());
            TypeConverter::SignatureConversion signature(newRegion->getNumArguments());
            (void)getTypeConverter()->convertSignatureArgs(
                newRegion->getArgumentTypes(), signature
            );
            rewriter.applySignatureConversion(&newRegion->front(), signature);
        }

        Operation *newOp = rewriter.create(state);
        rewriter.replaceOp(op, newOp->getResults());
        return success();
    }
};

/// Normalizes arithmetic cast ops after type conversion.
///
/// Removes redundant casts created by type normalization and rematerializes float
/// casts whose operation kind changes after conversion.
template <typename OpTy> struct ConvertFloatArithCastOp : public OpConversionPattern<OpTy> {
    using OpConversionPattern<OpTy>::OpConversionPattern;
    LogicalResult matchAndRewrite(
        OpTy op, typename OpTy::Adaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {
        Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
        Value operand = adaptor.getOperands()[0];

        if (resultType == operand.getType()) {
            rewriter.replaceOp(op, operand);
            return success();
        }

        Value converted = convertRankedFloat(rewriter, resultType, operand, op.getLoc());
        if (!converted) {
            return rewriter.notifyMatchFailure(op, "invalid float cast after conversion");
        }
        rewriter.replaceOp(op, converted);
        return success();
    }
};

/// Normalizes arithmetic integer cast ops after type conversion.
///
/// Removes redundant casts created by type normalization and rejects integer
/// casts whose width relationship becomes invalid under the new element types.
template <typename OpTy, typename TypeTy, typename OperandToResultWidthLegalityRelation>
struct ConvertTypeSensitiveArithCastOp : public OpConversionPattern<OpTy> {
    using OpConversionPattern<OpTy>::OpConversionPattern;
    LogicalResult matchAndRewrite(
        OpTy op, typename OpTy::Adaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {
        auto resultType = this->getTypeConverter()->convertType(op.getResult().getType());
        auto operandType = this->getTypeConverter()->convertType(op.getOperand().getType());

        auto resultEType = cast<TypeTy>(getElementTypeOrSelf(resultType));
        auto operandEType = cast<TypeTy>(getElementTypeOrSelf(operandType));
        if (resultEType == operandEType) {
            rewriter.replaceOp(op, adaptor.getOperands()[0]);
            return success();
        }

        if (!OperandToResultWidthLegalityRelation()(
                operandEType.getWidth(), resultEType.getWidth()
            )) {
            return rewriter.notifyMatchFailure(op, "invalid width combination after conversion");
        }
        rewriter.replaceOpWithNewOp<OpTy>(op, resultType, adaptor.getOperands()[0]);
        return success();
    }
};

using CastPatternInserter = function_ref<void(RewritePatternSet &, TypeConverter &, MLIRContext *)>;

/// Shared driver for whole-op primitive type conversion passes.
///
/// Applies generic cloning, cast normalization, signature rewriting, and dynamic legality.
LogicalResult runTypeConversion(
    Operation *op, TypeConverter &tc, CastPatternInserter addCasts,
    bool preserveIO = !clConvertIODType
) {
    MLIRContext *ctx = op->getContext();
    SmallVector<PreservedIOFuncInfo> preservedIO;
    ModuleOp module = dyn_cast<ModuleOp>(op);
    if (preserveIO) {
        collectPreservedIO(op, preservedIO);
    }

    RewritePatternSet patterns(ctx);
    patterns.insert<GenericTypeConversionPattern>(ctx, tc);
    addCasts(patterns, tc, ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, tc);
    populateFunctionOpInterfaceTypeConversionPattern<iree_compiler::IREE::Util::InitializerOp>(
        patterns, tc
    );
    populateFunctionOpInterfaceTypeConversionPattern<iree_compiler::IREE::Util::FuncOp>(
        patterns, tc
    );
    ConversionTarget target(*ctx);
    markAllTypesLegal(target, tc);
    LogicalResult conversionResult = applyFullConversion(op, target, std::move(patterns));
    if (failed(conversionResult) || preservedIO.empty()) {
        return conversionResult;
    }

    for (const auto &ioInfo : preservedIO) {
        func::FuncOp funcOp = module.lookupSymbol<func::FuncOp>(ioInfo.name.getValue());
        if (!funcOp) {
            return op->emitError("failed to locate function after type conversion: ")
                   << ioInfo.name;
        }
        if (failed(addBoundaryCasts(funcOp, ioInfo.originalType, tc))) {
            return failure();
        }
    }

    return success();
}

LogicalResult
runFloatConversion(Operation *op, TypeConverter &tc, bool preserveIO = !clConvertIODType) {
    return runTypeConversion(
        op, tc,
        [](RewritePatternSet &p, TypeConverter &tc, MLIRContext *ctx) {
            p.insert<
                ConvertFloatArithCastOp<arith::TruncFOp>, ConvertFloatArithCastOp<arith::ExtFOp>>(
                tc, ctx
            );
        },
        preserveIO
    );
}

template <typename SourceType, typename TargetType>
struct FloatTypeConverter : public PrimitiveTypeConverter<SourceType, TargetType> {
    explicit FloatTypeConverter() {
        this->addSourceMaterialization(convertRankedFloat);
        this->addTargetMaterialization(convertRankedFloat);
    }
};

template <typename SourceType, typename TargetType>
struct IntegerTypeConverter : public PrimitiveTypeConverter<SourceType, TargetType> {
    explicit IntegerTypeConverter() {
        this->addSourceMaterialization(convertRankedInteger);
        this->addTargetMaterialization(convertRankedInteger);
    }
};

struct ConvertDTypesBoundaryConverter : public TypeConverter {
    ConvertDTypesBoundaryConverter() {
        addConversion([](Type type) { return type; });
        addConversion([](Float16Type type) -> Type {
            return BFloat16Type::get(type.getContext());
        });
        addConversion([](Float32Type type) -> Type {
            return BFloat16Type::get(type.getContext());
        });
        addConversion([](IntegerType type) -> Type {
            // Only signless integers are converted: the arith cast ops used to materialize
            // conversions (trunci/extsi/extui) require signless operands, so signed/unsigned
            // i64 must pass through untouched to avoid emitting invalid IR.
            if (!type.isSignlessInteger(64)) {
                return type;
            }
            return IntegerType::get(type.getContext(), 32);
        });
        addConversion([&](ComplexType type) {
            return ComplexType::get(convertType(type.getElementType()));
        });
        addConversion([&](RankedTensorType type) {
            return RankedTensorType::get(
                type.getShape(), convertType(type.getElementType()), type.getEncoding()
            );
        });
        addConversion([&](VectorType type) {
            return VectorType::get(type.getShape(), convertType(type.getElementType()));
        });
        addConversion([&](iree_compiler::IREE::Util::PtrType ptrType) {
            return iree_compiler::IREE::Util::PtrType::get(convertType(ptrType.getTargetType()));
        });
        addSourceMaterialization(convertRankedFloat);
        addTargetMaterialization(convertRankedFloat);
        addSourceMaterialization(convertRankedInteger);
        addTargetMaterialization(convertRankedInteger);
    }
};

/// Demotes f32 element types to bf16 element types.
struct DemoteF32ToBF16Converter : public FloatTypeConverter<Float32Type, BFloat16Type> {
    Type getTargetType(Float32Type type) override { return BFloat16Type::get(type.getContext()); }
};

class TorqDemoteF32ToBF16Pass : public impl::TorqDemoteF32ToBF16Base<TorqDemoteF32ToBF16Pass> {
  public:
    using TorqDemoteF32ToBF16Base::TorqDemoteF32ToBF16Base;

    void runOnOperation() override {
        DemoteF32ToBF16Converter typeConverter;

        if (failed(runFloatConversion(getOperation(), typeConverter))) {
            return signalPassFailure();
        }
    }
};

/// Converts f16 element types to bf16 element types.
struct ConvertF16ToBF16Converter : public FloatTypeConverter<Float16Type, BFloat16Type> {
    Type getTargetType(Float16Type type) override { return BFloat16Type::get(type.getContext()); }
};

class TorqConvertF16ToBF16Pass : public impl::TorqConvertF16ToBF16Base<TorqConvertF16ToBF16Pass> {
  public:
    using TorqConvertF16ToBF16Base::TorqConvertF16ToBF16Base;

    void runOnOperation() override {
        ConvertF16ToBF16Converter typeConverter;

        if (failed(runFloatConversion(getOperation(), typeConverter))) {
            return signalPassFailure();
        }
    }
};

/// Demotes i64 element types to i32 element types.
struct DemoteI64ToI32Converter : public IntegerTypeConverter<IntegerType, IntegerType> {
    // Restrict to signless i64: arith trunci/extsi/extui require signless operands, so
    // converting signed/unsigned i64 would materialize invalid casts.
    bool isSourceType(IntegerType type) override { return type.isSignlessInteger(64); }
    Type getTargetType(IntegerType type) override {
        return IntegerType::get(type.getContext(), 32, type.getSignedness());
    }
};

void addIntegerCastPatterns(RewritePatternSet &p, TypeConverter &tc, MLIRContext *ctx) {
    p.insert<ConvertTypeSensitiveArithCastOp<arith::TruncIOp, IntegerType, std::greater<unsigned>>>(
        tc, ctx
    );
    p.insert<ConvertTypeSensitiveArithCastOp<arith::ExtUIOp, IntegerType, std::less<unsigned>>>(
        tc, ctx
    );
    p.insert<ConvertTypeSensitiveArithCastOp<arith::ExtSIOp, IntegerType, std::less<unsigned>>>(
        tc, ctx
    );
}

class TorqDemoteI64ToI32Pass : public impl::TorqDemoteI64ToI32Base<TorqDemoteI64ToI32Pass> {
  public:
    using TorqDemoteI64ToI32Base::TorqDemoteI64ToI32Base;

    void runOnOperation() override {
        DemoteI64ToI32Converter typeConverter;

        LogicalResult res =
            runTypeConversion(getOperation(), typeConverter, addIntegerCastPatterns);

        if (failed(res)) {
            return signalPassFailure();
        }
    }
};

class TorqConvertAllDTypesPass : public impl::TorqConvertAllDTypesBase<TorqConvertAllDTypesPass> {
  public:
    using TorqConvertAllDTypesBase::TorqConvertAllDTypesBase;

    void runOnOperation() override {
        ModuleOp module = getOperation();
        SmallVector<PreservedIOFuncInfo> preservedIO;
        if (!clConvertIODType) {
            collectPreservedIO(module, preservedIO);
        }

        ConvertF16ToBF16Converter f16ToBF16Converter;
        if (failed(runFloatConversion(module, f16ToBF16Converter, /*preserveIO=*/false))) {
            return signalPassFailure();
        }

        DemoteF32ToBF16Converter f32ToBF16Converter;
        if (failed(runFloatConversion(module, f32ToBF16Converter, /*preserveIO=*/false))) {
            return signalPassFailure();
        }

        DemoteI64ToI32Converter i64ToI32Converter;
        if (failed(runTypeConversion(
                module, i64ToI32Converter, addIntegerCastPatterns, /*preserveIO=*/false
            ))) {
            return signalPassFailure();
        }

        ConvertDTypesBoundaryConverter boundaryConverter;
        for (const auto &ioInfo : preservedIO) {
            func::FuncOp funcOp = module.lookupSymbol<func::FuncOp>(ioInfo.name.getValue());
            if (!funcOp) {
                module.emitError("failed to locate function after type conversion: ")
                    << ioInfo.name;
                return signalPassFailure();
            }
            if (failed(addBoundaryCasts(funcOp, ioInfo.originalType, boundaryConverter))) {
                return signalPassFailure();
            }
        }
    }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createTorqDemoteF32ToBF16Pass() {
    return std::make_unique<TorqDemoteF32ToBF16Pass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createTorqConvertF16ToBF16Pass() {
    return std::make_unique<TorqConvertF16ToBF16Pass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createTorqDemoteI64ToI32Pass() {
    return std::make_unique<TorqDemoteI64ToI32Pass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createTorqConvertAllDTypesPass() {
    return std::make_unique<TorqConvertAllDTypesPass>();
}

void buildTorqTypeConversionPipeline(OpPassManager &passManager) {
    if (clConvertDtypes) {
        passManager.addPass(createTorqConvertAllDTypesPass());
    }
}

} // namespace mlir::syna::torq
