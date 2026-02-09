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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
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
        return builder.create<arith::TruncFOp>(loc, type, inputs[0]);
    }

    if (inputElementType.getIntOrFloatBitWidth() < elementType.getIntOrFloatBitWidth()) {
        return builder.create<arith::ExtFOp>(loc, type, inputs[0]);
    }

    /// MLIR does not have a direct way to cast f16 <-> bf16 so we use an f32 bridge
    if ((isa<Float16Type>(inputElementType) && isa<BFloat16Type>(elementType)) ||
        (isa<BFloat16Type>(inputElementType) && isa<Float16Type>(elementType))) {
        Type f32ElementType = Float32Type::get(builder.getContext());
        Type f32Type = cloneTypeWithElementType(type, f32ElementType);
        Value extended = builder.create<arith::ExtFOp>(loc, f32Type, inputs[0]);
        return builder.create<arith::TruncFOp>(loc, type, extended);
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
        return builder.create<arith::TruncIOp>(loc, type, inputs[0]);
    }
    if (inWidth < outWidth) {
        if (isUnsigned)
            return builder.create<arith::ExtUIOp>(loc, type, inputs[0]);
        else
            return builder.create<arith::ExtSIOp>(loc, type, inputs[0]);
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
    for (auto [idx, arg] : llvm::enumerate(newEntry->getArguments())) {
        Type expectedType = convertedType.getInput(idx);
        Value mapped = arg;
        if (mapped.getType() != expectedType) {
            mapped = tc.materializeTargetConversion(builder, arg.getLoc(), expectedType, arg);
            if (!mapped) {
                funcOp.emitOpError("failed to materialize input boundary cast at index ") << idx;
                return failure();
            }
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
        SmallVector<NamedAttribute> newAttrs;
        if (op->hasTrait<OpTrait::ConstantLike>() ||
            isa<iree_compiler::IREE::Util::GlobalOpInterface>(op)) {
            for (auto attr : op->getAttrs()) {
                auto newAttr = iree_compiler::convertAttribute(
                    op->getLoc(), attr.getValue(), *getTypeConverter()
                );
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
/// Removes redundant casts created by type normalization and rejects casts whose
/// width relationship becomes invalid under the new element types.
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
        rewriter.replaceOpWithNewOp<OpTy>(op, resultType, op.getOperand());
        return success();
    }
};

using CastPatternInserter = function_ref<void(RewritePatternSet &, TypeConverter &, MLIRContext *)>;

/// Shared driver for whole-op primitive type conversion passes.
///
/// Applies generic cloning, cast normalization, signature rewriting, and dynamic legality.
LogicalResult runTypeConversion(Operation *op, TypeConverter &tc, CastPatternInserter addCasts) {
    MLIRContext *ctx = op->getContext();
    SmallVector<PreservedIOFuncInfo> preservedIO;
    ModuleOp module = dyn_cast<ModuleOp>(op);
    if (!clConvertIODType && module) {
        for (auto funcOp : module.getOps<func::FuncOp>()) {
            if (!isTopLevelFunc(funcOp, op))
                continue;
            preservedIO.push_back({funcOp.getSymNameAttr(), funcOp.getFunctionType()});
        }
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

LogicalResult runFloatConversion(Operation *op, TypeConverter &tc) {
    return runTypeConversion(op, tc, [](RewritePatternSet &p, TypeConverter &tc, MLIRContext *ctx) {
        p.insert<
            ConvertTypeSensitiveArithCastOp<arith::TruncFOp, FloatType, std::greater<unsigned>>>(
            tc, ctx
        );
        p.insert<ConvertTypeSensitiveArithCastOp<arith::ExtFOp, FloatType, std::less<unsigned>>>(
            tc, ctx
        );
    });
}

template <typename SourceType, typename TargetType>
struct FloatTypeConverter : public PrimitiveTypeConverter<SourceType, TargetType> {
    explicit FloatTypeConverter() {
        this->addArgumentMaterialization(convertRankedFloat);
        this->addSourceMaterialization(convertRankedFloat);
        this->addTargetMaterialization(convertRankedFloat);
    }
};

template <typename SourceType, typename TargetType>
struct IntegerTypeConverter : public PrimitiveTypeConverter<SourceType, TargetType> {
    explicit IntegerTypeConverter() {
        this->addArgumentMaterialization(convertRankedInteger);
        this->addSourceMaterialization(convertRankedInteger);
        this->addTargetMaterialization(convertRankedInteger);
    }
};

/// Demotes f32 element types to bf16 element types.
struct DemoteF32ToBF16Converter : public FloatTypeConverter<Float32Type, BFloat16Type> {
    Type getTargetType(Float32Type type) override { return BFloat16Type::get(type.getContext()); }
};

class TorqDemoteF32ToBF16Pass : public TorqDemoteF32ToBF16Base<TorqDemoteF32ToBF16Pass> {
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

class TorqConvertF16ToBF16Pass : public TorqConvertF16ToBF16Base<TorqConvertF16ToBF16Pass> {
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
    bool isSourceType(IntegerType type) override { return type.isInteger(64); }
    Type getTargetType(IntegerType type) override {
        return IntegerType::get(type.getContext(), 32, type.getSignedness());
    }
};

class TorqDemoteI64ToI32Pass : public TorqDemoteI64ToI32Base<TorqDemoteI64ToI32Pass> {
  public:
    using TorqDemoteI64ToI32Base::TorqDemoteI64ToI32Base;

    void runOnOperation() override {
        DemoteI64ToI32Converter typeConverter;

        LogicalResult res = runTypeConversion(
            getOperation(), typeConverter,
            [](RewritePatternSet &p, auto &tc, MLIRContext *ctx) {
                p.insert<ConvertTypeSensitiveArithCastOp<
                    arith::TruncIOp, IntegerType, std::greater<unsigned>>>(tc, ctx);
                p.insert<ConvertTypeSensitiveArithCastOp<
                    arith::ExtUIOp, IntegerType, std::less<unsigned>>>(tc, ctx);
                p.insert<ConvertTypeSensitiveArithCastOp<
                    arith::ExtSIOp, IntegerType, std::less<unsigned>>>(tc, ctx);
            }
        );

        if (failed(res)) {
            return signalPassFailure();
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

void buildTorqTypeConversionPipeline(OpPassManager &passManager) {
    if (!clConvertDtypes) {
        return;
    }
    passManager.addPass(createTorqDemoteF32ToBF16Pass());
    passManager.addPass(createTorqConvertF16ToBF16Pass());
    passManager.addPass(createTorqDemoteI64ToI32Pass());
}

} // namespace mlir::syna::torq