// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Dialect/TorqHL/GenericOp.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

#define DEBUG_TYPE "torqhl-ops"

namespace mlir::syna::torq_hl {

using RegionBuilderFn =
    llvm::function_ref<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)>;

mlir::MutableOperandRange GenericOp::getDpsInitsMutable() {
    // this should return always P (which is mandatory) and Q if present, adding will add to the Q
    // segment
    // FIXME: is this working correctly?
    return MutableOperandRange(
        getOperation(),
        (getD() ? 1 : 0) + (getW() ? 1 : 0) + (getBias() ? 1 : 0) + (getScale() ? 1 : 0),
        (getQ() ? 2 : 1)
    );
}

SmallVector<utils::IteratorType> GenericOp::getIteratorTypesArray() {
    SmallVector<utils::IteratorType> itTypes;

    for (int i = 0; i < getPMap().getNumDims(); i++) {
        if (getPMap().isFunctionOfDim(i)) {
            itTypes.push_back(utils::IteratorType::parallel);
        }
        else {
            itTypes.push_back(utils::IteratorType::reduction);
        }
    }

    return itTypes;
}

ArrayAttr GenericOp::getIndexingMaps() {

    SmallVector<AffineMap> maps;

    if (getDMap()) {
        maps.push_back(getDMap().value());
    }

    if (getWMap()) {
        maps.push_back(getWMap().value());
    }

    if (getBiasMap()) {
        maps.push_back(getBiasMap().value());
    }

    if (getScaleMap()) {
        maps.push_back(getScaleMap().value());
    }

    maps.push_back(getPMap());

    if (getQMap()) {
        maps.push_back(getQMap().value());
    }

    Builder builder(getContext());

    return builder.getAffineMapArrayAttr(maps);
}

static Value createAct(
    ImplicitLocOpBuilder &builder, Value p, Value b, Value a0, Value a1, ActConfigAttr actAttr,
    Type outputType
) {

    // setup the default values for a0, a1 and bias
    if (!a0) {
        a0 = arith::ConstantOp::create(builder, builder.getI32IntegerAttr(1));
    }

    if (!a1) {
        a1 = arith::ConstantOp::create(builder, builder.getI32IntegerAttr(1));
    }

    if (!b) {
        b = arith::ConstantOp::create(builder, builder.getI32IntegerAttr(0));
    }

    auto zeroi32 = arith::ConstantOp::create(builder, builder.getI32IntegerAttr(0));
    auto onei64 = arith::ConstantOp::create(builder, builder.getI64IntegerAttr(1));
    auto shiftFactorDiv4 =
        arith::ConstantOp::create(builder, builder.getI32IntegerAttr(actAttr.getShiftFactorDiv4()));
    auto zeroPoint =
        arith::ConstantOp::create(builder, builder.getI32IntegerAttr(actAttr.getOutputZeroPoint()));
    auto outputMax =
        arith::ConstantOp::create(builder, builder.getI32IntegerAttr(actAttr.getOutputMax()));
    auto outputMin =
        arith::ConstantOp::create(builder, builder.getI32IntegerAttr(actAttr.getOutputMin()));

    // t0 = p + b;
    auto t0 = arith::AddIOp::create(builder, p, b);

    // t1 = (t0<0) ? ((int64_t)t0)*a0 : ((int64_t)t0)*a1;
    auto isT0Negative = arith::CmpIOp::create(builder, arith::CmpIPredicate::slt, t0, zeroi32);
    auto t0i64 = arith::ExtSIOp::create(builder, builder.getI64Type(), t0);

    auto a0i64 = arith::ExtSIOp::create(builder, builder.getI64Type(), a0);
    auto a1i64 = arith::ExtSIOp::create(builder, builder.getI64Type(), a1);
    auto t1a0 = arith::MulIOp::create(builder, a0i64, t0i64);
    auto t1a1 = arith::MulIOp::create(builder, a1i64, t0i64);
    auto t1 = arith::SelectOp::create(builder, isT0Negative, t1a0, t1a1);

    // t2 = t1 + (1LL << (shift * 4) >> 1);      // rounding
    auto fouri32 = arith::ConstantOp::create(builder, builder.getI32IntegerAttr(4));
    auto shiftTimes4 = arith::MulIOp::create(builder, shiftFactorDiv4, fouri32);
    auto shiftTimes4i64 = arith::ExtSIOp::create(builder, builder.getI64Type(), shiftTimes4);
    auto shiftTimes4i64lshift = arith::ShLIOp::create(builder, onei64, shiftTimes4i64);
    auto shiftTimes4i64rshift = arith::ShRSIOp::create(builder, shiftTimes4i64lshift, onei64);
    auto t2 = arith::AddIOp::create(builder, t1, shiftTimes4i64rshift);

    // t3 = t2>>(4*shift);
    auto t3 = arith::ShRSIOp::create(builder, t2, shiftTimes4i64);

    // t4 = t3 + zp;
    auto zpi64 = arith::ExtSIOp::create(builder, builder.getI64Type(), zeroPoint);
    auto t4 = arith::AddIOp::create(builder, t3, zpi64);

    // ((x_)<(min_)?(min_):(x_)>(max_)?(max_):(x_))
    // q = CLIP3(min, max, t4);
    auto maxi64 = arith::ExtSIOp::create(builder, builder.getI64Type(), outputMax);
    auto mini64 = arith::ExtSIOp::create(builder, builder.getI64Type(), outputMin);
    auto minQ = arith::MinSIOp::create(builder, t4, maxi64);
    auto qi64 = arith::MaxSIOp::create(builder, minQ, mini64);

    auto q = arith::TruncIOp::create(builder, builder.getI32Type(), qi64);

    //  add a final i32 to i8 truncation if needed
    if (outputType == builder.getI8Type()) {
        q = arith::TruncIOp::create(builder, builder.getI8Type(), q);
    }

    return q;
}

void GenericOp::regionBuilder(
    ImplicitLocOpBuilder &builder, Block &block, ArrayRef<NamedAttribute> attrs
) {
    SmallVector<Value> values;

    builder.getInsertionBlock();

    int hasD = 0;
    int hasW = 0;
    int hasBias = 0;
    int hasScale = 0;
    int hasQ = 0;

    AluConfigAttr aluConfig;
    ActConfigAttr actConfig;

    for (NamedAttribute attr : attrs) {
        if (attr.getName() == "d_map") {
            hasD = 1;
        }
        else if (attr.getName() == "w_map") {
            hasW = 1;
        }
        else if (attr.getName() == "bias_map") {
            hasBias = 1;
        }
        else if (attr.getName() == "scale_map") {
            hasScale = 1;
        }
        else if (attr.getName() == "q_map") {
            hasQ = 1;
        }
        else if (attr.getName() == "alu_config") {
            aluConfig = cast<AluConfigAttr>(attr.getValue());
        }
        else if (attr.getName() == "act_config") {
            actConfig = cast<ActConfigAttr>(attr.getValue());
        }
    }

    BlockArgument dArg;

    if (hasD)
        dArg = block.getArgument(0);

    BlockArgument wArg;

    if (hasW)
        wArg = block.getArgument(hasD);

    BlockArgument biasArg;

    if (hasBias)
        biasArg = block.getArgument(hasD + hasW);

    BlockArgument scaleArg;

    if (hasScale)
        scaleArg = block.getArgument(hasD + hasW + hasBias);

    BlockArgument pArg = block.getArgument(hasD + hasW + hasBias + hasScale);

    BlockArgument qArg;

    if (hasQ)
        qArg = block.getArgument(hasD + hasW + hasBias + hasScale + 1);

    Value pOut;

    Value actOut;

    if (aluConfig) {

        Value extD;

        if (aluConfig.getOp0Mode() != ALUOp0Mode::WBYP) {

            assert(dArg && "Missing d argument for not ALUOp0Mode::WBYP");

            if (dArg.getType() == builder.getI8Type()) {
                extD = arith::ExtSIOp::create(builder, builder.getI32Type(), dArg);
            }
            else {
                assert(
                    aluConfig.getOp0Mode() == ALUOp0Mode::DBYP && "Expected i8 type for d argument"
                );
                extD = dArg;
            }
        }

        Value extW;

        if (aluConfig.getOp0Mode() != ALUOp0Mode::DBYP) {

            assert(dArg && "Missing w argument for not ALUOp0Mode::DBYP");

            if (wArg.getType() == builder.getI8Type()) {
                extW = arith::ExtSIOp::create(builder, builder.getI32Type(), wArg);
            }
            else {
                assert(
                    aluConfig.getOp0Mode() == ALUOp0Mode::WBYP && "Expected i8 type for w argument"
                );
                extW = wArg;
            }
        }

        Value combineDW;

        if (aluConfig.getOp0Mode() == ALUOp0Mode::ADD) {
            combineDW = arith::AddIOp::create(builder, extD, extW);
        }
        else if (aluConfig.getOp0Mode() == ALUOp0Mode::MUL) {
            combineDW = arith::MulIOp::create(builder, extD, extW);
        }
        else if (aluConfig.getOp0Mode() == ALUOp0Mode::DBYP) {
            combineDW = extD;
        }
        else if (aluConfig.getOp0Mode() == ALUOp0Mode::WBYP) {
            combineDW = extW;
        }
        else {
            assert(false && "Unsupported ALUOpMode0");
        }

        if (aluConfig.getOp1Mode() == ALUOp1Mode::ACC) {
            pOut = arith::AddIOp::create(builder, combineDW, pArg);
        }
        else {
            assert(false && "Unsupported ALUOpMode1");
        }
    }
    else {
        pOut = pArg;
    }

    if (actConfig) {
        actOut = createAct(builder, pOut, biasArg, scaleArg, scaleArg, actConfig, qArg.getType());
    }
    else {
        actOut = pOut;
    }

    SmallVector<Value> yieldValues;

    yieldValues.push_back(pOut);

    if (hasQ)
        yieldValues.push_back(actOut);

    linalg::YieldOp::create(builder, yieldValues);
}

std::string GenericOp::getLibraryCallName() {
    return linalg::generateLibraryCallName(getOperation());
}

void GenericOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects
) {

    if (hasPureTensorSemantics())
        return;

    for (auto [index, operand] : llvm::enumerate(getDpsInputs())) {
        if (!llvm::isa<MemRefType>(operand.getType()))
            continue;
        effects.emplace_back(
            MemoryEffects::Read::get(), &(getOperation()->getOpOperand(index)), /*stage=*/0,
            /*effectOnFullRegion=*/true, SideEffects::DefaultResource::get()
        );
    }

    for (OpOperand &operand : getDpsInitsMutable()) {
        if (!llvm::isa<MemRefType>(operand.get().getType()))
            continue;
        if (payloadUsesValueFromOperand(&operand)) {
            effects.emplace_back(
                MemoryEffects::Read::get(), &operand, /*stage=*/0,
                /*effectOnFullRegion=*/true, SideEffects::DefaultResource::get()
            );
        }
        effects.emplace_back(
            MemoryEffects::Write::get(), &operand, /*stage=*/0,
            /*effectOnFullRegion=*/true, SideEffects::DefaultResource::get()
        );
    }
}

static void buildTorqGenericOp(
    OpBuilder &builder, OperationState &state, GenericOpConfig &config,
    ArrayRef<NamedAttribute> attributes, RegionBuilderFn regionBuilder
) {
    assert(config.p && "p must be always set");

    SmallVector<Type> types;
    types.push_back(config.p.value().getType());

    if (config.q)
        types.push_back(config.q.value().getType());

    if (!isa<MemRefType>(config.p.value().getType())) {
        state.addTypes(types);
    }

    SmallVector<Value> operands;
    if (config.d) {

        assert(isa<ShapedType>(config.d.value().getType()));
        operands.push_back(config.d.value());
        state.addAttribute(GenericOp::getDMapAttrName(state.name), config.d.map());
    }

    if (config.w) {
        operands.push_back(config.w.value());
        state.addAttribute(GenericOp::getWMapAttrName(state.name), config.w.map());
    }

    if (config.bias) {
        operands.push_back(config.bias.value());
        state.addAttribute(GenericOp::getBiasMapAttrName(state.name), config.bias.map());
    }

    if (config.scale) {
        operands.push_back(config.scale.value());
        state.addAttribute(GenericOp::getScaleMapAttrName(state.name), config.scale.map());
    }

    operands.push_back(config.p.value());
    state.addAttribute(GenericOp::getPMapAttrName(state.name), config.p.map());

    if (config.q) {
        operands.push_back(config.q.value());
        state.addAttribute(GenericOp::getQMapAttrName(state.name), config.q.map());
    }

    state.addOperands(operands);

    state.addAttributes(attributes);
    state.addAttribute(
        GenericOp::getOperandSegmentSizesAttrName(state.name),
        builder.getDenseI32ArrayAttr(
            {config.d ? 1 : 0, config.w ? 1 : 0, config.bias ? 1 : 0, config.scale ? 1 : 0, 1,
             config.q ? 1 : 0}
        )
    );

    if (config.actConfig)
        state.addAttribute(GenericOp::getActConfigAttrName(state.name), config.actConfig);

    if (config.aluConfig)
        state.addAttribute(GenericOp::getAluConfigAttrName(state.name), config.aluConfig);

    Region &region = *state.addRegion();

    SmallVector<Type, 8> argTypes;
    SmallVector<Location, 8> argLocs;
    for (auto containers : {TypeRange(operands)}) {
        for (auto t : containers) {
            argTypes.push_back(isa<MemRefType, RankedTensorType>(t) ? getElementTypeOrSelf(t) : t);

            // TODO: Pass in a proper location here.
            argLocs.push_back(builder.getUnknownLoc());
        }
    }

    // RAII.
    OpBuilder::InsertionGuard guard(builder);
    Block *body = builder.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);

    builder.setInsertionPointToStart(body);
    ImplicitLocOpBuilder b(builder.getUnknownLoc(), builder);
    regionBuilder(b, *body, state.attributes.getAttrs());
}

GenericOpConfig GenericOpConfig::fromOperation(Operation *op) {
    auto genOp = cast<GenericOp>(op);

    GenericOpConfig config;

    if (genOp.getD()) {
        config.d = GenericOpParam(genOp.getD(), genOp.getDMapAttr());
    }

    if (genOp.getW()) {
        config.w = GenericOpParam(genOp.getW(), genOp.getWMapAttr());
    }

    if (genOp.getBias()) {
        config.bias = GenericOpParam(genOp.getBias(), genOp.getBiasMapAttr());
    }

    if (genOp.getScale()) {
        config.scale = GenericOpParam(genOp.getScale(), genOp.getScaleMapAttr());
    }

    if (genOp.getQ()) {
        config.q = GenericOpParam(genOp.getQ(), genOp.getQMapAttr());
    }

    config.p = GenericOpParam(genOp.getP(), genOp.getQMapAttr());

    config.aluConfig = genOp.getAluConfigAttr();
    config.actConfig = genOp.getActConfigAttr();

    return config;
}

bool usedOnlyAsPValue(Value value) {

    for (auto &use : value.getUses()) {

        if (auto genericOp = dyn_cast<torq_hl::GenericOp>(use.getOwner())) {

            // FIXME: do not use an hardcoded value
            auto pIdx = genericOp.getODSOperandIndexAndLength(4).first;

            if (use.getOperandNumber() == pIdx) {
                continue;
            }
        }

        return false;
    }

    return true;
}

GenericOpParam getParamFromAdaptor(OpOperand *opOperand, linalg::GenericOp::Adaptor &adaptor) {

    Value value = adaptor.getOperands()[opOperand->getOperandNumber()];
    AffineMapAttr attr =
        AffineMapAttr::get(cast<linalg::GenericOp>(opOperand->getOwner())
                               .getIndexingMapsArray()[opOperand->getOperandNumber()]);

    return torq_hl::GenericOpParam(value, attr);
}

} // namespace mlir::syna::torq_hl

#define GET_OP_CLASSES
#include "torq/Dialect/TorqHL/GenericOp.cpp.inc"
