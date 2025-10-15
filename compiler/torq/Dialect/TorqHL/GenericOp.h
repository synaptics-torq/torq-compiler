// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Dialect/TorqHL/TorqHLTraits.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::syna::torq_hl {

class GenericOpParam {

    Value _value;
    AffineMapAttr _map;

  public:
    GenericOpParam() : _value(nullptr), _map(nullptr) {}

    GenericOpParam(Value value, AffineMapAttr map) : _value(value), _map(map) {
        assert(!(_value != nullptr && _map == nullptr) && !(_value == nullptr && _map != nullptr));
    }

    GenericOpParam(Value value, AffineMap map) : _value(value), _map(AffineMapAttr::get(map)) {
        assert(!(_value != nullptr && _map == nullptr) && !(_value == nullptr && _map != nullptr));
    }

    Value value() { return _value; }
    AffineMapAttr map() { return _map; }

    bool empty() const { return !_value; }

    explicit operator bool() const { return !empty(); }
};

struct GenericOpConfig {
    GenericOpParam p;
    GenericOpParam d;
    GenericOpParam w;
    GenericOpParam bias;
    GenericOpParam scale;
    GenericOpParam q;
    AluConfigAttr aluConfig;
    ActConfigAttr actConfig;

    static GenericOpConfig fromOperation(Operation *op);
};

bool usedOnlyAsPValue(Value value);

torq_hl::GenericOpParam
getParamFromAdaptor(OpOperand *opOperand, mlir::linalg::GenericOp::Adaptor &adaptor);

} // namespace mlir::syna::torq_hl

#define GET_OP_CLASSES
#include "torq/Dialect/TorqHL/GenericOp.h.inc"
