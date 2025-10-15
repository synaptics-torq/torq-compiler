// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORQHW_IR_BASE
#define TORQHW_IR_BASE

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def TorqHW_Dialect : Dialect {
  let name = "torq_hw";
  let cppNamespace = "::mlir::syna::torq_hw";
  let summary = "Dialect that represent torq assembly code";
  let description = [{
    The TorqHW dialect represents in MLIR the concepts provided by the low level
    torq_api assembler library. The dialect is a one-to-one representation of th
    API.
  }];
  let extraClassDeclaration = [{
    void initializeTorqHWAttrs();
  }];
  let useDefaultAttributePrinterParser = 1;
}

class TorqHW_Op<string mnemonic, list<Trait> traits = []> :
    Op<TorqHW_Dialect, mnemonic, traits>;

#endif // TORQHW_IR_BASE