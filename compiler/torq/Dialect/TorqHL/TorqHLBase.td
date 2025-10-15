// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORQHL_DIALECT_IR_BASE
#define TORQHL_DIALECT_IR_BASE

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

def TorqHL_Dialect : Dialect {
  let name = "torq_hl";
  let cppNamespace = "::mlir::syna::torq_hl";
  let summary = "The TORQ high-level operator dialect.";
  let description = [{
    A dialect designed for expressing high-level operations for which specialized kernels exists.
  }];
  let extraClassDeclaration = [{
    void initializeTorqHLAttrs();
  }];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class TorqHL_Op<string mnemonic, list<Trait> traits = []> :
    Op<TorqHL_Dialect, mnemonic, traits>;


class TorqHL_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TorqHL_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def KernelInterface : OpInterface<"KernelInterface", [DestinationStyleOpInterface]> {
  let description = [{
    Operation that represents a TorqHL kernel.
  }];

  let cppNamespace = "::mlir::syna::torq_hl";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the encoding requirements for the kernel.",
      /*retTy=*/"::mlir::syna::torq_hl::KernelEncoding",
      /*methodName=*/"getKernelEncoding",
      /*args=*/(ins)
    >
  ];

}


#endif // TORQHL_DIALECT_IR_BASE