// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORQHL_DIALECT_IR_TORQHLATTRS
#define TORQHL_DIALECT_IR_TORQHLATTRS

include "torq/Dialect/TorqHL/TorqHLBase.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class TorqHL_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::syna::torq_hl";
}

class TorqHL_I32EnumAttr<string name, string description, string mnemonic,
                      list<I32EnumAttrCase> cases>
    : EnumAttr<TorqHL_Dialect, TorqHL_I32Enum<name, description, cases>, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def TorqHL_MemorySpace_Host : I32EnumAttrCase<"Host", 5, "host">;
def TorqHL_MemorySpace_Itcm : I32EnumAttrCase<"Itcm", 4, "itcm">;
def TorqHL_MemorySpace_Dtcm : I32EnumAttrCase<"Dtcm", 3, "dtcm">;
def TorqHL_MemorySpace_Lram : I32EnumAttrCase<"Lram", 2, "lram">;
def TorqHL_MemorySpace_Xram : I32EnumAttrCase<"Xram", 1, "xram">;


def TorqHL_ExecutorAttr : TorqHL_I32EnumAttr<"Executor",
    "Executor of a program ", "executor", [
      I32EnumAttrCase<"Slice", 0, "slice">,
      I32EnumAttrCase<"CSS", 1, "css">,
      I32EnumAttrCase<"NSS", 2, "nss">,
      I32EnumAttrCase<"Host", 3, "host">
]>;

def TorqHL_Program : TorqHL_Type<"Program", "program"> {
  let summary = "A program that can be executed on one of the executors";
  let description = [{    
    The program can be bound to some arguments for execution on an executor.

    The program is created by torq_hl.program operation.
  }];
  let parameters = (ins "Executor":$executor);
  let assemblyFormat = "`<` $executor `>`";
}

def TorqHL_Invocation : TorqHL_Type<"Invocation", "invocation"> {
  let summary = "An invocation of a program";
  let description = [{
    Represents a specific invocation of a program on a specific executor
  }];
  let parameters = (ins "Executor":$executor);  
  let assemblyFormat = "`<` $executor `>`";
}

def TorqHL_MemorySpaceAttr :
    TorqHL_I32EnumAttr<"MemorySpace", "Memory space of a memref", "memory_space", [
      TorqHL_MemorySpace_Lram,
      TorqHL_MemorySpace_Xram,
      TorqHL_MemorySpace_Dtcm,
      TorqHL_MemorySpace_Itcm,
      TorqHL_MemorySpace_Host
    ]>;


// Represents the physical layout of a tensor in memory.
// Stored inside the 'encoding' attribute of a tensor or the memory space attribute of a memref.
// The encoding describes a backing buffer in a specified memory space, the buffer is an
// n-dimension array with specified counts and optional strides and padding.
// When the encoding is dense count and strides can be omitted
def TorqHL_TensorEncodingAttr : AttrDef<TorqHL_Dialect, "TensorEncoding", []>
{    
    let mnemonic = "enc";
    let parameters = (
        ins        
        // Memory space of the backing buffer
        "MemorySpace": $mem_space,
        // Counts used to store the tensor
        OptionalArrayRefParameter<"int64_t">: $counts,
        // Strides used to store the tensor (in elements) if not dense
        OptionalArrayRefParameter<"int64_t">: $strides,
        // Padding in at the end of the backing buffer (in elements) if not 0
        OptionalParameter<"int64_t">: $padding
    );

    let genVerifyDecl = 1;

    let cppNamespace = "::mlir::syna::torq_hl";
    let assemblyFormat = "`mem_space` `=` $mem_space ( `counts` `=` `[` $counts^ `]` )? (`strides` `=` `[` $strides^ `]` )? (`padding` `=` $padding^ )?";

}

def TorqHL_TensorEncodingRequirementsAttr : AttrDef<TorqHL_Dialect, "TensorEncodingRequirements", []>
{

    let description = [{
      The strides_align elements represents the alignment for each dimension.
      
      There are two ways in which each alignment can be expressed:

      1. As a number of tensor elements when the alignment value is positive
      2. As a number of items in the previous dimension when the alignment value is negative

      Examples:
      
      shape = [1,2,3,3], align = [0, 0, 0, 0]   generates strides: [18,9,3,1]
      shape = [1,2,3,3], align = [0, 64, 0, 0]  generates strides: [128,64,3,1]
      shape = [1,2,3,3], align = [-4, 0, 0, 0]  generates strides: [36,9,3,1]
      shape = [1,2,3,3], align = [-4, 64, 0, 0] generates strides: [256,64,3,1]
      shape = [2,2,3,3], align = [-4, 64, 0, 0] generates strides: [256,64,3,1]

      The padding align requires the total size of the backing buffer to be a multiple 
      of the padding align value.

      The only_dense flag indicates that only dense layouts are allowed that match the
      required strides.

    }];
    let mnemonic = "enc_req";
    let parameters = (
        ins        
        // Memory space of the backing buffer
        "MemorySpace": $mem_space,
        // Strides alignment requirements (in elements) if not empty
        OptionalArrayRefParameter<"int64_t">: $strides_align,
        // Padding alignment requirement (in elements) if not 0
        OptionalParameter<"int64_t">: $padding_align,
        // Only dense layout are allowed
        OptionalParameter<"bool">: $only_dense
    );

    let cppNamespace = "::mlir::syna::torq_hl";
    let assemblyFormat = "`mem_space` `=` $mem_space (`strides_align` `=` `[` $strides_align^ `]` )? (`padding_align` `=` $padding_align^)? (`only_dense` `=` $only_dense^)?";

}

def TorqHL_VectorizationModeEnum : I32EnumAttr<"VectorizationModeEnum",
    "defines the alu vectorization mode",
    [
      I32EnumAttrCase<"None", 0>,
      I32EnumAttrCase<"64x4", 4>,
      I32EnumAttrCase<"32x8", 8>,
      I32EnumAttrCase<"16x16", 16>,
      I32EnumAttrCase<"32x32", 32>
    ]> {
      let cppNamespace = "::mlir::syna::torq_hl";

}

def TorqHL_DepthToSpaceModeEnum : I32EnumAttr<"DepthToSpaceModeEnum",
    "defines the alu vectorization mode",
    [
      I32EnumAttrCase<"DCR", 0>,
      I32EnumAttrCase<"CRD", 1>,
    ]> {
      let cppNamespace = "::mlir::syna::torq_hl";

}

def TorqHL_ActConfigAttr :
  AttrDef<TorqHL_Dialect, "ActConfig", []>
{

  let mnemonic = "act_config";
  let parameters = (ins 
    "uint32_t":$shiftFactorDiv4,
    "int32_t":$outputZeroPoint,
    "int32_t":$outputMin,
    "int32_t":$outputMax
  );

  let assemblyFormat = "`<` `shiftFactorDiv4` $shiftFactorDiv4 `,` `outputZeroPoint` $outputZeroPoint `,` `outputMin` $outputMin `,` `outputMax` $outputMax `>`";

}

def TorqHL_ALUOp0ModeAttr: I32EnumAttr<"ALUOp0Mode",
    "The ALU Op0 mode",
    [
      I32EnumAttrCase<"MUL",  1>,
      I32EnumAttrCase<"MULS", 2>,
      I32EnumAttrCase<"ADD",  3>,
      I32EnumAttrCase<"SUB",  4>,
      I32EnumAttrCase<"AND",  5>,
      I32EnumAttrCase<"OR",   6>,
      I32EnumAttrCase<"XOR",  7>,
      I32EnumAttrCase<"GT",   8>,
      I32EnumAttrCase<"MAX",  9>,
      I32EnumAttrCase<"MIN",  10>,
      I32EnumAttrCase<"WBYP", 11>,
      I32EnumAttrCase<"DBYP", 12>,
    ]> {
  let cppNamespace = "::mlir::syna::torq_hl";
}

def TorqHL_ALUOp1ModeAttr: I32EnumAttr<"ALUOp1Mode",
    "The ALU Op1 mode",
    [
      I32EnumAttrCase<"ACC",  1>,
      I32EnumAttrCase<"SACC", 2>,
      I32EnumAttrCase<"MAX",  3>,
      I32EnumAttrCase<"MIN",  4>,
      I32EnumAttrCase<"MUL",  5>,
    ]> {
  let cppNamespace = "::mlir::syna::torq_hl";
}

def TorqHL_ElementwiseOpEnum : I32EnumAttr<"ElementwiseOpEnum", "Type of elementwise operation", [
  I32EnumAttrCase<"BITWISE_AND",       0>,
  I32EnumAttrCase<"BITWISE_OR",        1>,
  I32EnumAttrCase<"BITWISE_XOR",       2>,
  I32EnumAttrCase<"BITWISE_NOT",       3>,
  I32EnumAttrCase<"MINIMUM",           4>,
  I32EnumAttrCase<"MAXIMUM",           5>,
  I32EnumAttrCase<"GREATER",           6>,
  I32EnumAttrCase<"GREATER_EQUAL",     7>,
  I32EnumAttrCase<"EQUAL",             8>,
  I32EnumAttrCase<"LOGICAL_AND",       9>,
  I32EnumAttrCase<"LOGICAL_OR",        10>,
  I32EnumAttrCase<"LOGICAL_XOR",       11>,
  I32EnumAttrCase<"LOGICAL_NOT",       12>,
  I32EnumAttrCase<"SUB",               13>,
  I32EnumAttrCase<"ADD",               14>
]> {
  let cppNamespace = "::mlir::syna::torq_hl";
}

def TorqHL_ShiftModeEnum : I32EnumAttr<"ShiftModeEnum", "Type of shift operation", [
  I32EnumAttrCase<"LSR",       0>, // Logical Shift Right
  I32EnumAttrCase<"LSL",       1>, // Logical Shift Left
  I32EnumAttrCase<"ASR",       2>, // Arightmetic Shift Right
  I32EnumAttrCase<"ASL",       3>, // Arithmetic Shift Left
]> {
  let cppNamespace = "::mlir::syna::torq_hl";
}

def TorqHL_AluConfigAttr :
  AttrDef<TorqHL_Dialect, "AluConfig", []>
{

  let mnemonic = "alu_config";
  let parameters = (ins 
    "ALUOp0Mode":$op0_mode,
    "ALUOp1Mode":$op1_mode
  );

  let assemblyFormat = "`<` `op0_mode` $op0_mode `,` `op1_mode` $op1_mode `>`";

}

#endif // TORQHL_DIALECT_IR_TORQHLATTRS
