// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORQHW_IR_TORQHWATTRS
#define TORQHW_IR_TORQHWATTRS

include "torq/Dialect/TorqHW/TorqHWBase.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class I32EnumAttrCaseWithDoc<string sym, int val, string doc> : I32EnumAttrCase<sym, val>;


def TorqHW_ACTModeAttr: I32EnumAttr<"ACTMode",
    "The ACT mode",
    [
      I32EnumAttrCaseWithDoc<"ACT", 0, "Standard mode">,
      I32EnumAttrCaseWithDoc<"ABS", 1, "q=abs(p)">,
      I32EnumAttrCaseWithDoc<"NEG", 2, "q=-p">,
      I32EnumAttrCaseWithDoc<"CLZ", 3, "q=count_leading_zeros(p)">,
      I32EnumAttrCaseWithDoc<"CEL", 4, "q=ceiling(p)">,
      I32EnumAttrCaseWithDoc<"FLR", 5, "q=floor(p)">,
      I32EnumAttrCaseWithDoc<"LSL", 6, "q=p<<s>">,
      I32EnumAttrCaseWithDoc<"LSR", 7, "q=(uint)p>>s>">,
      I32EnumAttrCaseWithDoc<"ASR", 8, "q=(int)p>>s>">,
      I32EnumAttrCaseWithDoc<"I2F", 9, "q=(float)p>">,
      I32EnumAttrCaseWithDoc<"F2I", 10, "q=(int)p>">
    ]> {
  let cppNamespace = "::mlir::syna::torq_hw";
}

def TorqHW_ALUOp0ModeAttr: I32EnumAttr<"ALUOp0Mode",
    "The ALU Op0 mode",
    [
      I32EnumAttrCaseWithDoc<"MUL",   1, "x=D*W">,
      I32EnumAttrCaseWithDoc<"DBYP", 12, "x=D">,
    ]> {
  let cppNamespace = "::mlir::syna::torq_hw";
}

def TorqHW_ALUOp1ModeAttr: I32EnumAttr<"ALUOp1Mode",
    "The ALU Op1 mode",
    [
      I32EnumAttrCaseWithDoc<"ACC",  1, "P=P+x">,
      I32EnumAttrCaseWithDoc<"SACC", 2, "P=(P<<8)+x">,
      I32EnumAttrCaseWithDoc<"AMAX",  3, "">,
      I32EnumAttrCaseWithDoc<"AMIN",  4, "">,
      I32EnumAttrCaseWithDoc<"MAX",  5, "">,
      I32EnumAttrCaseWithDoc<"MIN",  6, "">,
      I32EnumAttrCaseWithDoc<"GT",  7, "">,
      I32EnumAttrCaseWithDoc<"GE",  8, "">,
      I32EnumAttrCaseWithDoc<"EQ",  9, "">,
      I32EnumAttrCaseWithDoc<"OR",  10, "">,
      I32EnumAttrCaseWithDoc<"AND",  11, "">,
      I32EnumAttrCaseWithDoc<"XOR",  12, "">,
      I32EnumAttrCaseWithDoc<"NOT",  13, "">,
      I32EnumAttrCaseWithDoc<"BOR",  14, "">,
      I32EnumAttrCaseWithDoc<"BAND", 15, "">,
      I32EnumAttrCaseWithDoc<"BXOR", 16, "reserved">,
      I32EnumAttrCaseWithDoc<"BNOT", 17, "reserved">,
      I32EnumAttrCaseWithDoc<"BYP", 18, "">,
      I32EnumAttrCaseWithDoc<"MUL", 19, "">,
      I32EnumAttrCaseWithDoc<"SEL", 20, "">

    ]> {
  let cppNamespace = "::mlir::syna::torq_hw";
}

def TorqHW_RegDimTagAttr : I32EnumAttr<"RegDimTag",
    "Dimension tag for register-based NDL descriptor",
    [
      I32EnumAttrCaseWithDoc<"I", 0, "Duplicate the same byte (bit-wise)">,
      I32EnumAttrCaseWithDoc<"B", 1, "Byte loop in Data (bit-wise)">,
      I32EnumAttrCaseWithDoc<"J", 2, "Duplicate the same Data (bit-wise)">,
      I32EnumAttrCaseWithDoc<"D", 3, "Data loop in Data Group (bit-wise)">,
      I32EnumAttrCaseWithDoc<"K", 4, "Duplicate the same Data (bit-wise)">,
      I32EnumAttrCaseWithDoc<"G", 5, "Data Group loop in Step (bit-wise)">,
      I32EnumAttrCaseWithDoc<"L", 6, "Repeat the same Step (cycle-wise), not implemented in hardware">,
      I32EnumAttrCaseWithDoc<"S", 7, "Step loop in Word (cycle-wise)">,
      I32EnumAttrCaseWithDoc<"M", 8, "Repeat the same Word (cycle-wise)">,
      I32EnumAttrCaseWithDoc<"W", 9, "Word loop in Block">,
      I32EnumAttrCaseWithDoc<"N", 10, "Repeat the same Block (cycle-wise)">,
      I32EnumAttrCaseWithDoc<"T", 11, "Block loop in Task">
    ]> {
  let cppNamespace = "::mlir::syna::torq_hw";
}

def TorqHW_MemDimTagAttr : I32EnumAttr<"MemDimTag",
    "Dimension tag for memory-based NDL descriptor",
    [
      I32EnumAttrCaseWithDoc<"X", 0, "Horizontal index in output tensor">,
      I32EnumAttrCaseWithDoc<"Y", 1, "Vertical index in output tensor">,
      I32EnumAttrCaseWithDoc<"A", 2, "Linear index on flattened X-Y plane of output tensor (a = y*yn+x)">,
      I32EnumAttrCaseWithDoc<"I", 3, "Horizontal index in kernel">,
      I32EnumAttrCaseWithDoc<"J", 4, "Vertical index in kernel">,
      I32EnumAttrCaseWithDoc<"U", 5, "Channel index within a channel group of input tensor">,
      I32EnumAttrCaseWithDoc<"V", 6, "Channel index within a channel group of output tensor">,
      I32EnumAttrCaseWithDoc<"G", 7, "Channel group index of both input and output tensors">,
      I32EnumAttrCaseWithDoc<"O", 8, "Other index">,
      I32EnumAttrCaseWithDoc<"B", 9, "Byte loop">,
      I32EnumAttrCaseWithDoc<"D", 10, "Data loop">
    ]> {
  let cppNamespace = "::mlir::syna::torq_hw";
}

def TorqHW_DimTypeAttr : I32EnumAttr<"DimType",
    "The memory dimension type",
    [
      I32EnumAttrCaseWithDoc<"L", 0, "Low Dimension">,
      I32EnumAttrCaseWithDoc<"H", 1, "High Dimension">,
      I32EnumAttrCaseWithDoc<"S", 2, "Special Dimension">
    ]> {
    let cppNamespace = "::mlir::syna::torq_hw";
}

def TorqHW_RoundingModeAttr : I32EnumAttr<"RoundingMode",
    "ACT Rounding mode",
    [
      I32EnumAttrCaseWithDoc<"OFF", 1, "No rounding">,
      I32EnumAttrCaseWithDoc<"NTE", 2, "Round to nearest, tie towards even">,
      I32EnumAttrCaseWithDoc<"NTO", 3, "(Unsupported) Round to nearest, tie towards odd">,
      I32EnumAttrCaseWithDoc<"NTP", 4, "Round to nearest, tie towards positive infinity">,
      I32EnumAttrCaseWithDoc<"NTN", 5, "(Unsupported) Round to nearest, tie towards negative infinity">,
      I32EnumAttrCaseWithDoc<"NTI", 6, "(Unsupported) Round to nearest, tie towards infinity(away from zero)">,
      I32EnumAttrCaseWithDoc<"NTZ", 7, "(Unsupported) Round to nearest, tie towards zero">,
      I32EnumAttrCaseWithDoc<"DBL", 8, "(Not recommended) Double rounding as defined in TOSA">,
    ]> {
    let cppNamespace = "::mlir::syna::torq_hw";
}

def TorqHW_WeightFormatAttr : I32EnumAttr<"WeightFormat",
    "The weight format, for weight decompression only, NOT for ALU",
    [
      I32EnumAttrCaseWithDoc<"SI", 1, "Signed integer for 2,4,6-bit integer">,
      I32EnumAttrCaseWithDoc<"UI", 2, "Unsigned integer for 2,4,6-bit integer">,
      I32EnumAttrCaseWithDoc<"FP", 3, "Reserved for future FP4, FP8 support">,
      I32EnumAttrCaseWithDoc<"BF", 4, "Reserved">,
      I32EnumAttrCaseWithDoc<"NF", 5, "Reserved for future NF4 support">,
    ]> {
    let cppNamespace = "::mlir::syna::torq_hw";
}

def TorqHW_NumberFormatAttr : I32EnumAttr<"NumberFormat",
    "For ALU and ACT",
    [
      I32EnumAttrCaseWithDoc<"I", 1, "Integer">,
      I32EnumAttrCaseWithDoc<"BF", 2, "BFloat">,
    ]> {
    let cppNamespace = "::mlir::syna::torq_hw";
}

def TorqHW_SliceCFGAttr :
  AttrDef<TorqHW_Dialect, "SliceCFG", []>
{
  let mnemonic = "slice_cfg";
  let parameters = (ins
    ArrayRefParameter<"ALUOp0Mode", "ALU Op0 mode">:$alu_op0_mode,
    ArrayRefParameter<"ALUOp1Mode", "ALU Op1 mode">:$alu_op1_mode,
    AttrParameter<"uint32_t", "">:$alu_d_unsigned,
    AttrParameter<"uint32_t", "">:$alu_w_unsigned,

    AttrParameter<"ACTMode", "">:$act_mode,

    // Left shift amount for every group of 4 adjacent int32 data from P-bus: {0,0,0,0}, {0,8,0,8}, {0,8,8,16}, {0,8,16,24}
    OptionalArrayRefParameter<"uint32_t", "">:$act_lsh,
    OptionalParameter<"int32_t", "">:$act_rsh,

    OptionalParameter<"int32_t", "">:$act_clip_min,
    OptionalParameter<"int32_t", "">:$act_clip_max,
    OptionalParameter<"int32_t", "">:$act_zero_point,

    OptionalParameter<"uint32_t", "">:$no_p_clear,
    OptionalParameter<"uint32_t", "">:$no_p_output,

    OptionalParameter<"uint32_t", "">:$kernel_left,
    OptionalParameter<"uint32_t", "">:$kernel_right,
    OptionalParameter<"uint32_t", "">:$kernel_top,
    OptionalParameter<"uint32_t", "">:$kernel_bottom,

    OptionalParameter<"int32_t", "">:$pad_left,
    OptionalParameter<"int32_t", "">:$pad_right,
    OptionalParameter<"int32_t", "">:$pad_top,
    OptionalParameter<"int32_t", "">:$pad_bottom,
    OptionalParameter<"int32_t", "">:$pad_value,

    OptionalParameter<"int32_t", "1">:$stride,
    OptionalParameter<"int32_t", "0">:$stride_offset,
    
    OptionalParameter<"RoundingMode", "">:$act_round_mode,
    OptionalParameter<"WeightFormat", "">:$weight_format,
    OptionalParameter<"uint32_t", "">:$alu_disable,
    OptionalParameter<"uint32_t", "">:$act_disable,
    OptionalParameter<"NumberFormat", "">:$alu_format,
    OptionalParameter<"NumberFormat", "">:$act_format,
    OptionalParameter<"uint32_t", "">:$act_sum_bits,
    OptionalArrayRefParameter<"int32_t","">:$table
  );

  // default attr builder includes all optional parameters
  // specific attr builders are defined below
  let builders = [
    AttrBuilder<(ins
        "ArrayRef<ALUOp0Mode>":$alu_op0_mode,
        "ArrayRef<ALUOp1Mode>":$alu_op1_mode,
        "uint32_t":$alu_d_unsigned,
        "uint32_t":$alu_w_unsigned,
        "ACTMode":$act_mode,
        "ArrayRef<uint32_t>":$act_lsh,
        CArg<"int32_t", "0">:$act_rsh,
        CArg<"int32_t", "0">:$act_clip_min,
        CArg<"int32_t", "0">:$act_clip_max,
        CArg<"int32_t", "0">:$act_zero_point,
        CArg<"uint32_t", "0">:$no_p_clear,
        CArg<"uint32_t", "0">:$no_p_output,
        CArg<"uint32_t", "0">:$kernel_left,
        CArg<"uint32_t", "0">:$kernel_right,
        CArg<"uint32_t", "0">:$kernel_top,
        CArg<"uint32_t", "0">:$kernel_bottom,
        CArg<"int32_t", "0">:$pad_left,
        CArg<"int32_t", "0">:$pad_right,
        CArg<"int32_t", "0">:$pad_top,
        CArg<"int32_t", "0">:$pad_bottom,
        CArg<"int32_t", "0">:$pad_value,
        CArg<"int32_t", "1">:$stride,
        CArg<"int32_t", "0">:$stride_offset,
        CArg<"uint32_t", "32">:$act_sum_bits,
        CArg<"ArrayRef<int32_t>", "{}">:$table
        )>,

    AttrBuilder<(ins
        "ArrayRef<ALUOp0Mode>":$alu_op0_mode,
        "ArrayRef<ALUOp1Mode>":$alu_op1_mode,
        "uint32_t":$alu_d_unsigned,
        "uint32_t":$alu_w_unsigned,
        "ACTMode":$act_mode,
        "ArrayRef<uint32_t>":$act_lsh,
        CArg<"int32_t", "0">:$act_rsh,
        CArg<"int32_t", "0">:$act_clip_min,
        CArg<"int32_t", "0">:$act_clip_max,
        CArg<"int32_t", "0">:$act_zero_point,
        CArg<"uint32_t", "0">:$no_p_clear,
        CArg<"uint32_t", "0">:$no_p_output,
        CArg<"uint32_t", "0">:$kernel_left,
        CArg<"uint32_t", "0">:$kernel_right,
        CArg<"uint32_t", "0">:$kernel_top,
        CArg<"uint32_t", "0">:$kernel_bottom,
        CArg<"int32_t", "0">:$pad_left,
        CArg<"int32_t", "0">:$pad_right,
        CArg<"int32_t", "0">:$pad_top,
        CArg<"int32_t", "0">:$pad_bottom,
        CArg<"int32_t", "0">:$pad_value,
        CArg<"int32_t", "1">:$stride
        )>,

    AttrBuilder<(ins
        "ArrayRef<ALUOp0Mode>":$alu_op0_mode,
        "ArrayRef<ALUOp1Mode>":$alu_op1_mode,
        "uint32_t":$alu_d_unsigned,
        "uint32_t":$alu_w_unsigned,
        "ACTMode":$act_mode,
        "ArrayRef<uint32_t>":$act_lsh,
        CArg<"int32_t", "0">:$act_rsh,
        CArg<"int32_t", "0">:$act_clip_min,
        CArg<"int32_t", "0">:$act_clip_max,
        CArg<"int32_t", "0">:$act_zero_point,
        CArg<"uint32_t", "0">:$alu_disable,
        CArg<"uint32_t", "0">:$act_disable,
        CArg<"NumberFormat", "NumberFormat::I">:$alu_format,
        CArg<"NumberFormat", "NumberFormat::I">:$act_format,
        CArg<"uint32_t", "32">:$act_sum_bits
        )>,
  
      AttrBuilder<(ins
        CArg<"uint32_t", "0">:$alu_disable,
        CArg<"uint32_t", "0">:$act_disable,
        CArg<"int32_t", "0">:$pad_value
        )>
  ];

  // here we need to manually create an assembly format
  // because arrayref parameters will not be parseable
  // if we use the default assembly format

  let assemblyFormat = [{
    `{`
        `alu_op0_mode` `=` `[` $alu_op0_mode `]` `,`
        `alu_op1_mode` `=` `[` $alu_op1_mode `]` `,`
        `alu_d_unsigned` `=` $alu_d_unsigned `,`
        `alu_w_unsigned` `=` $alu_w_unsigned `,`
        `act_mode` `=` $act_mode `,`
        (`act_lsh` `=` `[` $act_lsh^ `]` `,`)?
        (`act_rsh` `=` $act_rsh^ `,`)?
        (`act_clip_min` `=` $act_clip_min^ `,`)?
        (`act_clip_max` `=` $act_clip_max^ `,`)?
        (`act_zero_point` `=` $act_zero_point^ `,`)?
        (`no_p_clear` `=` $no_p_clear^ `,`)?
        (`no_p_output` `=` $no_p_output^ `,`)?
        (`kernel_left` `=` $kernel_left^ `,`)?
        (`kernel_right` `=` $kernel_right^ `,`)?
        (`kernel_top` `=` $kernel_top^ `,`)?
        (`kernel_bottom` `=` $kernel_bottom^ `,`)?
        (`pad_left` `=` $pad_left^ `,`)?
        (`pad_right` `=` $pad_right^ `,`)?
        (`pad_top` `=` $pad_top^ `,`)?
        (`pad_bottom` `=` $pad_bottom^ `,`)?
        (`pad_value` `=` $pad_value^ `,`)?
        (`stride` `=` $stride^ `,`)?
        (`stride_offset` `=` $stride_offset^ `,`)?
        (`act_round_mode` `=` $act_round_mode^ `,`)?
        (`weight_format` `=` $weight_format^ `,`)?
        (`alu_disable` `=` $alu_disable^ `,`)?
        (`act_disable` `=` $act_disable^ `,`)?
        (`alu_format` `=` $alu_format^ `,`)?
        (`act_format` `=` $act_format^ `,`)?
        (`act_sum_bits` `=` $act_sum_bits^ `,`)?
        (`table` `=` `[` $table^`]`)?
     `}`
  }];
}


def TorqHW_RegDimAttr :
  AttrDef<TorqHW_Dialect, "RegDim", []>
{
  let mnemonic = "reg_dimension";
  let parameters = (ins
    AttrParameter<"DimType", "">:$type,
    AttrParameter<"RegDimTag", "">:$tag,
    AttrParameter<"int64_t", "">:$count,
    AttrParameter<"int64_t", "">:$stride
  );

  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins
      CArg<"DimType", "{}">:$type,
      CArg<"RegDimTag", "{}">:$tag,
      CArg<"int64_t", "1">:$count,
      CArg<"int64_t", "0">:$stride
      ), [{
      return $_get($_ctxt, type, tag, count, stride);
    }]>,
    AttrBuilder<(ins
      "RegNdlDimData":$data
      ), [{
      return $_get($_ctxt, data.type, data.tag, data.count, data.stride);
    }]>
  ];

  let assemblyFormat = [{
    $tag `(` $type `)` `[` $count `,` $stride `]`
  }];
}


def TorqHW_MemDimAttr :
  AttrDef<TorqHW_Dialect, "MemDim", []>
{
  let mnemonic = "mem_dimension";
  let parameters = (ins
    AttrParameter<"DimType", "">:$type,
    AttrParameter<"MemDimTag", "">:$tag,
    AttrParameter<"int64_t", "">:$count,
    AttrParameter<"AffineMapAttr", "">:$stride
  );

   let builders = [
    AttrBuilder<(ins
      CArg<"DimType", "{}">:$type,
      CArg<"MemDimTag", "{}">:$tag,
      CArg<"int64_t", "1">:$count,
      CArg<"int64_t", "0">:$stride
      ), [{
      return $_get($_ctxt, type, tag, count, AffineMapAttr::get(AffineMap::getConstantMap(stride, $_ctxt)));
    }]>,
    AttrBuilder<(ins
      "MemNdlDimData":$data,
      CArg<"int64_t", "0">:$numSyms
      ), [{
      return $_get($_ctxt, data.type, data.tag, data.count, data.getStrideAttr(numSyms, $_ctxt));
    }]>
  ];

  let extraClassDeclaration = [{
    std::optional<int64_t> getStrideAsI64(ArrayAttr symbolValues = {}) const;
  }];

  let assemblyFormat = [{
    $tag `(` $type `)` `[` $count `,` $stride `]`
  }];
}

def TorqHW_DmaDimAttr :
  AttrDef<TorqHW_Dialect, "DmaDim", []>
{
  let mnemonic = "dma_dimension";
  let parameters = (ins
    AttrParameter<"int64_t", "">:$count,
    AttrParameter<"int64_t", "">:$stride
  );

  let assemblyFormat = [{
    `[` $count `,` $stride `]`
  }];
}

def TorqHW_NdlTypeAttr: I32EnumAttr<"NdlType",
    "Memory based Flexible NDL Type",
    [
      I32EnumAttrCaseWithDoc<"REF",  0, "Describes the original compute task by definition; Some dimensions affect CFG_REGS; Must be written first">,
      I32EnumAttrCaseWithDoc<"DEDR", 1, "LRAM to DBUS, Flexible NDL">,
      I32EnumAttrCaseWithDoc<"DEWR", 2, "LRAM to WBUS, Flexible NDL">,
      I32EnumAttrCaseWithDoc<"DEBR", 3, "LRAM to DE.BBUS, Flexible NDL">,
      I32EnumAttrCaseWithDoc<"DEQW", 4, "DE.QBUS to LRAM, Flexible NDL">,
      I32EnumAttrCaseWithDoc<"CEDW", 5, "CE.CEDW agent;  DBUS to IRAM, Fixed-format NDL">,
      I32EnumAttrCaseWithDoc<"CEDR", 6, "CE.CEDR agent;  IRAM to ALU, Fixed-format NDL">,
      I32EnumAttrCaseWithDoc<"CEWW", 7, "CE.CEWW agent;  WBUS to WRAM, Fixed-format NDL">,
      I32EnumAttrCaseWithDoc<"CEWR", 8, "CE.CEWR agent;  WRAM to ALU, Fixed-format NDL">,
      I32EnumAttrCaseWithDoc<"CEPW", 9, "CE.CEPR agent;  PRAM to ALU, Fixed-format NDL">,
      I32EnumAttrCaseWithDoc<"CEPR", 10, "CE.CEPW agent; ALU to PRAM, Fixed-format NDL">,
      I32EnumAttrCaseWithDoc<"ACBW", 11, "ACT.ACBW agent; DE.BBUS to BREG, Fixed-format NDL">,
      I32EnumAttrCaseWithDoc<"ACBR", 12, "ACT.ACBR agent; BREG to ACT, Fixed-format NDL">,
      I32EnumAttrCaseWithDoc<"ACPW", 13, "CE.ACPW agent;  PBUS to ACT, Fixed-format NDL">,
      I32EnumAttrCaseWithDoc<"ACPR", 14, "CE.ACPR agent;  PRAM to PBUS, Fixed-format NDL">,
      I32EnumAttrCaseWithDoc<"ALDW", 15, "CE.ALDW agent; IRAM to ALU, Fixed-format NDL">
    ]> {
  let cppNamespace = "::mlir::syna::torq_hw";
}

def TorqHW_MemNdlAttr :
    AttrDef<TorqHW_Dialect, "MemNdl", []>
{
  let mnemonic = "mem_ndl";
  let parameters = (ins
    AttrParameter<"NdlType", "">:$type,
    ArrayRefParameter<"MemDimAttr", "">:$dims,
    DefaultValuedParameter<"int64_t", "0">:$index,
    DefaultValuedParameter<"int64_t", "0">:$offset,
    DefaultValuedParameter<"int64_t", "0">:$set_id,
    DefaultValuedParameter<"int64_t", "0">:$sync_mode,
    DefaultValuedParameter<"int64_t", "0">:$sync_nhd
  );

  let builders = [
    AttrBuilder<(ins "MemNdlData":$data)>,
    AttrBuilder<(ins "MemNdlData":$data, "int64_t":$numSyms)>
  ];

  let assemblyFormat = [{
    `{` $type `dims` `=` `[` $dims`]`
     (`index` `=` $index^)?
     (`offset` `=` $offset^)?
     (`set_id` `=` $set_id^)?
     (`sync_mode` `=` $sync_mode^)?
     (`sync_nhd` `=` $sync_nhd^)?
     `}`
  }];
}

def TorqHW_RegNdlAttr :
    AttrDef<TorqHW_Dialect, "RegNdl", []>
{
  let mnemonic = "reg_ndl";
  let parameters = (ins
    AttrParameter<"NdlType", "">:$type,
    OptionalArrayRefParameter<"RegDimAttr">:$dims,
    DefaultValuedParameter<"int64_t", "0">:$set_id
  );

  let assemblyFormat = [{
    `{` $type (`dims` `=` `[` $dims^ `]`)? (`set_id` `=` `[` $set_id^ `]`)?
     `}`
  }];

  let builders = [
    AttrBuilder<(ins "RegNdlData":$data)>
  ];

}

def TorqHW_MemNdlArrayAttr: ArrayOfAttr<TorqHW_Dialect, "MemNdlArray",
                                      "Array of MemNdlAttr", "MemNdlAttr", []>;

def TorqHW_RegNdlArrayAttr: ArrayOfAttr<TorqHW_Dialect, "RegNdlArray",
                                          "Array of RegNdlAttr", "RegNdlAttr", []>;


def TorqHW_DmaNdlAttr :
    AttrDef<TorqHW_Dialect, "DmaNdl", []>
{
  let mnemonic = "dma_ndl";
  let parameters = (ins
    ArrayRefParameter<"DmaDimAttr", "">:$dims
  );

  let assemblyFormat = [{
    `{` `dims` `=` `[` $dims `]` `}`
  }];
}


def TorqHW_NdlCyclesAttr :
  AttrDef<TorqHW_Dialect, "NdlCycles", []>
{
  let mnemonic = "ndl_cycles";
  let parameters = (ins
    AttrParameter<"int64_t", "">:$dedr, // LRAM to DBUS
    AttrParameter<"int64_t", "">:$dewr, // LRAM to WBUS
    AttrParameter<"int64_t", "">:$debr, // LRAM to DE.BBUS
    AttrParameter<"int64_t", "">:$deqw, // DE.QBUS to LRAM
    AttrParameter<"int64_t", "">:$cedw, // DBUS to IRAM
    AttrParameter<"int64_t", "">:$cedr, // IRAM to ALU
    AttrParameter<"int64_t", "">:$ceww, // WBUS to WRAM
    AttrParameter<"int64_t", "">:$cewr, // WRAM to ALU
    AttrParameter<"int64_t", "">:$cepr, // PRAM to ALU (omitted cepw is ALU to PRAM)
    AttrParameter<"int64_t", "">:$acbw, // DE.BBUS to BREG
    AttrParameter<"int64_t", "">:$acbr, // BREG to ACT
    AttrParameter<"int64_t", "">:$acpr  // PRAM to PBUS
  );

  let assemblyFormat = [{
    `dedr` `(` $dedr `)` `dewr` `(` $dewr `)` `debr` `(` $debr `)`
    `deqw` `(` $deqw `)` `cedw` `(` $cedw `)` `cedr` `(` $cedr `)`
    `ceww` `(` $ceww `)` `cewr` `(` $cewr `)` `cepr` `(` $cepr `)`
    `acbw` `(` $acbw `)` `acbr` `(` $acbr `)` `acpr` `(` $acpr `)`
  }];
}

#endif // TORQHW_IR_TORQHWATTRS
