// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORQHW_IR_TORQHWOPS
#define TORQHW_IR_TORQHWOPS

include "torq/Dialect/TorqHW/TorqHWBase.td"
include "torq/Dialect/TorqHW/TorqHWAttrs.td"
include "torq/Dialect/TorqHL/TorqHLAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"


class TorqHW_DmaCfgOp<string mnemonic> : TorqHW_Op<mnemonic> {
  let arguments = (ins
    AnyMemRef:$read,
    AnyMemRef:$write,
    TorqHW_DmaNdlAttr:$read_ndl,
    TorqHW_DmaNdlAttr:$write_ndl,
    OptionalAttr<I32Attr>:$read_address,
    OptionalAttr<I32Attr>:$write_address);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `from` $read `:` type($read) `to` $write `:` type($write) attr-dict  
  }];
}

def TorqHW_DmaInCfgOp: TorqHW_DmaCfgOp<"dma_in_cfg">;
def TorqHW_DmaOutCfgOp: TorqHW_DmaCfgOp<"dma_out_cfg">;

class TorqHW_DmaInOutOp<string mnemonic> : TorqHW_Op<mnemonic> {
  let assemblyFormat = "attr-dict";
}
def TorqHW_DmaInStartOp: TorqHW_DmaInOutOp<"dma_in_start">;
def TorqHW_DmaInWaitOp: TorqHW_DmaInOutOp<"dma_in_wait">;

def TorqHW_DmaOutStartOp: TorqHW_DmaInOutOp<"dma_out_start">;
def TorqHW_DmaOutWaitOp: TorqHW_DmaInOutOp<"dma_out_wait">;

def TorqHW_SliceStartOp : TorqHW_Op<"slice_start"> {
  let summary = "Slice Start Operation";
  let arguments = (ins
    TorqHL_Invocation:$invocation,
    AnyMemRef:$program,
    IndexAttr:$id,
    Variadic<AnyMemRef>: $args);
}

def TorqHW_SliceWaitOp : TorqHW_Op<"slice_wait"> {
  let summary = "Slice Wait Operation";
  let arguments = (ins IndexAttr:$id);
}

def TorqHW_CSSWaitOp : TorqHW_Op<"css_wait"> {
  let summary = "Wait for CSS to signal an interrupt";  
  let assemblyFormat = "attr-dict";
}

def TorqHW_CSSStartOp : TorqHW_Op<"css_start"> {
  let summary = "Start program on CSS";

  let arguments = (ins
    TorqHL_Invocation:$invocation,
    AnyMemRef:$program,
    AnyMemRef:$args_addresses,
    Variadic<AnyMemRef>: $args,
    OptionalAttr<I32Attr>:$program_address,
    OptionalAttr<I32Attr>:$arg_addresses_address);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $invocation `:` qualified(type($invocation)) `program` $program `(` $args_addresses `=` $args `)` attr-dict `:` type($program) `,` type($args_addresses) `,` type($args)
  }];

}

def TorqHW_CDMAStartOp : TorqHW_Op<"cdma_start"> {
  let summary = "Start cDMA transfer";
  let arguments = (ins
    AnyMemRef:$dest,
    AnyMemRef:$src,
    OptionalAttr<I32Attr>:$dest_address,
    OptionalAttr<I32Attr>:$src_address);
  
  let assemblyFormat = [{
    `from` $src `:` type($src) `to` $dest `:` type($dest) attr-dict      
  }];

}

def TorqHW_CDMAWaitOp : TorqHW_Op<"cdma_wait"> {
  let summary = "Wait for cDMA to finish transfer";  
  let assemblyFormat = "attr-dict";
}

def TorqHW_NextOp : TorqHW_Op<"next", [Terminator]> {
  let summary = "Jump to code block in dest, this is a special type of nss_task that is also a terminator.";
  let arguments = (ins AnyMemRef:$dest_address);
  let successors = (successor AnySuccessor:$dest_block);
}

def TorqHW_GetFunctionBlockOp : TorqHW_Op<"get_function_block"> {
  let summary = "Get a XRAM memref containing the bitstream of the given code block of the current function";
  let arguments = (ins IndexAttr:$id);
  let results = (outs AnyMemRef:$block);
}

def TorqHW_GetCSSTaskBlockOp : TorqHW_Op<"get_css_task_block"> {
  let summary = "Get a XRAM memref containing the compiled code for a given CSS task";
  let arguments = (ins FlatSymbolRefAttr:$task_name);
  let results = (outs AnyMemRef:$block);
}

def TorqHW_GetAddressOp : TorqHW_Op<"get_address"> {
  let summary = "Gets the address of a given memref (either XRAM or LRAM depending on the memory space)";
  let arguments = (ins AnyMemRef:$memref);
  let results = (outs Index:$address);

  let builders = [
    OpBuilder<(ins "Value":$memref),
    [{
      build($_builder, $_state, $_builder.getIndexType(), memref);
    }]>
  ];
}

def TorqHW_NssTaskOp : TorqHW_Op<"nss_task", [NoTerminator]> {
  let summary = "nss task";
  let description = [{
    nss task
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    attr-dict `(` `)` regions
  }];
}

def TorqHW_SliceProgramOp : TorqHW_Op<"slice_program"> {
  let summary = "slice program";
  let description = [{
    slice
  }];

  let arguments = (ins 
    I32Attr:$slice_id,
    I32Attr:$cfg_desc_laddr,
    I32Attr:$cfg_desc_xaddr
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    attr-dict `(` `)` regions
  }];
}

def MemNdlArg : CArg<"ArrayRef<MemNdlDimData>", "{}">;
def RegNdlArg : CArg<"ArrayRef<RegNdlDimData>", "{}">;

def TorqHW_SliceTaskOp : TorqHW_Op<"slice_task", [AttrSizedOperandSegments]> {
  let summary = "slice task";
  let description = [{
    slice config
  }];

  let arguments = (ins
    StrAttr:$op_name,
    Variadic<AnyMemRef>:$d,
    Variadic<AnyMemRef>:$w,
    Variadic<AnyMemRef>:$b,
    Variadic<AnyMemRef>:$q,
    Variadic<AnyMemRef>:$dx,
    Variadic<AnyMemRef>:$bx,
    Variadic<Index>:$symbols,
    TorqHW_SliceCFGAttr:$slice_cfg_attr,
    TorqHW_MemNdlArrayAttr:$mem_ndls,
    TorqHW_RegNdlArrayAttr:$reg_ndls
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `{`
      `\n` `op` `:` $op_name
      (`\n` `D` $d^ `:` type($d))?
      (`\n` `W` $w^ `:` type($w))?
      (`\n` `B` $b^ `:` type($b))?
      (`\n` `Q` $q^ `:` type($q))?
      (`\n` `DX` $dx^ `:` type($dx))?
      (`\n` `BX` $bx^ `:` type($bx))?
      (`\n` `SYMBOLS` $symbols^)?
      attr-dict
      `\n` `CFG` $slice_cfg_attr
      `\n` $mem_ndls
      `\n` $reg_ndls
    `\n` `}`
  }];

  let builders = [

    OpBuilder<(ins "StringRef":$op_name,
          "Value":$d, "Value":$w, "Value":$b, "Value":$q,
          "SliceCFGAttr":$slice_cfg_attr, "Ndls":$ndls)>,

    OpBuilder<(ins "StringRef":$op_name,
          "ValueRange":$d, "ValueRange":$w, "ValueRange":$b, "ValueRange":$q,
          "ValueRange":$symbols,
          "SliceCFGAttr":$slice_cfg_attr, "Ndls":$ndls)>,

    OpBuilder<(ins "StringRef":$op_name,
          "ValueRange":$d, "ValueRange":$w, "ValueRange":$b, "ValueRange":$q,
          "ValueRange":$dx, "ValueRange":$bx,
          "ValueRange":$symbols,
          "SliceCFGAttr":$slice_cfg_attr, "Ndls":$ndls)>
  ];

}

def TorqHW_DispatchProfilingOp : TorqHW_Op<"dispatch_profiling", []> {
  let summary = "Profiling of dispatch";
  let description = [{
    total memory and cycle profiling inside one dispatch
  }];

  let arguments = (ins
    I32Attr:$total_mem,
    I32Attr:$total_cycle
  );

  let assemblyFormat = [{
    attr-dict
    `\n` `memory` `:` $total_mem `bytes`
    `\n` `total_cycle` `:` $total_cycle
  }];
}

def TorqHW_SliceProfilingOp : TorqHW_Op<"slice_profiling", []> {
  let summary = "Profiling of slice";
  let description = [{
    slice memory and cycle profiling
  }];

  let arguments = (ins
    I32Attr:$slice_mem,
    StrAttr:$slice_meminfo,
    I32Attr:$max_cycle,
    TorqHW_NdlCyclesAttr:$slice_ndl_cycles
  );

  let assemblyFormat = [{
    attr-dict
    `\n` `memory` `:` $slice_mem `bytes`
    `\n` `memory_info` `:` $slice_meminfo
    `\n` `max_cycle` `:` $max_cycle
    `\n` `ndl_cycles` `:` $slice_ndl_cycles
  }];
}

#endif // TORQHW_IR_TORQHWOPS
