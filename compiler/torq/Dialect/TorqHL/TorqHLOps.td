// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORQHL_DIALECT_IR_TORQHLOPS
#define TORQHL_DIALECT_IR_TORQHLOPS

include "torq/Dialect/TorqHL/TorqHLBase.td"
include "torq/Dialect/TorqHL/TorqHLAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/Traits.td"

include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def UpdateInPlaceTrait: NativeOpTrait<"UpdateInPlaceTrait">;

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "", "::mlir::ShapedType">;

class TorqHL_LayerOp<string mnemonic, list<Trait> traits = []>: 
        TorqHL_Op<mnemonic, !listconcat(traits, [
                              DeclareOpInterfaceMethods<TilingInterface>,
                              DeclareOpInterfaceMethods<KernelInterface, ["getKernelEncoding"]>,
                              MemoryEffectsOpInterface])> {
  dag commonArgs = (ins
    TensorOrMemref:$init
  );

  let results = (outs
    Optional<AnyRankedTensor>:$output
  );

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }

    void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
      return getLayerOpEffects(getOperation(), effects);
    }
  }];
}

def TorqHL_AddOp: TorqHL_LayerOp<"add"> {

  let summary = "Add operation";
  let description = [{
    Add two tensors
  }];

  let arguments = !con(commonArgs, (ins
    StrAttr:$name, // addOp will be reused by subOp, naming to differentiate
    I32Attr:$input_zp,
    I32Attr:$output_zp,
    I32Attr:$output_min,
    I32Attr:$output_max,
    I32Attr:$shift_factor,
    TensorOrMemref:$weights,
    TensorOrMemref:$scale_bias,
    TensorOrMemref:$input1,
    TensorOrMemref:$input2,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$segment_output,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$rhs_is_const
  ));
}

def TorqHL_MatMulOp: TorqHL_LayerOp<"matmul"> {
  let summary = "MatMul operation";
  let description = [{
    Matrix multiply two tensors
  }];
  let arguments = !con(commonArgs, (ins
    I32Attr:$output_zp,
    I32Attr:$output_min,
    I32Attr:$output_max,
    I32Attr:$shift,
    TensorOrMemref:$scale_bias,
    TensorOrMemref:$input1,
    TensorOrMemref:$input2
  ));
}

def TorqHL_MulOp: TorqHL_LayerOp<"mul"> {

  let summary = "Mul operation";
  let description = [{
    Elementwise multiply two tensors
  }];

  let arguments = !con(commonArgs, (ins
    I32Attr:$output_min,
    I32Attr:$output_max,
    TensorOrMemref:$scale_bias,
    I8Attr:$shift, // result right shift, used with i32 data type only
    TensorOrMemref:$input1,
    TensorOrMemref:$input2
  ));
}

def TorqHL_Conv2DOp: TorqHL_LayerOp<"conv2d"> {
  let summary = "2-D Convolution operation";
  let description = [{
    2-D convolution operation
  }];

  let arguments = !con(commonArgs, (ins
    I32Attr:$input_zp,
    I32Attr:$weight_zp,
    I32Attr:$output_zp,
    I32Attr:$output_min,
    I32Attr:$output_max,
    I32Attr:$shift_factor,
    I32Attr:$groups,
    DenseI64ArrayAttr:$pad,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$dilation,
    TorqHL_VectorizationModeEnum:$vectorization_mode,
    TensorOrMemref:$weights,
    TensorOrMemref:$scale_bias,
    TensorOrMemref:$input,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$nhwc_input,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$segment_output
  ));
}

def TorqHL_DepthwiseConv2DOp: TorqHL_LayerOp<"depthwise_conv2d"> {
  let summary = "Depthwise 2-D Convolution operation";
  let description = [{
    Depthwise 2-D convolution operation
  }];

  let arguments = !con(commonArgs, (ins
    I32Attr:$input_zp,
    I32Attr:$weight_zp,
    I32Attr:$output_zp,
    I32Attr:$output_min,
    I32Attr:$output_max,
    I32Attr:$shift_factor,
    I32Attr:$groups,
    DenseI64ArrayAttr:$pad,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$dilation,
    TorqHL_VectorizationModeEnum:$vectorization_mode,
    TensorOrMemref:$weights,
    TensorOrMemref:$scale_bias,
    TensorOrMemref:$input,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$nhwc_input,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$segment_output
  ));
}

def TorqHL_FullyConnectedOp: TorqHL_LayerOp<"fully_connected"> {
  let summary = "Fully connected operation";
  let description = [{
    Fully connected operation
  }];

  let arguments = !con(commonArgs, (ins
    I32Attr:$input_zp,
    I32Attr:$weight_zp,
    I32Attr:$output_zp,
    I32Attr:$output_min,
    I32Attr:$output_max,
    I32Attr:$shift_factor,
    TorqHL_VectorizationModeEnum:$vectorization_mode,
    TensorOrMemref:$weights,
    TensorOrMemref:$scale_bias,
    TensorOrMemref:$input
  ));
}

def TorqHL_AvgPool2DOp: TorqHL_LayerOp<"avg_pool2d"> {
  let summary = "Average 2-D pooling operation";
  let description = [{
    Average 2-D pooling operation
  }];

  let arguments = !con(commonArgs, (ins
    I32Attr:$input_zp,
    I32Attr:$output_zp,
    I32Attr:$output_min,
    I32Attr:$output_max,
    I32Attr:$shift_factor,
    TensorOrMemref:$weights,
    TensorOrMemref:$scale_bias,
    TensorOrMemref:$input
  ));
}

def TorqHL_ReduceMeanOp: TorqHL_LayerOp<"reduce_mean"> {
  let summary = "Average 2-D pooling operation";
  let description = [{
    Reduce mean last axis operation
  }];

  let arguments = !con(commonArgs, (ins
    I32Attr:$input_zp,
    I32Attr:$output_zp,
    I32Attr:$output_min,
    I32Attr:$output_max,
    I32Attr:$shift_factor,
    TensorOrMemref:$weights,
    TensorOrMemref:$scale_bias,
    TensorOrMemref:$input
  ));
}

def TorqHL_ReduceOp: TorqHL_LayerOp<"reduce"> {
  let summary = "Reduce operator";
  let description = [{
    Reduce operation
  }];

  let arguments = !con(commonArgs, (ins
    // we use $name to differentiate different tosa reduce ops
    StrAttr:$name,
    I32Attr:$axis,
    I32Attr:$output_zp,
    I32Attr:$output_min,
    I32Attr:$output_max,     // for output rescale
    I32Attr:$shift_factor,   // for input rescale
    TensorOrMemref:$weights, // for input rescale
    TensorOrMemref:$scale_bias,
    TensorOrMemref:$input
  ));
}

def TorqHL_ElementWiseBinaryOp: TorqHL_LayerOp<"elementwisebinary"> {
  let summary = "Elementwise Binary operator";
  let description = [{
    Elementwise Binary operation
  }];

  let arguments = !con(commonArgs, (ins
    TorqHL_ElementwiseOpEnum:$op_type,
    TensorOrMemref:$input1,
    TensorOrMemref:$input2,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_unsigned
  ));
}

def TorqHL_ElementWiseUnaryOp: TorqHL_LayerOp<"elementwiseunary"> {
  let summary = "Elementwise Unary operator";
  let description = [{
    Elementwise Unary operation
  }];

  let arguments = !con(commonArgs, (ins
    // we use $op_type to differentiate different tosa ops
    TorqHL_ElementwiseOpEnum:$op_type,
    TensorOrMemref:$input
  ));
}

def TorqHL_ElementWiseShiftOp: TorqHL_LayerOp<"elementwiseshift"> {
  let summary = "Elementwise Shift operator";
  let description = [{
    Elementwise Shift operation
  }];

  let arguments = !con(commonArgs, (ins
    // we use $op_type to differentiate different tosa ops
    TorqHL_ShiftModeEnum:$op_type,
    BoolAttr:$round,
    TensorOrMemref:$input1,
    TensorOrMemref:$input2
  ));
}

def TorqHL_ActOp: TorqHL_LayerOp<"act"> {
  let summary = "activation operator";
  let description = [{
    Activation operation
  }];

  let arguments = !con(commonArgs, (ins
    // we use $name to differentiate various activation ops
    StrAttr:$name,
    I32Attr:$input_zp,  // only for quantize NEGATE op
    I32Attr:$output_zp, // only for quantize NEGATE op
    I64Attr:$min_int,   // only for CLAMP op
    I64Attr:$max_int,   // only for CLAMP op
    F32Attr:$min_fp,    // only for CLAMP op
    F32Attr:$max_fp,    // only for CLAMP op
    TensorOrMemref:$input
  ));
}

def TorqHL_TableOp : TorqHL_LayerOp<"table"> {
  let summary = "Table lookup op";

  let description = [{
    Table lookup operation using a precomputed lookup table
  }];

  let arguments = !con(commonArgs, (ins
    TensorOrMemref:$scale_bias,
    TensorOrMemref:$input,
    DenseI32ArrayAttr:$table
  ));
}

def TorqHL_ArgMaxOp: TorqHL_LayerOp<"argmax"> {
  let summary = "ArgMax operator";
  let description = [{
    ArgMax operation on one axis
  }];

  let arguments = !con(commonArgs, (ins
    I32Attr:$axis,
    TensorOrMemref:$scale_bias,
    TensorOrMemref:$input
  ));
}

def TorqHL_GatherOp: TorqHL_LayerOp<"gather"> {
  let summary = "Gather operation";
  let description = [{
    Gather operation 
    }];

    let arguments = !con(commonArgs, (ins
      TensorOrMemref: $values,
      TensorOrMemref: $indices
      ));
}     

def TorqHl_IdentityOp: TorqHL_LayerOp<"identity"> {
  let summary = "Identity operator";
  let description = [{
    Returns a tensor with the same shape, size, type
    and content as the input.
  }];

  let arguments = !con(commonArgs, (ins
    TensorOrMemref:$input
  ));
}

def TorqHL_ScatterOp: TorqHL_LayerOp<"scatter", [UpdateInPlaceTrait]> {
  let summary = "Scatter operation";
  let description = [{
    Scatter operation 
    }];

    let arguments = !con(commonArgs, (ins
      TensorOrMemref: $indices,
      TensorOrMemref: $input,
      TensorOrMemref:$scale_bias
      ));
}

def TorqHL_TransposeOp: TorqHL_LayerOp<"transpose"> {
  let summary = "Transpose operation";
  let description = [{
    Transpose operation
  }];

  let arguments = !con(commonArgs, (ins
    DenseI64ArrayAttr:$perm,
    TensorOrMemref:$input
  ));
}

def TorqHL_FillOp: TorqHL_LayerOp<"fill"> {
  let summary = "Fill operation";
  let description = [{
    Fill operation to populate a tensor or memory reference with a specified value.
  }];

  let arguments = !con(commonArgs, (ins
    I32Attr:$value
  ));
}

def TorqHL_SegmentationOp: TorqHL_LayerOp<"segmentation"> {
  let summary = "Segmentation operation";
  let description = [{
    Segmentation operation
  }];

  let arguments = !con(commonArgs, (ins
    I32Attr:$input_zp,
    I32Attr:$output_zp,
    I32Attr:$output_min,
    I32Attr:$output_max,
    TensorOrMemref:$weights,
    TensorOrMemref:$scale_bias,
    TensorOrMemref:$input
  ));
}

def TorqHL_MaxPool2dOp: TorqHL_LayerOp<"maxpool2d"> {
  let summary = "2-D Max Pooling operation";
  let description = [{
    Performs 2-D max pooling on the input tensor, reducing spatial dimensions by applying a max filter over a defined window.
  }];

  let arguments = !con(commonArgs, (ins
    I32Attr:$input_zp,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$pad,
    DenseI64ArrayAttr:$kernel,
    TensorOrMemref:$weights,
    TensorOrMemref:$scale_bias,
    TensorOrMemref:$input,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$segment_output
  ));
}

def TorqHL_TransposeReshapeOp : TorqHL_LayerOp<"transpose_reshape"> {
  let summary = "Fused 2D transpose with reshape";
  let description = [{
    Combines reshape and transpose kernel for optimization
    Example:
      Input: [1, cols] = [op_rows, stride]
      Reshaped : [op_rows,op_cols]
      Output: [op_cols, op_rows] 
  }];

  let arguments = !con(commonArgs, (ins
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$kernel,
    DenseI64ArrayAttr:$perm,
    TensorOrMemref:$input
  ));
}


def TorqHL_Conv1DOp : TorqHL_LayerOp<"conv1d"> {
  let summary = "1-D Convolution operation";
  let description = [{
    1-D convolution operation.

    Input:  [N, C_in,1, W]
    Weights: [C_out, C_in,1, KW]
    Output: [N, C_out,1, W_out]
  }];

  let arguments = !con(commonArgs, (ins
    I32Attr:$input_zp,
    I32Attr:$weight_zp,
    I32Attr:$output_zp,
    I32Attr:$output_min,
    I32Attr:$output_max,
    I32Attr:$shift_factor,
    I32Attr:$groups,
    DenseI64ArrayAttr:$pad,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$dilation,
    TorqHL_VectorizationModeEnum:$vectorization_mode,
    TensorOrMemref:$weights,
    TensorOrMemref:$scale_bias,
    TensorOrMemref:$input,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$segment_output
  ));
}

def TorqHL_ConvertOp : TorqHL_LayerOp<"convert"> {
  let summary = "Convert the tensor to a different encoding";
  let description = [{
    Copies a tensor to a new tensor with a given encoding.
    
    The operation either converts the input tensor to the specified encoding
    if this is set or to an encoding satisfiying the requirements in the 
    requirements attribute.
  }];

  let arguments = !con(commonArgs, (ins
    TensorOrMemref:$input,
    OptionalAttr<TorqHL_TensorEncodingRequirementsAttr>:$requirements,
    OptionalAttr<TorqHL_TensorEncodingAttr>:$encoding
  ));

  let hasVerifier = 1;

  let hasCanonicalizer = true;

  let assemblyFormat = [{    
    `out` `(` $init `:` type($init) `)` `in` `(` $input `:` type($input) `)` attr-dict ( `->` type($output)^ )?
  }];
}

def TorqHL_MapBindingOp: TorqHL_Op<"map_binding"> {
  let summary = "Map binding";
  let description = [{
    Maps the given binding to a specific address in XRAM
  }];

  let arguments = (ins
    IndexAttr:$offset,
    IndexAttr:$binding_index,
    BoolAttr:$is_read_only,
    BoolAttr:$is_write_only
  );

  let results = (outs
    AnyMemRef:$output
  );

}

def TorqHL_ConstOp: TorqHL_Op<"const"> {
  let summary = "Inlined constant";
  let description = [{
    Inlined constant
  }];

  let arguments = (ins
    TypedAttrInterface:$value    
  );

  let results = (outs
    TensorOrMemref:$output
  );

}

def TorqHL_LoadOp: TorqHL_Op<"load"> {
  let summary = "Copy contents XRAM to LRAM";
  let description = [{
    Copy contents XRAM to LRAM.

    Unsafe loads may overwrite the padding bytes of the output memref.
  }];

  let arguments = (ins
    AnyMemRef:$output,
    AnyMemRef:$input,
    DenseI64ArrayAttr:$input_strides_bytes,
    DenseI64ArrayAttr:$shape,
    I64Attr:$element_size_bytes,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$unsafe
  );
  
  let hasCanonicalizer = 1;

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `:` type($input) `to` $output `:` type($output) attr-dict
  }];
}


def TorqHL_HostCopyOp: TorqHL_Op<"host_copy"> {
  let summary = "Copy between memrefs using the host";
  
  let arguments = (ins
    AnyMemRef:$output,
    AnyMemRef:$input,
    DenseI64ArrayAttr:$input_strides_bytes,
    DenseI64ArrayAttr:$output_strides_bytes,
    DenseI64ArrayAttr:$shape,
    I64Attr:$element_size_bytes
  );

  let hasVerifier = 1;
  
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    $input `:` type($input) `to` $output `:` type($output) attr-dict
  }];

}


def TorqHL_StoreOp: TorqHL_Op<"store"> {
  let summary = "Copy contents LRAM to XRAM";
  let description = [{
    Copy contents LRAM to XRAM

    Unsafe stores may write the padding bytes of the output memref.
  }];

  let arguments = (ins
    AnyMemRef:$output,
    AnyMemRef:$input,
    DenseI64ArrayAttr:$output_strides_bytes,
    DenseI64ArrayAttr:$shape,
    I64Attr:$element_size_bytes,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$unsafe
  );
  
  let hasCanonicalizer = 1;

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `:` type($input) `to` $output `:` type($output) attr-dict
  }];
}

def TorqHl_ReturnOp: TorqHL_Op<"return", [Terminator]> {
  let summary = "Terminator for a program";

  let description = [{
    Terminator of the region of a a torq_hl.program.
  }];

  let arguments = (ins
    Variadic<TensorOrMemref>:$outputs
  );

  let assemblyFormat = [{
    ($outputs^ `:` type($outputs))? attr-dict
  }];

}

def TorqHl_ProgramOp: TorqHL_Op<"program", [IsolatedFromAbove]> {
  let summary = "A sequence of operations that can be executed on an executor";

  let description = [{
    Represents a sequence of operations that can be executed on an executor.
    
    The program can have a series of arguments, that are listed as block arguments.

    The arguments are memrefs that can be written and read by the operations in 
    the program.

    The operation returns a reference to the program that can be used by
    the torq_hl.bound_program operation to associate some arguments to the program.    
  }];

  let arguments = (ins
    StrAttr:$name
  );

  let regions = (region
    AnyRegion:$body
  );

  let results = (outs
    TorqHL_Program:$program
  );

  let assemblyFormat = [{
    $name attr-dict `:` qualified(type($program)) regions
  }];

  let hasCanonicalizer = 1;

}

def TorqHl_CreateInvocationOp: TorqHL_Op<"create_invocation"> {
  let summary = "Creates an invocation of a program on a given executor";

  let description = [{
    Represents an invocation of a program on a specific executor.

    The optional address and executor attributes store the actual addresses and 
    executor id that are known only by after the torq_hl.start_program operation 
    was executed.
  }];

  let arguments = (ins
    StrAttr:$name,
    TorqHL_Program:$program,
    OptionalAttr<IndexAttr>:$executor_id,
    OptionalAttr<DenseI64ArrayAttr>:$executor_code_addresses,
    OptionalAttr<DenseI64ArrayAttr>:$xram_code_addresses,
    OptionalAttr<DenseI64ArrayAttr>:$executor_args_addresses
  );

  let results = (outs
    TorqHL_Invocation:$invocation,
    Variadic<AnyMemRef>:$code_sections
  );

  let assemblyFormat = [{
    $name `program` `(` $program `:` qualified(type($program)) `)` (`on` $executor_id^ )?  attr-dict `:` qualified(type($invocation)) (`,` type($code_sections)^)?
  }];

}

def TorqHl_CallProgramOp: TorqHL_Op<"call_program", [DestinationStyleOpInterface, AttrSizedOperandSegments]> {
  let summary = "Call program";

  let arguments = (ins
    TorqHL_Program:$program,
    Variadic<TensorOrMemref>:$inits,
    Variadic<TensorOrMemref>:$inputs
  );

  let results = (outs Variadic<AnyRankedTensor>);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getInitsMutable(); }
  }];

  let assemblyFormat = [{
    $program `inits` `(` $inits `)` `inputs` `(` $inputs `)` attr-dict `:` functional-type(operands, results)
  }];

}

def MemRefOrInvocation :
  AnyTypeOf<[AnyMemRef, TorqHL_Invocation], "">;


def TorqHl_StartProgramOp: TorqHL_Op<"start_program", [AttrSizedOperandSegments]> {
  let summary = "Starts execution of an invocation";

  let arguments = (ins
    TorqHL_Invocation:$invocation,
    Variadic<AnyMemRef>:$code_sections,
    Variadic<MemRefOrInvocation>:$args
  );

  let assemblyFormat = [{
    $invocation `:` qualified(type($invocation)) `args` `(` $args `:` type($args) `)` (`code` `(` $code_sections^ `:` type($code_sections) `)`)? attr-dict
  }];
  
  let hasCanonicalizer = 1;

}

def TorqHl_WaitProgramOp: TorqHL_Op<"wait_program"> {
  let summary = "Wait for a executor to complete a program";
  let arguments = (ins
    TorqHL_Invocation:$invocation,
    OptionalAttr<DenseI64ArrayAttr>:$result_addresses
  );

  let results = (outs
    Variadic<AnyMemRef>:$outputs
  );

  let assemblyFormat = [{
    $invocation `:` qualified(type($invocation)) attr-dict (`:` type($outputs)^)?
  }];

}

def TorqHl_ImportProgramOp: TorqHL_Op<"import_program", [ConstantLike, Pure]> {
  let summary = "Import externally defined program";

  let arguments = (ins    
    FlatSymbolRefAttr:$name
  );

  let results = (outs
    TorqHL_Program:$program
  );

  let hasFolder = 1;

}

def TorqHL_FMAOp: TorqHL_LayerOp<"fma"> {
  let summary = "Fused Multiply-Add operation";
  let description = [{
    Fused Multiply-Add operation: input * weights + scale_bias
  }];

  let arguments = !con(commonArgs, (ins
    I32Attr:$output_zp,
    I32Attr:$output_min,
    I32Attr:$output_max,
    I32Attr:$shift_factor,
    TensorOrMemref:$weights,
    TensorOrMemref:$scale_bias,
    TensorOrMemref:$input
  ));
}

def TorqHL_ResizeNearestNeighborOp: TorqHL_LayerOp<"ResizeNearestNeighbor"> {
  let summary = "Resize Nearest Neighbor Operation";
  let description =[{
    Resize Nearest Neighbor : Scale up the input by a scale factor by copying the nearest pixel by distance
  }];
  let arguments = !con(commonArgs, (ins 
    I32Attr: $scale_factor,
    TensorOrMemref: $input
  ));
}

def TorqHL_BroadcastOp: TorqHL_LayerOp<"broadcast"> {
  let summary = "broadcast operator";
  let description = [{
    Broadcasts the input tensor to a specified shape.
    The input tensor is repeated along the specified dimensions to match the output shape
  }];

  let arguments = !con(commonArgs, (ins
    DenseI64ArrayAttr:$dimensions,
    TensorOrMemref:$input
  ));
}

def TorqHL_DepthToSpaceOp: TorqHL_LayerOp<"DepthToSpace"> {
  let summary = "Depth to space operator";
  let description = [{
    Depth to space operator converts a tensor fo 4CxHxW to Cx2Hx2W 
  }];
  let arguments = !con(commonArgs, (ins 
    I32Attr:$blockSize,
    TorqHL_DepthToSpaceModeEnum:$mode_type,
    TensorOrMemref:$weights,
    TensorOrMemref:$input
  ));
}

#endif // TORQHL_DIALECT_IR_TORQHLOPS

