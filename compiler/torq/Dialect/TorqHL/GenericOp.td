// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORQHL_DIALECT_IR_GENERIC_OP
#define TORQHL_DIALECT_IR_GENERIC_OP

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
        TorqHL_Op<mnemonic, !listconcat(traits, [DeclareOpInterfaceMethods<TilingInterface>,
                              DestinationStyleOpInterface])> {
  dag commonArgs = (ins
    TensorOrMemref:$init
  );

  let results = (outs
    Optional<AnyRankedTensor>:$output
  );

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
  }];
}

// the list of interface of this op match what is defined for named structured linalg op
def TorqHl_GenericOp: TorqHL_Op<"generic", [AttrSizedOperandSegments,
                                              SingleBlockImplicitTerminator<"linalg::YieldOp">,
                                              DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                                              DestinationStyleOpInterface,
                                              LinalgStructuredInterface,
                                              ReifyRankedShapedTypeOpInterface]> {

  let summary = "Generic Torq slice operation";

  let description = [{
    This operation performs a computation from of the form:

    for d1, ..., dn:  
      p[p_map(d1, ..., dn)] = ALU(d[d_map(d1, ..., dn)], 
                                  w[w_map(d1, ..., dn)], 
                                  p[p_map(d1, ..., dn)])

      q[q_map(d1, ..., dn)] = ACT(bias[bias_map(d1, ..., dn)], scale[scale_map(d1, ..., dn)], p[p_map(d1, ..., dn)])

    the operation returns the final value of p and q.

    The functions ALU and ACT depend on the parameters alu_config and act_config.  

    All inputs except p are optional, in that case the ACT and ALU operations have to
    be configured accordingly so that they don't use the corresponding value.

    This operator is a named linalg operator and can be converted to linalg.genericOp
    using the LinalgStructuredInterface.

    The first return value is always P, the second optional return value is Q.

  }];

  let arguments = (ins
    Optional<TensorOrMemref>:$d,
    Optional<TensorOrMemref>:$w,
    Optional<TensorOrMemref>:$bias,
    Optional<TensorOrMemref>:$scale,
    TensorOrMemref:$p,
    Optional<TensorOrMemref>:$q,
    OptionalAttr<AffineMapAttr>:$d_map,
    OptionalAttr<AffineMapAttr>:$w_map,
    OptionalAttr<AffineMapAttr>:$bias_map,
    OptionalAttr<AffineMapAttr>:$scale_map,
    AffineMapAttr:$p_map,
    OptionalAttr<AffineMapAttr>:$q_map,    
    OptionalAttr<TorqHL_ActConfigAttr>:$act_config,
    OptionalAttr<TorqHL_AluConfigAttr>:$alu_config
  );
  
  let results = (outs
    Variadic<AnyRankedTensor>:$out
  );

  let regions = (region AnyRegion:$region);

  code extraClassDeclaration = [{
    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      return !this->getBody()->getOps<linalg::IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return llvm::cast<linalg::LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }

    SmallVector<utils::IteratorType> getIteratorTypesArray();

    ArrayAttr getIndexingMaps();

    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);

    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    ::mlir::MutableOperandRange getDpsInitsMutable();

    std::string getLibraryCallName();

  }];

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<
    (ins "GenericOpConfig":$config, CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
    [{
      buildTorqGenericOp($_builder, $_state, config, attributes, GenericOp::getRegionBuilder());
    }]>
  ];

  let hasCustomAssemblyFormat = 1;

  let assemblyFormat = [{ `\n`
    (`D` $d^ `:` type($d) `map` `=` $d_map `\n` )?
    (`W` $w^ `:` type($w) `map` `=` $w_map `\n` )?
    (`bias` $bias^ `:` type($bias) `map` `=` $bias_map `\n` )?
    (`scale` $scale^ `:` type($scale) `map` `=` $scale_map `\n` )?
    `P` $p `:` type($p) `map` `=` $p_map `\n` 
    (`Q` $q^  `:` type($q) `map` `=` $q_map `\n` )?
    (`alu_config` `=` $alu_config^ `\n` )?
    (`act_config` `=` $act_config^ `\n` )?
    attr-dict 
    `:`  type(results)
    regions
  }];

}

#endif // TORQHL_DIALECT_IR_GENERIC_OP
