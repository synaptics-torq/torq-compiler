# TOSA to Torq Conversion

The conversion of a TOSA operator to a Torq kernel is one of the
main activities in the development of the Torq compiler.

This activity can be decomposed in the following subtasks:

1. Decide which TOSA operator, or _sequence_ of operations, to lower to a TorqHL kernel.
    It's important not to consider each TOSA operator in isolation, because there are cases
    when operators occour in predefined patterns (eg. Rescale - Convolution - Rescale) where each
    operation in the sequence is very inefficient or impossible to implement in the Torq (e.g.
    the TOSA _Rescale_ operator is 32 bits) while the combination of them can be computed in a 
    single pass by the Torq ALU and Activation units.

2. Decide which are the inputs and outputs of the TorqHL kernel and which are the required
    parameters. 

3. Define a corresponding TorqHL operation in
    [TorqHLOps.td](https://github.com/synaptics-torq/torq-compiler/blob/main/compiler/torq/Dialect/TorqHL/TorqHLOps.td) .
    Inputs and outputs are represented with data tensors, while parameters are represented with
    [attributes](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/#attributes).

4. In the [TOSAToTorqHL](https://github.com/synaptics-torq/torq-compiler/blob/main/compiler/torq/Conversions/TosaToTorqHL/Patterns.cpp)
    pass add a pattern that matches the TOSA operator(s) and lowers it
    to the corresponding TorqHL operator.

5. Design the actual implementation of the operation on the Torq, this means selecting the
    right configuration for the ALU, Activation and DMA units, and the right set of NDLs to
    access the required data, weights and biases. Weights and biases can come from the corresponding
    TOSA operator (as in the case of Convolution) or can be hardcoded in the kernel implementation
    (as in the case of the Add operator).

6. In the [LowerTorqHLPass](https://github.com/synaptics-torq/torq-compiler/blob/main/compiler/torq/Conversions/TorqHLToTorqHW/Passes.cpp)
    pass add a pattern that matches the TorqHL operator and lowers it
    to the corresponding TorqHW operators.

:::{note}
The TorqHL operator defined in TorqHLOps.td is sometimes very similar to the corresponding
TOSA operator. Why not keeping the TOSA operator then instead of defining our own?
The first reason is that even if very similar the two definitions are never exactly the same.
The second reason is to decouple our code from changes in the TOSA dialect:
only the conversion pass to TorqHL would have to be modified in this case.
:::


## Example


In this example we see how to support a REDUCE_MAX operator.

1. Create a sample tosa MLIR file containing this operator.
The best way is to start from an actual model containing corresponding TFLite operator.
In this way we can see how the operator in the model is converted to the TOSA dialect and if this
operator is converted in isolation or as a group of related operators.
If the desired operator is not available in TFlite, it's possible to create a model with a similar
operator and change the actual type later in MLIR. For example:

```{code} shell
$ ../torq-compiler/scripts/gen-model.py -m reduce_max -o model.tflite

Quantized TFLite REDUCE_MAX model created successfully: model.tflite
```

:::{tip}
The ``gen-model.py`` script can be easily customized to create models with the desired layer(s)
:::

Now that we have a model we can convert it to TOSA as explained in @tflite-to-tosa:

```{code} shell
$ iree-import-tflite model.tflite -o model.tosa
../iree-build/third_party/iree/tools/iree-opt model.tosa -o model.mlir
```

The generated MLIR file is very simple:

```{code} mlir
module {
  func.func @main(%arg0: tensor<3x4xi8> {ml_program.identifier = "input"}) -> (tensor<4xi8> {ml_program.identifier = "Identity"}) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<3x4xi8>) -> tensor<1x4xi8>
    %1 = tosa.reshape %0 {new_shape = array<i64: 4>} : (tensor<1x4xi8>) -> tensor<4xi8>
    return %1 : tensor<4xi8>
  }
}
```

Let's copy it to ``torq-compiler/tests/testdata/tosa_ops/reduce_max.mlir`` for future reference.

2. From the [TOSA ReduceMaxOP](https://mlir.llvm.org/docs/Dialects/TOSA/#tosareduce_max-mlirtosareducemaxop)
documentation we see that this operation is quite simple, it has one input tensor, one output and 
one attribute, the axis on which to perform reduction.

3. Add a ``ReduceMax`` TorqHL operator to
    [TorqHLOps.td](https://github.com/synaptics-torq/torq-compiler/blob/main/compiler/torq/Dialect/TorqHL/TorqHLOps.td) .

Good starting points for the definition of this operator are similar operators in the TorqHL dialect
in the same ``.td`` file, or the definition of the corresponding TOSA operator in 
``iree/third_party/llvm-project/mlir/include/mlir/Dialect/Tosa/IR/TosaOps.td``

```{code} mlir
def TorqHL_ReduceMaxOp: TorqHL_LayerOp<"reduce_max"> {
    let summary = "Reduce Max operator";
    let description = [{
    Reduce-Max operation on one axis
    }];

    let arguments = !con(commonArgs, (ins
    I32Attr:$axis,
    TensorOrMemref:$input
    ));
}
```

:::{note}
The output tensor is not explicitly defined in the definition above because it is already specified
in the ``TorqHL_LayerOp`` structure from which it derives.
:::

4. Let's add our lowering to the [Patterns.cpp](https://github.com/synaptics-torq/torq-compiler/blob/main/compiler/torq/Conversions/TosaToTorqHL/Patterns.cpp)
file in TOSAToTorqHL conversion.
This means creating a new conversion class based on ``OpConversionPattern`` and inserting it to
``populateTOSAToTorqHLPatterns()``.

5. Add bufferization support by adding torq_hl::ReduceMaxOp to registerTorqHLBufferizationInterfaces()
   *Bufferization* is the process of converting operands from {term}`SSA` representation to concrete buffers
   that can be read and written.

6. If we try to compile a model containing a reduce-max layer at this point we will get an error
   from *MLIR* like the following:
   :::{code} shell
   $ error: failed to legalize operation 'torq_hl.reduce_max' that was explicitly
   marked illegal
   :::
   This means that we are not allowed to use these high level operations directly to generate the 
   instructions for the Torq, they must be converted to lower-level operations expressed in the
   `TorqHW` dialect. This can be done by taking a sample kernel written in `C` that performs the
   same operation and converting it in a sequence of `TorqHW` operations.
   See for example the existing conversions in
   [OpPatterns](https://github.com/synaptics-torq/torq-compiler/blob/main/compiler/torq/Conversions/TorqHLToTorqHW/)