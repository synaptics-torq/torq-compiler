## Adding a new compute optimization

This section explains how to add a new optimization that rewrites a given linalg-on-tensors pattern into a more efficient version
of the pattern. The rewrite transforms a set of linalg operations to a combination of torq_hl kernels and/or linalg operations that
can be efficiently executed on the hardware.

To implement such an optimization it is recommended to follow two steps:

1. Add a test case for the pattern in ``test/testdata/linalg_ops``. To create the test case it is possible to use as reference an
   existing mlir files in the same directory, look for mlir files in the ``third_party/iree`` directory, or run the compiler on
   some input model as follows:

   ```
   $ {BUILD_DIR}/third_party/iree/tools/torq-compile --compile-to=input mymodel.mlir -o mymodel_linalg.mlir
   ```

   Make sure to edit the file so that ``func.func`` and ``func.return`` operations are used instead of the ``util`` dialect variants.

2. Implement and register a new rewrite pattern in the ``LinalgToTorqHLConversionPass``.

   1. Implement a new class in ``compiler/torq/Conversions/LinalgToTorqHL/Patterns.cpp``

      ```{c++}
      // this pattern triggers when a linalg::SomeOp is found in the IR
      struct MyOptimizationPattern : public OpRewritePattern<linalg::SomeOp> {

         MyOptimizationPattern(MLIRContext *context)
            : OpRewritePattern<linalg::SomeOp>(context, /*benefit=*/0) {
            setDebugName("MyOptimizationPattern");
         }

         LogicalResult matchAndRewrite(linalg::SomeOp op, PatternRewriter &rewriter) const override {

            if (op.getSomeProperty() != SOME_VALUE) {
               return rewriter.notifyMatchFailure(op, "Some property must be SOME_VALUE");
            }

            rewriter.replaceOpWithNewOp<torq_hl::SomeOp>(op, ...);

            return success();
         }
      };
      ```

   2. Register the class in the ``populateLinalgToTorqHLPrePatterns`` function in the same file:

      ```{c++}
      patterns.insert<MyOptimizationPattern>(context);
      ```

More information on how pattern rewriting works can be found in the [MLIR documentation](https://mlir.llvm.org/docs/PatternRewriter/).

To test the pattern in isolation use the following command:

```
$ {BUILD_DIR}/third_party/iree/tools/iree-opt \
   --pass-pipeline='builtin.module(func.func(torq-linalg-to-torqhl-pre-conversion{enable-patterns=MyOptimizationPattern}))' \
   --debug-only=dialect-conversion \
   mytest.mlir
```

The command executes the conversion on the input test mlir file and prints on stdout the logs
of the pattern application process and the resulting mlir after application.

Notice that in the same optimization pass other patterns may apply before and after the new pattern. To debug
the overall pattern application process that happens concurrently use the following command line:

```
$ {BUILD_DIR}/third_party/iree/tools/iree-opt \
   --pass-pipeline='builtin.module(func.func(torq-linalg-to-torqhl-pre-conversion))' \
   --debug-only=dialect-conversion \
   mytest.mlir
```

To dump the state of the IR at the moment the pattern is applied use the following code at the beginning
of the ``matchAndRewrite`` function:

```{c++}
op->getParentOp()->dump();
```

The debug log will show the order in which patterns are applied. To change the order in which patterns
that match the same operation should be applied you can change the ``benefit`` argument of the constructor.

The compiler will further process the output of this pass with other passes. To test the pattern within 
the whole compiler pipeline use the following command:

```
$ rm -rf tmp/ir && mkdir -p tmp/ir && pytest tests/test_linalg.py -k "test_with_torq[mytest.mlir]" -s \
   --extra-torq-compiler-options="--mlir-print-ir-before=torq-linalg-to-torqhl-pre-conversion --mlir-print-ir-after=torq-linalg-to-torqhl-pre-conversion --mlir-print-ir-tree-dir=$(pwd)/tmp/ir"
```

The compiler will create in ``tmp/ir`` two dump files, one before and one after applying the conversion.