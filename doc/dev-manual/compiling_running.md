
### Compile and run a model

Example MLIR models are provided in the `tests/` directory.

- Compile a mlir for the Torq target:

    ```{code} shell
    $ ../iree-build/third_party/iree/tools/torq-compile \
        tests/testdata/tosa_ops/add-single-rescale.mlir \
        -o add-single-rescale.vmfb
    ```

- Run the generated model with the Torq simulator:

    ```{code} shell
    $ ../iree-build/runtime/tools/torq-run-module --module=add-single-rescale.vmfb \
        --function=main --input="1x56x56x24xi8=1"
    ```

    :::{tip}
    ``add-single-rescale.mlir`` is an ASCII text file containing the representation of the
    model using the TOSA dialect. In some cases, such files can be quite large since they include all the model weights, and some text editors may struggle to open them. Visual Studio Code works well with this kind of files.
    :::