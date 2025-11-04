
### Compile and run a model

Example MLIR models are provided in the `tests/` directory.

- Compile a mlir for the Torq target:

    ```{code} shell
    $ scripts/torq-compile ../iree-build/third_party/iree/tools/iree-compile \
        tests/testdata/tosa_ops/conv2d-stride4.mlir \
        -o conv2d-stride4.vmfb
    ```

- Run the generated model with the Torq simulator:

    ```{code} shell
    $ ../iree-build/third_party/iree/tools/iree-run-module --device=torq --module=conv2d-stride4.vmfb \
        --function=main --input="1x256x256x1xi8=0"
    ```

    :::{tip}
    ``conv2d-stride4.mlir`` is an ASCII text file containing the representation of the
    model using the TOSA dialect. In some cases, such files can be quite large since they include all the model weights, and some text editors may struggle to open them. Visual Studio Code works well with this kind of files.
    :::