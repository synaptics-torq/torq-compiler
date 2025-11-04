## Debug tips and tricks

- To see the logs of a given module append ``-debug-only=${MODULE_NAME}`` to the command line.
    A _module_ normally corresponds to one pass or to a subset of one pass.
    The available modules can be listed by searching for ``DEBUG_TYPE`` in all ``.h`` and ``.cpp``
    files in *torq-compiler*.

- To debug a torqhl conversion pass use the following parameters:

  ```{code} shell
  $ iree-opt input.mlir -o output.mlir --torq-tosa-transformation-pipeline \
      -debug-only=dialect-conversion
  ```

- To debug a torqhw conversion pass use the following parameters:

  ```{code} shell
  $ iree-opt input.mlir -o output.mlir --torq-torqhw-transformation-pipeline \
      -debug-only=dialect-conversion
  ```

  :::{note}
  Additional tracing and debugging options in the [MLIR doc](https://mlir.llvm.org/docs/PassManagement/#ir-printing).
  :::

- To debug using gdb the compiler plugin you can use the following command line:

  ```{code} shell
  $ gdb --args /bin/bash scripts/torq-compile ../iree-build/third_party/iree/tools/iree-compile ${OTHER_TORQ_COMPILE_PARAMS}
  ```

- To debug using gdb the hal runtime plugin you can use the following command line:

  ```{code} shell
  $ gdb --args ../iree-build/third_party/iree/tools/iree-run-module ${OTHER_RUN_MODULE_PARAMS}
  ```

- To get a full dump of each pass IR with a test run::

  ```{code} shell
  $ pytest -k ${TEST_PATTERN} --debug-ir -s  
  ```

  Where `${TEST_PATTERN}` is a pattern to match the test that need to be run. The directory containing the logs is listed in the output.

- To run the test cases with custom compilation or runtime options use::

  ```{code} shell
  $ pytest -k ${TEST_PATTERN} --extra-torq-compiler-options="--option1 --option2" --extra-torq-runtime-options="--option3 --option4"
  ```

  Where `${TEST_PATTERN}` is a pattern to match the test that need to be run.

- To trace the contents of all buffers during model execution use the following command line:

  ```{code} shell
  $ pytest -k ${TEST_PATTERN} --trace-buffers -s
  ```
 
  Where `${TEST_PATTERN}` is a pattern to match the test that need to be run. The directory containing the buffer
  dump is listed in the output. The buffer logs can be inspected with:

  ```{code} shell
  $ streamlit run apps/buffer_viewer/buffer_viewer.py PATH/TO/DUMP_DIR
  ```

- To run lit style tests use the following commands:

  ```{code} shell

  $ export PATH=${PATH}:${PWD}/../iree-build/llvm-project/bin:${PWD}/../iree-build/tools
      
  $ llvm-lit path_to_test_case.mlir

  ```

- To disassemble the RISC-V binary images used on the CSS use the command:

    ```{code} shell
    $ riscv64-unknown-elf-objdump -D -b binary -m riscv:rv32 /tmp/css-XXXX.bin
    ```

- To run the compiler in single threaded mode we add the option ``--mlir-disable-threading``

- To debug LRAM and XRAM contents during execution you can create models with debug information with the option ``--torq-enable-buffer-debug-info``
  then to dump the contents of the buffers at each interrupt add the option ``--torq_dump_buffers_dir=PATH/TO/DUMP_DIR`` to the runtime. The dumps are
  captured in NumPy format after each block of the NSS program is executed. The dump waits for any in-flight slice and dma operation to finish before
  executing the dump. The buffer number corresponds to the index of the memref.alloc operation in the MLIR that is serialized. To view these dumps
  you can use the a web based tool:

  ```{code} shell
  $ streamlit run apps/buffer_viewer/buffer_viewer.py PATH/TO/DUMP_DIR
  ```

- To compare two tensors you can use the following tool:

  ```{code} shell
  $ streamlit run apps/buffer_diff/buffer_diff.py PATH/TO/TENSOR1 PATH/TO/TENSOR2
  ```

- To use a debugger to debug CSS task running on qemu add the option ``--torq_qemu_debug=PORT_NUMBER`` to the runtime then use ``gdb-multiarch`` from
  your distribution to connect to it:

  ```{code}
  $ gdb-multiarch

  $ (gdb) target remote :1090
  ```

- After killing a model emulation the terminal is corrupted (characters are not echo-ed back). This is due to qemu disabling echo and being aborted. To restore the pseudoterminal use the command ``stty echo``.

- To dump ELF files with symbols for all the CSS programs use the parameter  ``--torq-create-css-symbols=PATH`` of the compiler, these ELF files can be
  used to debug on qemu

- To analyze the code size of CSS tasks dump the elf files as specified above and then use the following command:

  ```{code} shell
  $ ./scripts/create_css_task_call_graph.py path/to/elf report.png
  ```

  The generated PNG shows the call dependencies detected in the code of each function and the cumulative size of the function and all the functions it calls

- To force NSS program blocks to contain only one slice task set the compiler option ``--torq-nss-task-size=1``, this ensures each slice start happens in a
  different NSS task and therefore the input/outputs can be visible in the buffer trace.

- To compare the output and the IR of two branches of the code base you can use the following script:

  ```{code} shell
  $ ./scripts/compare_builds.sh wip/goodbranch wip/badbranch pytest tests/test_css_qemu.py -k matmul
  ```

  The script automatically checks out the branches, compiles them and runs the provided pytest command with the `--debug-ir` option pointing to `tmp/comparison`.

- Torq Compiler Flags and Options
  - To disable the segmentation operation fusion use the option ``--torq-disable-segmentation-fusion``

  - To disable the fusion of all operators into one single dispatch use the option ``--torq-disable-dispatch-fusion``

  - To see all supported Torq flags, run:

    ```{code} shell
    $ ../iree-build/third_party/iree/tools/iree-compile --help | grep "torq"
    ```
