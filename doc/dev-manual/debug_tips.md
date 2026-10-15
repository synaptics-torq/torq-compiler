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
  $ iree-opt input.mlir -o output.mlir \
      --pass-pipeline='builtin.module(func.func(torq_hl.program(torq-convert-nss-program-to-torqhw)))' \
      -debug-only=dialect-conversion
  ```

  :::{note}
  Additional tracing and debugging options in the [MLIR doc](https://mlir.llvm.org/docs/PassManagement/#ir-printing).
  :::

- To debug using gdb the compiler plugin you can use the following command line:

  ```{code} shell
  $ gdb --args ../iree-build/third_party/iree/tools/torq-compile ${OTHER_TORQ_COMPILE_PARAMS}
  ```

- To debug using gdb the hal runtime plugin you can use the following command line:

  ```{code} shell
  $ gdb --args ../iree-build/runtime/tools/torq-run-module ${OTHER_RUN_MODULE_PARAMS}
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
  $ streamlit run webapps/buffer_viewer/buffer_viewer.py PATH/TO/DUMP_DIR
  ```

- To run lit style tests use the following commands:

  ```{code} shell

  $ export PATH=${PATH}:${PWD}/../iree-build/llvm-project/bin:${PWD}/../iree-build/third_party/iree/tools
      
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
  $ streamlit run webapps/buffer_viewer/buffer_viewer.py PATH/TO/DUMP_DIR
  ```

- To compare two tensors you can use the following tool:

  ```{code} shell
  $ streamlit run webapps/buffer_diff/buffer_diff.py PATH/TO/TENSOR1 PATH/TO/TENSOR2
  ```

- To analyze the code size of CSS tasks dump the elf files as specified above and then use the following command:

  ```{code} shell
  $ ./scripts/create_css_task_call_graph.py path/to/elf report.png
  ```

  The generated PNG shows the call dependencies detected in the code of each function and the cumulative size of the function and all the functions it calls

- To force NSS program blocks to contain only one slice task set the compiler option ``--torq-nss-task-size=2``, this ensures each slice start happens in a
  different NSS task and therefore the input/outputs can be visible in the buffer trace.

- To compare the output and the IR of two branches of the code base you can use the following script:

  ```{code} shell
  $ ./scripts/compare_builds.sh wip/goodbranch wip/badbranch pytest tests/test_css_mpact.py -k matmul
  ```

  The script automatically checks out the branches, compiles them and runs the provided pytest command with the `--debug-ir` option pointing to `tmp/comparison`.

- Torq Compiler Flags and Options
  - To disable the segmentation operation fusion use the option ``--torq-disable-segmentation-fusion``

  - To disable the fusion of all operators into one single dispatch use the option ``--torq-disable-dispatch-fusion``

  - ``--torq-create-css-symbols=<DIR>``: dump each CSS program as a standalone ``.elf`` under
    ``DIR/<dispatch>/<program>.elf`` so a linked kernel can be inspected or run in isolation.
    Debug aid; no effect on the compiled model unless the directory is set.

  - ``--torq-vm-print-statistics``: print the LRAM/XRAM swap-in/swap-out statistics collected
    while materializing buffers (swap-outs, stores/loads skipped, bytes dropped/stored/loaded).
    Useful to see how much spill traffic a model generates. Debug aid; does not change codegen.

  - ``--torq-dump-compute-constants-ir=<DIR>``: dump the IR of every pass run to JIT-compute
    compile-time constants into a file tree under ``DIR``. Debug aid.

  - ``--torq-insert-debug-trap-in-compute-constants``: insert a ``llvm.debugtrap`` (plus debug
    info) at the entry of the JIT'd constants-computation code so a breakpoint can be set on it.
    Debug aid; does not change the computed values.

  - ``--torq-disable-async-slice-wait``: disable the asynchronous slice wait, i.e. emit an
    explicit wait after each slice start. Used to generate deterministic HW test vectors for the
    HW team. Off by default (async wait).

  - ``--torq-nss-block-size=<N>`` (default 4): max number of operations in an NSS program block.
    Changing it re-partitions work across slices; useful for debugging codegen structure.

  - ``--torq-fallback-f32-to-host`` (default on): route every f32 operation to the host executor.
    Keep on for correctness; turn off to measure how much f32 work an NPU lowering would take
    (performance comparison).

  - ``--torq-from-prebufferized-ir``: compile already-bufferized (memref) inputs, skipping the
    bufferization, allocation and map-bindings passes. Debug/test aid; off by default.

  - ``--torq-hl-tile-input-size-threshold=<N>`` and ``--torq-hl-tile-dim-size=<S>``: together
    enable ``TorqHLTileLargeInputs``, which tiles TorqHL ops (conv/matmul/pool) whose single
    input exceeds ``N`` elements using a fixed tile of ``S`` per parallel domain. Both default to
    0 (disabled). The pass is kept because it exercises the TorqHL TilingInterface we plan to
    enable for general large-input tiling.

  - ``--torq-swap-extract-convert-max-waste-percent=<N>`` (default 12): the most extra LRAM, in
    percent of LRAM, that SwapExtractAndConvert may keep live. 12 is safe on every model we
    measured but not the best for all of them (yolo_pose was best at 24). Use it to tune a model
    or to compare performance.

  - ``--torq-eliminate-transpose-mode=enable|disable`` (default ``enable``): replace a
    ``linalg.transpose`` that keeps the flat (row-major) element order, i.e. one that only moves
    size-1 dims, with a free ``tensor.collapse_shape``/``expand_shape``. The transpose kernel
    disappears. Value-preserving; use ``disable`` to compare with the transpose kernel or to rule
    the rewrite out when debugging.

  - ``--torq-collapse-unrolled-concat-mode=enable|disable`` (default ``enable``): before dispatch
    formation, rewrite a ``tensor.concat`` of 32 or more operands whose operands include runs of
    consecutive unit ``tensor.extract_slice`` of one source (an unrolled stack) into the source
    tensor, or a single slice of it. This replaces one small strided copy per element with a few
    contiguous copies. Value-preserving; use ``disable`` to compare or to rule the rewrite out.

  - ``--torq-fold-scalar-matmul-mode=enable|disable`` (default ``enable``): replace a bf16 matmul
    whose reduction dim and output feature dim are both 1 (``[M,1] x [1,1]``, or
    ``[B,M,1] x [B,1,1]`` batched), with the default indexing maps and a zero init, by the
    elementwise product of its operands, so a row scale does not run a matmul kernel.
    Value-preserving; use ``disable`` to compare or to rule the rewrite out.

  - ``--torq-probe-avoid-relocation`` (default on): when the tile that the tile-fit probe picks
    fits only by relocating LRAM buffers (storing them to XRAM and loading them straight back),
    look for a slightly smaller tile that needs no relocation. Turn it off to compare with the
    relocating tile.

  - ``--torq-probe-relocation-min-bytes=<N>`` (default 32768): the least bytes a tile must
    relocate before that search runs. A smaller relocation costs less than an extra tile.

  - ``--torq-probe-relocation-max-tile-growth=<P>`` (default 25): the most additional tiles, in
    percent, accepted for a tile without relocation. The limit rounds down, so a tiling of 1 to
    3 tiles keeps its relocation.

  - ``--torq-disable-tnf-avoid-producer-recompute`` (default off): by default, tile-and-fuse does
    not fuse a producer whose slice is the same on every iteration of a tiled loop, e.g. the LHS
    producer of a matmul tiled along N, because it would be recomputed for every tile; the
    producer is computed once and its result read back from DRAM. This flag fuses such producers
    anyway. Use it to compare performance, or to rule this behaviour out when debugging tiling.

  - To see all supported Torq flags, run:

    ```{code} shell
    $ ../iree-build/third_party/iree/tools/torq-compile --help | grep "torq"
    ```

- To see debug at which pass a given error is emitted or find which pass is getting stuck use the following options:

    ```
    --debug-only pass-manager --mlir-disable-threading
    ```

## Troubleshooting

- If the model execution affects the system responsiveness (e.g. the NPU DMA traffic is starving other bus masters), try reducing the DMA burst size by lowering the MTU value:

  ```{code} shell
  $ torq-compile ... --torq-dma-in-mtu=2 --torq-dma-out-mtu=2
  ```

  The default MTU is 4 (16 beats / 128 bytes per AXI burst). Lowering it to 2 (4 beats / 32 bytes) reduces the burst length, giving other bus masters more opportunities to access DDR at the cost of lower NPU DMA throughput.

  
