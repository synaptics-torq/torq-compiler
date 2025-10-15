## Debug tips and tricks

- To inspect the IR generated use the ``--dump-compilation-phases-to=<dir-name>`` parameter.

- To see the full stack trace when an errors is generated use the ``--mlir-print-stacktrace-on-diagnostic`` parameter.

- To see the full IR when an error occour ``--mlir-print-ir-after-failure``.

- To avoid dumping huge attributes (for instance constants) : ``--mlir-elide-elementsattrs-if-larger=SIZE``

- To generate the debugging info in a directory instead of stderr: ``--mlir-print-ir-tree-dir=<dirname>``

- To print line numbers (locations) in the source input when printing IRs use ``--mlir-print-debuginfo --mlir-pretty-debuginfo --mlir-print-local-scope``

- To view more debug information during execution pass the parameter `--torq_debug` to `iree-run-module`

- A {term}`VMFB` file can be inspected using ``zip``.
    For example to inspect a Torq flat buffer install ``flatc`` on your host and then run a command
    like the following:

  ```{code} shell
  unzip  -p ../mobilenetv2.vmfb main_dispatch_0_torq_fb.fb > dump.bin
  flatc -t -o . \
      runtime/schemas/torq_executable_def.fbs -- dump.bin
  cat dump.json
  ```

- To convert a tosa file (which is actually a binary MLIR file) to the text representation::

  ```{code} shell
  iree-opt model.tosa  -o model.mlir
  ```

- To debug a hang on the runtime perform the following steps:

  1. Compile the test network until the ``executable-targets`` phase:

     ```{code} shell
     torq-compile --compile-to=executable-targets -o test.hw.mlir test.mlir
     ```

  2. Compile the resulting dump from the ``executable-targets`` phase:

     ```{code} shell
     torq-compile --compile-from=executable-targets -o test.vmfb test.hw.mlir
     ```

  3. Run the model with the ``--torq_debug`` option:

     ```{code} shell
     iree-run-module .... --torq_debug
     ```

  4. In the debug log the line where the hardware gets stuck

  
- To analyze the intermediate files used to create the CSS binary images append the option ``--torq-keep-css-linking-artifacts``
  to the compiler invocation. With this command the compiler will also print the linking command line used to generate the binary
  file. By executing the command line without the ``--oformat=binary`` parameter it is possible to get a elf binary.

- To debug an instruction fault in a CSS task you can add the flag ``--torq_qemu_trace_instructions`` to the runtime and then inspect 
  the ``/tmp/qemu_trace_XXXX`` produced during the execution.

- To print additional logs during execution of qemu based emulation use the flag ``--torq_qemu_verbose``.

- To kill an execution that hang during CSS emulation on qemu use the keystrokes ``Ctrl-A X`` (these keystrokes are consumed by qemu emulator).

- To run the compiler in single threaded mode we add the option ``--mlir-disable-threading``
