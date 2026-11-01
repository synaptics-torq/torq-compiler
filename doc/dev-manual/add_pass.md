## Adding a New Pass

1. Create pass definition in Passes.td
2. Add constructor function declaration in Passes.h
3. Add pass to pipeline
4. Create file for the pass
5. Define the class for pass
6. Define the `runOnOperation()` function
7. Define the constructor for the pass
8. (Optional) Define the pattern class
9. Add the .cpp file for the pass to the CMakeLists.txt

### Example

Example with implementation details of the NarrowFloatSelectPass.
The complete set of code is available under *compiler/torq/Codegen*
1. Create a copy of any pass available in the Passes.td.
   Change the name of the pass to `NarrowFloatSelect`
   Change the first template argument of InterfacePass. The first argument of InterfacePass is the
   commandline argument used to identify the pass. Here changed the name to *torq-narrow-float-select*
   Change summary to provide a brief summary of the pass.
   Change the `constructor` to a custom construction name, here changed to *mlir::syna::torq::createNarrowFloatSelectPass()*.
   Add the dependentDialects as needed.
   Refer [Passes.td](https://github.com/synaptics-torq/torq-compiler/blob/main/compiler/torq/Codegen/Passes.td)
2. In Passes.h add declaration for the constructor we have named above.
   `std::unique_ptr<InterfacePass<FunctionOpInterface>> createNarrowFloatSelectPass()`
   In PassesDetail.h add `#define GEN_PASS_DEF_NARROWFLOATSELECT` to generate the base class of the pass.
3. Add the pass to a pipeline using the `addPass` method, for example in *TORQLowerExecutableTargetPass.cpp*.
   NarrowFloatSelect is added to the CPU program pipeline in *CompileCpuProgramsPass.cpp*.
4. Create the file `NarrowFloatSelectPass.cpp` in the folder *compiler/torq/Codegen*
5. Define the ***NarrowFloatSelectPass*** class in the file. It should inherit the generated
   *impl::NarrowFloatSelectBase* using CRTP
   `class NarrowFloatSelectPass : public impl::NarrowFloatSelectBase<NarrowFloatSelectPass>`
6. Define the `runOnOperation()` function. Refer
   [NarrowFloatSelectPass](https://github.com/synaptics-torq/torq-compiler/blob/main/compiler/torq/Codegen/NarrowFloatSelectPass.cpp)
7. Define the `createNarrowFloatSelectPass()` constructor. It should return a unique_ptr to InterfacePass.
   Refer the *NarrowFloatSelectPass.cpp* for more info.
8. Define the pattern rewrite class as needed. Here the class name is `NarrowSelectOfExtF`. It inherits from *OpRewritePattern*.
   Define its `matchAndRewrite()` function. Refer the `NarrowSelectOfExtF` class inside NarrowFloatSelectPass.cpp.
9. Add `NarrowFloatSelectPass.cpp` to the source list in *compiler/torq/Codegen/CMakeLists.txt*.
