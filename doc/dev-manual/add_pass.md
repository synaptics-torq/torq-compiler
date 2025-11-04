## Adding a New Pass

1. Create pass definition in Passes.td
2. Add constructor function declaration in Passes.h
3. Add pass to pipeline
4. Create file for the pass
5. Define the class for pass
6. Define the `runOnOperation()` function
7. Define the contructor for the pass
8. (Optional) Define the pattern class
9. Add the .cpp file for the pass to the CMakeLists.txt

### Example

Example with implementation details of the KernelSelectionPass for the TorqHL to TorqHW conversion.
The complete set of code is available under *compiler/torq/Transforms*
1. Create a copy of any pass available in the Passes.td.
   Change the name of the pass to `KernelSelection` 
   Change the first template argument of InterfacePass. The first argument of InterfacePass is the 
   commandline argument used to identify the pass. Here changed the name to *torq-kernel-selection*
   Change summary to provide a brief summary of the pass.
   Change the `constructor` to a custom construction name, here changed to *mlir::syna::torq::createKernelSelectionPass()*.
   Add the dependentDialects as needed.
   Refer [Passes.td](https://github.com/synaptics-torq/torq-compiler/blob/main/compiler/torq/Codegen/Passes.td)
2. In Passes.h add declaration for the constructor we have named above. 
   `std::unique_ptr<InterfacePass<FunctionOpInterface>> createKernelSelectionPass()`
3. Add pass to the `TORQLowerExecutableTargetPass` inside the *TORQLowerExecutableTargetPass.cpp*  pipeline using `addPass` method.
   Here `pipeline.addPass(createKernelSelectionPass())`
4. Create the file `KernelSelectionPass.cpp` in the folder *compiler/torq/CodeGen* 
5. Define the ***KernelSelectionPass*** class in the file. It should inherit the *KernelSelectionBase* using CRTP
   `class KernelSelectionPass : public KernelSelectionBase<KernelSelectionPass>`
6. Define the `runOnOperation()` function. Refer 
   [KernelSelectionPass](https://github.com/synaptics-torq/torq-compiler/blob/main/compiler/torq/Codegen/KernelSelectionPass.cpp) 
7. Define the `createKernelSelectionPass()` constructor. It should return a unique_ptr to InterfacePass. 
   Refer the *KernelSelectionPass.cpp* for more info.
8. Define the pattern rewrite class as needed. Here the class name is `ConstPattern`. It inherits from *OpRewritePattern*.
   Define it's `matchAndRewrite()` function. Refer `ConstPattern` class inside KernelSelectionPass.cpp.