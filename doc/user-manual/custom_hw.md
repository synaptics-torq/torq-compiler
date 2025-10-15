## Compiling Models for Custom Synaptics SoCs

This explains how to compile models for custom Synaptics SoCs using the Torq compiler.

### NSS Subsystem Overview

The NSS (NPU Sub System) in Synaptics SoCs consists of the following components:

- **LRAM:** Fast local memory for data storage and processing.
- **DMA:** Direct Memory Access block, capable of reading and writing data between system memory and LRAM.
- **SLICE:** Data processing blocks that read, process, and write back data to LRAM.

You can configure the LRAM size and the number of slices at model compilation time to test custom hardware configurations.

### Custom Hardware Configuration

To compile a model for a custom hardware configuration, use the following options:

```{code} shell
$ torq-compile ... --torq-hw=custom --torq-hw-custom=<LRAM>,<slices>,<tiling_memory>
```
- `<LRAM>`: Size of LRAM (integer)
- `<slices>`: Number of SLICE blocks (integer)
- `<tiling_memory>`: Tiling memory size in kilobytes (kb)
    > **Tiling memory:** specifies the available memory for tiling operations. When an operation's memory requirement exceeds the LRAM value, the compiler will automatically tile the operation to fit within the specified tiling memory.

#### Example Configurations

**Configuration 1**
- LRAM: 512
- Slices: 2
- Tiling memory: specify as needed  
  Compile with:
  ```shell
  $ torq-compile ... --torq-hw=custom --torq-hw-custom=512,2,400
  ```

**Configuration 2**
- LRAM: 256
- Slices: 1
- Tiling memory: specify as needed  
  Compile with:
  ```shell
  $ torq-compile ... --torq-hw=custom --torq-hw-custom=256,1,200
  ```

> **Note:** When compiling models with a 256 LRAM configuration, stability may vary due to the need for very small tile sizes. Support for this configuration is still under development. Full compatibility is not guaranteed in this release. For a more representative configuration, you can compile with 512 size of LRAM and keep the rest of the configuration the same.