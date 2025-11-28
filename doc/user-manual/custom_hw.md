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
$ torq-compile ... --torq-hw=<LRAM>:<Slices>:<tiling_memory>:<CSS features>:<NSS features>
```
- `<LRAM>`: Size of LRAM (integer)
- `<slices>`: Number of SLICE blocks (integer)
- `<tiling_memory>`: Tiling memory size in kilobytes (kb)
    > **Tiling memory:** specifies the available memory for tiling operations. When an operation's memory requirement exceeds the LRAM value, the compiler will automatically tile the operation to fit within the specified tiling memory.
- `<CSS features>`: indicates the css features to enable (at the moment "+m")
- `<NSS features >`: indicates the css features to enable (at the moment "nss_v1")

#### Example Configurations

**SL2610 Configuration**

This is the default configuration:

- LRAM: 512
- Slices: 2
- Tiling memory: 450

**Custom configuration 1**
- Version: 1
- LRAM: 512
- Slices: 1
- Tiling memory: 450
- CSS features: +m
- NSS features: nss_v1

  Compile with:
  ```shell
  $ torq-compile ... --torq-hw=512:1:450:+m:nss_v1
  ```

**Custom configuration 2**

- LRAM: 512
- Slices: 1
- Tiling memory: specify as needed
- CSS features: +m
- NSS features: nss_v2

  Compile with:
  ```shell
  $ torq-compile ... --torq-hw=512,1,400,+m,nss_v2
  ```

> **Note:** When compiling models with a smaller LRAM configuration, stability may vary due to the need for very small tile sizes. Support for this configuration is still under development. Full compatibility is not guaranteed in this release. For a more representative configuration, you can compile with 512kB size of LRAM and keep the rest of the configuration the same.