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
$ torq-compile ... --torq-hw=<LRAM>:<Slices>:<CSS features>:<NSS features>
```
- `<LRAM>`: Size of LRAM (integer)
- `<slices>`: Number of SLICE blocks (integer)
- `<CSS features>`: indicates the css features to enable (at the moment "coral_v1")
- `<NSS features >`: indicates the NSS features to enable (at the moment "nss_v1")

#### Example Configurations

**SL2610 Configuration**

This is the default configuration for TorqV1 as found in SL2610 SoC family:
- LRAM: 512
- Slices: 2
- CSS features: coral_v1
- NSS features: nss_v1

**Custom configuration 1: enabling only 1 slice**
- LRAM: 512
- Slices: 1
- CSS features: coral_v1
- NSS features: nss_v1

  Compile with:
  ```shell
  $ torq-compile ... --torq-hw=512:1:coral_v1:nss_v1
  ```

**Custom configuration 2: use experimental support for TorqV2 (simulation only)**

- LRAM: 512
- Slices: 2
- CSS features: coral_v2
- NSS features: nss_v2

  Compile with:
  ```shell
  $ torq-compile ... --torq-hw=512:1:coral_v2:nss_v2
  ```

> **Note:** When compiling models with a smaller LRAM configuration, stability may vary due to the need for very small tile sizes. Support for this configuration is still under development. Full compatibility is not guaranteed in this release. For a more representative configuration, you can compile with 512kB size of LRAM and keep the rest of the configuration the same.
