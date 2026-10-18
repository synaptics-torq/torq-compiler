## Compiling Models for Synaptics SoCs

This explains how to select the target Synaptics SoC when compiling a model with the Torq compiler,
and how to describe a custom hardware configuration.

### NSS Subsystem Overview

The NSS (NPU Sub System) in Synaptics SoCs consists of the following components:

- **LRAM:** Fast local memory for data storage and processing.
- **DMA:** Direct Memory Access block, capable of reading and writing data between system memory and LRAM.
- **SLICE:** Data processing blocks that read, process, and write back data to LRAM.
- **CSS:** A RISC-V control core that runs the operations the slices do not handle.

The target hardware is selected at compile time with `--torq-hw`. A compiled model records the
hardware it was compiled for, and the runtime refuses to run it on a different one.

To list the supported SoCs:

```{code} shell
$ torq-compile --torq-hw=list
Available Torq Hardware Types:
  SL2610: Synaptics SL2610 SoC family
  SR200: Synaptics SR200 SoC family
```

### Supported SoCs

| | SL2610 | SR200 |
|---|---|---|
| Torq generation | TorqV1 | TorqV2 |
| LRAM | 512 KB | 256 KB |
| Slices | 2 | 1 |
| CSS core | RV32IM (`coral_v1`) | RV32IMF with the Zve32x vector extension (`coral_v2`) |
| NSS features | `nss_v1` | `nss_v2` |
| Host CPU | Arm Cortex-A (`aarch64-unknown-linux-gnu`) | Arm Cortex-M85 (`thumbv8.1m.main-none-eabihf`) |
| DDR bandwidth | 6.4 GB/s | 3.2 GB/s |
| Hardware ID | 0x0 | 0x1 |

**SL2610**

The SL2610 SoC family has a TorqV1 NPU with two slices and 512 KB of LRAM. This is the default
target.

```{code} shell
$ torq-compile ... --torq-hw=SL2610
```

**SR200**

The SR200 SoC family has a TorqV2 NPU: one slice and 256 KB of LRAM. Its CSS core adds
single-precision floating point and the Zve32x vector extension to the SL2610 core, so CSS
programs can use hardware floating point and vector instructions.

SR200 is an MCU: the host CPU is an Arm Cortex-M85 running Zephyr OS. The operations that run on
the host are compiled for this core with Helium (MVE, integer and floating point) and the FPU
(half, single and double precision) enabled.

```{code} shell
$ torq-compile ... --torq-hw=SR200
```

With half the LRAM and one slice, SR200 models are tiled into smaller pieces than on SL2610. The
compiler selects tile sizes for this automatically.

### Custom Hardware Configuration

To compile a model for a custom hardware configuration, use:

```{code} shell
$ torq-compile ... --torq-hw=<hw_id>:<lram_size_kb>:<slice_count>:<css_features>:<nss_features>[:<dma_throughput>[:<dma_factor>:<host_triple>:<host_cpu>:<host_cpu_features>]]
```

- `<hw_id>`: hardware ID recorded in the compiled model (`0x0` for SL2610, `0x1` for SR200).
  The runtime checks it against the hardware it runs on.
- `<lram_size_kb>`: size of LRAM in KB.
- `<slice_count>`: number of SLICE blocks.
- `<css_features>`: CSS core features, `coral_v1` or `coral_v2`.
- `<nss_features>`: NSS features, `nss_v1` or `nss_v2`.
- `<dma_throughput>` (default 8): DMA theoretical throughput in bytes per cycle, used by the
  compiler's cost estimates.
- `<dma_factor>` (default 1.0): fraction of the theoretical DMA throughput reached in practice.
  SL2610 uses 0.65, measured with 6.4 GB/s of DDR bandwidth. SR200 has 3.2 GB/s and uses 0.33.
- `<host_triple>`, `<host_cpu>`, `<host_cpu_features>` (defaults `native`, `host`, `host`): the
  LLVM target of the host CPU that runs the operations the NPU does not support. `native` compiles
  them for the machine running `torq-compile`. The built-in SoCs set their own host: SL2610 an
  `aarch64-unknown-linux-gnu` Cortex-A, SR200 a `thumbv8.1m.main-none-eabihf` Cortex-M85 with
  `+mve,+mve.fp,+fp-armv8d16,+fullfp16,+dsp`.

The compiler prints the parsed configuration.

#### Example Configurations

**SL2610 with only 1 slice**

```{code} shell
$ torq-compile ... --torq-hw=0x0:512:1:coral_v1:nss_v1
```

**SR200 NPU with 512 KB of LRAM**

```{code} shell
$ torq-compile ... --torq-hw=0x1:512:1:coral_v2:nss_v2:8:0.33:thumbv8.1m.main-none-eabihf:cortex-m85:+mve,+mve.fp,+fp-armv8d16,+fullfp16,+dsp
```

### Host CPU Override

To run a model on the simulator, compile the host operations for the machine that runs the
simulator instead of the SoC host CPU:

```{code} shell
$ torq-compile ... --torq-hw=SR200 --torq-target-host-triple=native
```

`--torq-target-host-triple`, `--torq-target-host-cpu` and `--torq-target-host-cpu-features`
override the host fields of any `--torq-hw` configuration. `native` also resets the CPU and its
features to those of the compiling machine. The test harness and `torq-lab` add
`--torq-target-host-triple=native` when they run on the simulator or the FPGA.

> **Note:** With a small LRAM, the compiler must use very small tiles. Some models may then fail
> to compile or run slower. 512 KB of LRAM with the rest of the configuration unchanged is a
> more representative setting.
