# Handling Unsupported Ops


## CSS Fallback 

The Torq hardware is composed of two subsystems:

- **NSS:** NPU Sub System (Neural Processing Unit)
- **CSS:** General Purpose CPU Subsystem

When compiling models, some operations may not be supported by the NSS. For those unsupported ops, the system can automatically fall back to the CSS. This ensures that all model operations can be executed, even if they are not natively supported by the NSS.

Emulation allows users to test and validate CSS programs without requiring access to physical CSS hardware. By simulating the hardware environment using tools like QEMU, we can compile and execute CSS binaries on a virtual RISC-V system.

```{important}
If you are using the provided Docker environment, all QEMU dependencies are already pre-installed. For other environments, please refer to [Getting Started](./getting_started.md) for the list of required libraries and packages, including QEMU and other dependencies needed for emulation.
```

CSS programs can be compiled in the following formats:

- By default, they are compiled to a bare metal image that can run on CSS hardware.
- When the option `--torq-css-qemu` is specified, the binary is compiled to a RISC-V bare metal image compatible with CSS but using a memory map suitable to run in `qemu-system-riscv32`.
- Optionally, you can pass `--torq-disable-slices` to disable the use of NSS, so the entire model runs on the CSS.

The runtime supports emulation with QEMU and it executes the CSS binaries in QEMU.

To compile the program using QEMU emulation, pass the following arguments at the time of compilation:

```{code} shell
$ torq-compile ... --iree-input-type=tosa --torq-css-qemu
```


##  Host Fallback

- By default, Host fallback is enabled. If the operation cannot be compiled for NSS or CSS, it will automatically fall back to Host execution.

You can disable Host fallback by passing the following option:

```{code} shell
$ torq-compile ... --torq-disable-host
```