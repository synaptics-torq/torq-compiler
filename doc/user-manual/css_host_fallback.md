# Handling Unsupported Ops


## CSS Fallback 

The Torq hardware is composed of two subsystems:

- **NSS:** NPU Sub System (Neural Processing Unit)
- **CSS:** General Purpose CPU Subsystem

When compiling models, some operations may not be supported by the NSS. For those unsupported ops, the system can automatically fall back to the CSS. This ensures that all model operations can be executed, even if they are not natively supported by the NSS.

Emulation allows users to test and validate CSS programs without requiring access to physical CSS hardware. By simulating the hardware environment we can compile and execute CSS binaries on a virtual RISC-V system.

```{important}
If you are using the provided Docker environment, all dependencies are already pre-installed. For other environments, please refer to [Getting Started](./getting_started.md) for the list of required libraries and packages.
```

Optionally, you can pass `--torq-disable-slices` to disable the use of NSS, so the entire model runs on the CSS.

##  Host Fallback

- By default, Host fallback is enabled. If the operation cannot be compiled for NSS or CSS, it will automatically fall back to Host execution.

You can disable Host fallback by passing the following option:

```{code} shell
$ torq-compile ... --torq-disable-host
```