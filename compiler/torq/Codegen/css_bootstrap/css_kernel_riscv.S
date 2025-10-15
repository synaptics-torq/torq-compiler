// This macro includes a binary file and creates symbols to find start, end and size
.macro BINARY_OBJ section, name, path, file
// In MACH we need to declare a section for each object
#ifdef __MACH__
    .section __DATA,__\section
#else
    .section .rodata
#endif
    .globl _binary_\name\()_start
    _binary_\name\()_start:
    .incbin "\path\()/\file"
    .globl _binary_\name\()_end
    _binary_\name\()_end:
    .globl _binary_\name\()_size
    _binary_\name\()_size = _binary_\name\()_end - _binary_\name\()_start
// In MACH we need to prepend an additional "_" to the symbol names
#ifdef __MACH__
    .globl __binary_\name\()_start
    __binary_\name\()_start = _binary_\name\()_start
    .globl __binary_\name\()_end
    __binary_\name\()_end = _binary_\name\()_end
#endif
.endm

// Section name must be less than 16 char
BINARY_OBJ css_k_ld, css_kernel_ld, ".", "css_kernel.ld"
BINARY_OBJ css_k_s_o, css_kernel_s_o, ".", "css_kernel.s.o.data"
BINARY_OBJ css_k_kelvin, css_kernel_kelvin_c_o, ".", "css_kernel_kelvin.c.o.data"
BINARY_OBJ css_k_qemu, css_kernel_qemu_c_o, ".", "css_kernel_qemu.c.o.data"
BINARY_OBJ css_libc_a, css_libc_a, CSS_PREBUILT_DIR, "libc.a"
BINARY_OBJ css_libm_a, css_libm_a, CSS_PREBUILT_DIR, "libm.a"
BINARY_OBJ css_rt_a, css_compiler_rt_a, CSS_PREBUILT_DIR, "libclang_rt.builtins-riscv32.a"
