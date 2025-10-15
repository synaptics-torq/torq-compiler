#pragma once

#include <stddef.h>
#include <stdint.h>

#define BINARY_OBJ(name)                                                                           \
    extern char _binary_##name##_end;                                                              \
    extern size_t _binary_##name##_size;                                                           \
    extern char _binary_##name##_start;

extern "C" {

BINARY_OBJ(css_kernel_ld)
BINARY_OBJ(css_kernel_s_o)
BINARY_OBJ(css_kernel_kelvin_c_o)
BINARY_OBJ(css_kernel_qemu_c_o)
BINARY_OBJ(css_libc_a)
BINARY_OBJ(css_libm_a)
BINARY_OBJ(css_compiler_rt_a)
}
