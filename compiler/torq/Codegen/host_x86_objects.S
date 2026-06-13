// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Embeds libclang_rt.builtins-x86_64.a inside the compiler binary so the
// LLVMCPU host dispatch link step can resolve soft-float helpers such as
// __truncsfbf2 without a file-system lookup. Mirrors the .incbin mechanism
// the CSS bootstrap uses for the riscv32 archives.

#ifdef __MACH__
    .section __DATA,__rodata
#else
    .section .rodata
#endif

    .globl _binary_libclang_rt_builtins_x86_64_a_start
_binary_libclang_rt_builtins_x86_64_a_start:
    .incbin HOST_X86_ARCHIVE_PATH

    .globl _binary_libclang_rt_builtins_x86_64_a_end
_binary_libclang_rt_builtins_x86_64_a_end:

    .globl _binary_libclang_rt_builtins_x86_64_a_size
    _binary_libclang_rt_builtins_x86_64_a_size = _binary_libclang_rt_builtins_x86_64_a_end - _binary_libclang_rt_builtins_x86_64_a_start

#ifdef __MACH__
    .globl __binary_libclang_rt_builtins_x86_64_a_start
    __binary_libclang_rt_builtins_x86_64_a_start = _binary_libclang_rt_builtins_x86_64_a_start
    .globl __binary_libclang_rt_builtins_x86_64_a_end
    __binary_libclang_rt_builtins_x86_64_a_end = _binary_libclang_rt_builtins_x86_64_a_end
#endif
