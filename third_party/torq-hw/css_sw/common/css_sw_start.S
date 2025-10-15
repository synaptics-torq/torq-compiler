// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  02/13/2025, Hongjie Guan

.section .init
.global _start
_start:
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop
    la sp, __stack_pointer
    // zero out .bss
    la t0, __bss_start
    la t1, __bss_end
l0: sw zero, 0(t0)
    addi t0, t0, 4
    bltu t0, t1, l0
    // call main
    jal ra, css_sw_main
    // halt
    .word 0x08000073

