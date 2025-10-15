/*
Copyright 2024 Synaptics Incorporated. All rights reserved.
Created:  02/13/2025, Hongjie Guan
*/

OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv", "elf32-littleriscv")
OUTPUT_ARCH(riscv)

__itcm_start = 0x00000000;
__itcm_size  = 8K;
__dtcm_start = 0x00010000;
__dtcm_size  = 32K;

__stack_pointer = __dtcm_start + __dtcm_size;

ENTRY(_start)

MEMORY {
    ITCM (rx) : ORIGIN = __itcm_start, LENGTH = __itcm_size
    DTCM (rw) : ORIGIN = __dtcm_start, LENGTH = __dtcm_size
}

SECTIONS {
    . = 0;

    .init : {
        __text_start = .;
        * (.init)
        . = ALIGN(16);
    } > ITCM

    .text : ALIGN(16) {
        * (.text*)
        . = ALIGN(16);
        . = . + 0x20; /* append extra zero bytes at the end of .text */
        . = ALIGN(16);
        __text_end = .;
    } > ITCM

    .data : ALIGN(16) {
        __data_start = .;
        __global_pointer$ = __data_start + 0x800;
        * (.sdata*)
        * (.rodata*)
        * (.data*)
        . = ALIGN(16);
        __data_end = .;
    } > DTCM /* AT > ROM */

    .bss (NOLOAD) : ALIGN(16) {
        __bss_start = .;
        * (.bss*)
        * (COMMON)
        . = ALIGN(16);
        __bss_end = .;
    } > DTCM
}

