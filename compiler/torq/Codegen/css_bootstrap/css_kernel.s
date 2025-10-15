.section .init
.global _start
_start:

    # setup global pointer
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop

    # setup stack pointer
    la sp, __stack_pointer

    # jump to c entry code
    jal ra, css_sw_main