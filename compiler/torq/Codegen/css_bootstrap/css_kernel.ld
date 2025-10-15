OUTPUT_ARCH(riscv)

/* see third_party/torq-hw/reg/torq_regs_css_view.h */
__itcm_size  = 0x00002000;
__dtcm_size  = 0x00008000;
__css_regs_size = 0x00001000;

__stack_pointer = __dtcm_start + __dtcm_size;
__stack_start = __stack_pointer - __stack_size;

ENTRY(_start)

MEMORY {
    ITCM (rx) : ORIGIN = __itcm_start, LENGTH = __itcm_size
    DTCM (rw) : ORIGIN = __dtcm_start, LENGTH = __dtcm_size
    CSS_REGS (rw) : ORIGIN = __css_regs_start, LENGTH = __css_regs_size
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
        /* FIXME: all data should be moved to DTCM but this complicates tiling so we keep it here for the moment, 
           this has a performance impact on code that uses tables */
        * (.sdata*)
        * (.rodata*)
        * (.data*)
        . = ALIGN(16);
        . = . + 0x20; /* append extra zero bytes at the end of .text */
        . = ALIGN(16);
        __text_end = .;
    } > ITCM

    .data : ALIGN(16) {
        __data_start = .;
        __global_pointer$ = __data_start + 0x800;
        . = ALIGN(16);
        __data_end = .;
    } > DTCM

    .css_regs (NOLOAD) : ALIGN(16) {
        css_regs = .;
        . += __css_regs_size;
    } > CSS_REGS
}
