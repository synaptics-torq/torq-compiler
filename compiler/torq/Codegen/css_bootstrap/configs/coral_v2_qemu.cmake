set(CSS_CONFIG_MABI "ilp32f")
set(CSS_CONFIG_MARCH "rv32imf_zve32x")
set(CSS_CONFIG_MATTRS "+m,+f,+zve32x")
set(CSS_CONFIG_CFLAGS -DCSS_HW_QEMU -DENABLE_FP -DENABLE_VEC)

# we map the ITCM at the start of the DRAM (where the ROM code expects the BIOS to be)
# and we map DTCM just after, CSS registers are mapped in memory after the DTCM so that
# we can set them by editing the memory backing store.
# These values must be synchronized with the values used by TorqHW.

set(CSS_ITCM_START "0x80000000")
set(CSS_DTCM_START "0x80010000")
set(CSS_REGS_START "0x80020000")
