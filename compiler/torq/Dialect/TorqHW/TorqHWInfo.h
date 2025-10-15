#pragma once

#include <cstdint>

namespace mlir::syna::torq {

/** Various constants describing the Torq hardware.
 The constants are used in the NDL generation and in the code generation.
 \todo Switch to a more flexible approach that can support different hardware configurations.
*/
struct HwInfo {
    enum {
        mac_count = 256, // Number MACs in the ALU (TODO: rename to alu_width as for act_width?)
        max_input = 64,  // TODO: rename to alu_group_width ?
        iram_seg_width = 72,
        iram_seg = 4,
        iram_depth = 1,
        wram_seg_width = 32,
        wram_width = 16,
        wram_depth = 1,
        pram_width = mac_count,  // Number of elements in the PRAM
        pram_dsize = 4,          // Size in bytes of each PRAM element
        pdat_width = pram_dsize, // pdat_width is used by hw example to be easy to understand
        pram_depth = 1,
        pram_bank = 2,
        breg_width = 8,
        bdat_width = breg_width, // bdat_width is used by hw example to be easy to understand
        qram_width = 8,
        act_limit = 4,  // TODO: rename to act_groups ?
        act_width = 16, // TODO: rename to act_count as for mac_count ?
        slice_count = 2,
        // Derived constants
        iram_width = iram_seg_width * iram_seg,
        bbus_width = breg_width * act_limit,
        table_lookup_count = 2, // ACT can only do 2 lookup at a time
        // bdat_width = breg_width
        // act_group_width = act_limit
        // act_groups = act_limit
        xram_size = 4294967296, // Size of the XRAM in bytes (4 GB)
        xram_page_size = 4096,  // Size of a page in the XRAM in bytes (matches Linux)
        cdma_lram_base_address =
            0x00000000, // REG_ADDR__TORQ_HV_LRAM in torq-hw/reg/torq_regs_host_view.h
        cdma_itcm_base_address =
            0x001c0000, // REG_ADDR__TORQ_HV_ITCM in torq-hw/reg/torq_regs_host_view.h
        cdma_dtcm_base_address =
            0x001d0000,         // REG_ADDR__TORQ_HV_DTCM in torq-hw/reg/torq_regs_host_view.h
        dtcm_size = 0x00008000, // REG_SIZE__TORQ_HV_DTCM in torq-hw/reg/torq_regs_host_view.h
        itcm_size = 0x00002000, // REG_SIZE__TORQ_HV_ITCM in torq-hw/reg/torq_regs_host_view.h
        css_itcm_base_address =
            0x00000000, // REG_ADDR__TORQ_CV_ITCM in torq-hw/reg/torq_regs_css_view.h
        css_dtcm_base_address =
            0x00010000,              // REG_ADDR__TORQ_CV_DTCM in torq-hw/reg/torq_regs_css_view.h,
        css_stack_size = 512,        // size of the CSS stack in bytes
        nss_max_program_size = 0x280 // maximum size of a NSS program in bytes
    };
};

} // namespace mlir::syna::torq
