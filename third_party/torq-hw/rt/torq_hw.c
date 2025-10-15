// Copyright 2023-2024 Synaptics Incorporated. All rights reserved.
// Created:  11/22/2023, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#include <stdint.h>
#include "torq_log.h"
#include "torq_reg_util.h"
#include "torq_regs_host_view.h"
#include "torq_nss_regs.h"
#include "torq_css_regs.h"
#include "torq_sys.h"
#include "torq_reg_access.h"

int torq_hw_start(torq_sys_t *sys, uint32_t start_lram_addr)
{
    torq_reg_write32(sys, RA_(NSS,CFG), RF_LSH(NSS,CFG_LINK_EN, 1) | RF_BMSK_LSH(NSS,CFG_DESC, start_lram_addr));  // set NSS CFG descriptor address
    torq_reg_write32(sys, RA_(NSS,CTRL), RF_LSH(NSS,CTRL_IEN_NSS, 1));  // enable NSS interrupt (source)
    torq_reg_write32(sys, RA_(CSS,IEN_HST), RF_LSH(CSS,IEN_HST_NSS, 1));  // enable NSS interrupt (for host)
    torq_reg_write32(sys, RA_(NSS,START), RF_LSH(NSS,START_NSS, 1));  // kick off NSS CFG agent
    return 0;
}

int torq_hw_wait(torq_sys_t *sys)
{
    torq_wfi(sys);  // wait for interrupt
    int i = 0;
    while (1) {
        uint32_t status = torq_reg_read32(sys, RA_(NSS,STATUS)); // poll NSS status
#ifdef TORQ_HW_DEBUG
        printf("Try %d: NSS_STATUS: %08x\n", i++, status);
#endif
        if (RF_FMSK_RSH(NSS,STATUS_NSS, status)) break;
    }
    printf("Wait done\n");
    return 0;
}

int torq_hw_end(torq_sys_t *sys)
{
    torq_reg_write32(sys, RA_(NSS,STATUS), 1);  // clear NSS status
    torq_cli(sys);  // clear interrupt
    return 0;
}

int torq_hw_set_dump(torq_sys_t *sys, const char *dump_dir)
{
    sys->dump_dir = dump_dir;
    return 0;
}
