// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  06/28/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#include <stdint.h>
#include "torq_log.h"
#include "torq_sys.h"

#if defined(COSIM)

#include "cosim.h"

void torq_reg_write32(torq_sys_t *sys, uint32_t addr, uint32_t data)
{
    cosim_ahb_write32(sys->reg_base + addr, data);
}

uint32_t torq_reg_read32(torq_sys_t *sys, uint32_t addr)
{
    return cosim_ahb_read32(sys->reg_base + addr);
}

#elif defined(SV_TEST)

#include "fpga_pci_sv.h"

void torq_reg_write32(torq_sys_t *sys, uint32_t addr, uint32_t data)
{
    int r;
    r = fpga_pci_bar1_poke(0L, sys->reg_base + addr, data); xdie(r);
}

uint32_t torq_reg_read32(torq_sys_t *sys, uint32_t addr)
{
    int r;
    uint32_t data;
    r = fpga_pci_bar1_peek(0L, sys->reg_base + addr, &data); xdie(r);
    return data;
}

#elif defined(CMODEL)

#include "torq_cm.h"

void torq_reg_write32(torq_sys_t *sys, uint32_t addr, uint32_t data)
{
#ifdef TORQ_HW_DEBUG
    printf("Writing addr %08x data %08x\n", addr, data);
#endif
    int r;
    r = torq_cm_set_dump(sys->cm, sys->dump_dir); xdie(r<0);
    r = torq_cm_write32(sys->cm, sys->reg_base + addr, &data); xdie(r<0);
}

uint32_t torq_reg_read32(torq_sys_t *sys, uint32_t addr)
{
    int r;
    uint32_t data;
    r = torq_cm_set_dump(sys->cm, sys->dump_dir); xdie(r<0);
    r = torq_cm_read32(sys->cm, sys->reg_base + addr, &data); xdie(r<0);
#ifdef TORQ_HW_DEBUG
    printf("Reading addr %08x data %08x\n", addr, data);
#endif
    return data;
}

#else

void torq_reg_write32(torq_sys_t *sys, uint32_t addr, uint32_t data)
{
#ifdef TORQ_HW_DEBUG
    printf("Writing addr %08x data %08x\n", addr, data);
#endif
    volatile uint32_t *p = (volatile uint32_t *)(sys->reg_vbase + addr);
    *p = data;
}

uint32_t torq_reg_read32(torq_sys_t *sys, uint32_t addr)
{
    volatile uint32_t *p = (volatile uint32_t *)(sys->reg_vbase + addr);
#ifdef TORQ_HW_DEBUG
    printf("Reading addr %08x data %08x\n", addr, *p);
#endif
    return *p;
}

#endif
