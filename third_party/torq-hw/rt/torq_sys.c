// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  06/28/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#include <stdint.h>
#include <string.h>
#include "torq_log.h"
#include "torq_sys.h"

#if defined(COSIM)

#include "cosim.h"

int torq_sys_open(torq_sys_t *sys)
{
    memset(sys, 0, sizeof(*sys));
    sys->xram_base = 0x00000000;
    sys->xram_size = 0x00200000;
    sys->xram_vbase = cosim_p2v((uint32_t)(sys->xram_base));
    sys->lram_base  = 0x00000000;
    sys->lram_size = 0x00080000;
    sys->lram_vbase = (unsigned char *)~0L; //not supported
    sys->reg_base = 0x00000000;
    sys->reg_size = 0x00010000;
    sys->reg_vbase  = (unsigned char *)~0L; //not supported
    return 0;
}

int torq_sys_close(torq_sys_t *sys)
{
    return 0;
}

int torq_wfi(torq_sys_t *sys)
{
    cosim_tb_cmd("wfi", 0, 0);
    return 0;
}

int torq_cli(torq_sys_t *sys)
{
    return 0;
}

#elif defined(SV_TEST)

int torq_sys_open(torq_sys_t *sys)
{
    int r;
    memset(sys, 0, sizeof(*sys));
    sys->xram_base = 0x800000000; //ddr_c
    sys->xram_size = 0x400000000;
    sys->xram_vbase = (unsigned char *)~0L; //not supported
    sys->lram_base = 0x00000000;
    sys->lram_size = 0x001f0000;
    sys->lram_vbase = (unsigned char *)~0L; //not supported
    sys->reg_base = 0x00000000;
    sys->reg_size = 0x00010000;
    sys->reg_vbase  = (unsigned char *)~0L; //not supported
    return 0;
}

int torq_sys_close(torq_sys_t *sys)
{
    return 0;
}

int torq_wfi(torq_sys_t *sys)
{
    return 0;
}

int torq_cli(torq_sys_t *sys)
{
    return 0;
}

#elif defined(CMODEL)

#include "torq_cm.h"

int torq_sys_open(torq_sys_t *sys)
{
    memset(sys, 0, sizeof(*sys));
    sys->xram_size = 64*1024*1024;
    sys->xram_base = 0;
    sys->xram_vbase = malloc(sys->xram_size); xdie(!sys->xram_vbase);
    sys->lram_base = 0x00000000;
    sys->lram_size = 0x001f0000;
    sys->lram_vbase = (unsigned char *)~0L; //not supported
    sys->reg_base = 0x00000000;
    sys->reg_size = 0x00010000;
    sys->reg_vbase  = (unsigned char *)~0L; //not supported
    sys->cm = torq_cm_open(sys->xram_vbase, sys->xram_base, sys->xram_size); xdie(!sys->cm);
#ifdef TORQ_BEH_CSS_SW
    {
        int r;
        void css_sw_main(void *);
        r = torq_cm__set_css_cpu_code(sys->cm, css_sw_main); xdie(r<0);
    }
#endif
    return 0;
}

int torq_sys_close(torq_sys_t *sys)
{
    int r;
    free(sys->xram_vbase);
    r = torq_cm_close(sys->cm); xdie(r<0);
    return r;
}

int torq_wfi(torq_sys_t *sys)
{
    return 0;
}

int torq_cli(torq_sys_t *sys)
{
    return 0;
}

#elif defined(AWS_FPGA)

#include "fpga_pci.h"

int torq_sys_open(torq_sys_t *sys)
{
    int r;
    const int slot_id = 0;
    memset(sys, 0, sizeof(*sys));
    r = fpga_pci_attach(slot_id, 0, 4, 0, &sys->xram_pci_hdl);
    xdie_v(r, "unable to attach xram_pci_hdl\n");
    sys->xram_base = 0x800000000; //ddr_c
    sys->xram_size = 0x400000000;
    r = fpga_pci_get_address(sys->xram_pci_hdl, sys->xram_base, sys->xram_size, (void **)&sys->xram_vbase);

    r = fpga_pci_attach(slot_id, 0, 1, 0, &sys->npu_pci_hdl);
    xdie_v(r, "unable to attach npu_pci_hdl\n");
    sys->lram_base = 0x00000000;
    sys->lram_size  = 0x00080000;
    r = fpga_pci_get_address(sys->npu_pci_hdl, sys->lram_base, sys->lram_size, (void **)&sys->lram_vbase);
    sys->reg_base = 0x00000000;
    sys->reg_size = 0x00010000;
    r = fpga_pci_get_address(sys->npu_pci_hdl, sys->reg_base, sys->reg_size, (void **)&sys->reg_vbase);
    return 0;
}

int torq_sys_close(torq_sys_t *sys)
{
    int r;
    r = fpga_pci_detach(sys->xram_pci_hdl);
    xdie_v(r, "unable to detach xram_pci_hdl\n");
    r = fpga_pci_detach(sys->npu_pci_hdl);
    xdie_v(r, "unable to detach npu_pci_hdl\n");
    return 0;
}

int torq_wfi(torq_sys_t *sys)
{
    return 0;
}

int torq_cli(torq_sys_t *sys)
{
    return 0;
}

#elif defined(EMBEDDED_FW)

int torq_sys_open(torq_sys_t *sys)
{
    memset(sys, 0, sizeof(*sys));
    sys->xram_base = 0x60000000;
    sys->xram_size = 0x00200000;
    sys->xram_vbase = (unsigned char *)(size_t)(sys->xram_base);
    sys->lram_base = 0x20000000;
    sys->lram_size = 0x00080000;
    sys->lram_vbase = (unsigned char *)(size_t)(sys->lram_base);
    sys->reg_base = 0x40000000;
    sys->reg_size   = 0x00010000;
    sys->reg_vbase  = (unsigned char *)(size_t)(sys->reg_base);
    return 0;
}

int torq_sys_close(torq_sys_t *sys)
{
    return 0;
}

int torq_wfi(torq_sys_t *sys)
{
    return 0;
}

int torq_cli(torq_sys_t *sys)
{
    return 0;
}

#else //ASIC Host

int torq_sys_open(torq_sys_t *sys)
{
    memset(sys, 0, sizeof(*sys));
    sys->xram_base  = 0x00000000;
    sys->xram_size  = 0xf0000000;
    sys->xram_vbase = (unsigned char *)(size_t)(sys->xram_base);
    sys->lram_base  = 0xf8000000; //TODO: TBD, depending on SoC address map
    sys->lram_size  = 0x00080000;
    sys->lram_vbase = (unsigned char *)(size_t)(sys->lram_base);
    sys->reg_base   = 0xf8000000; //TODO: TBD, depending on SoC address map
    sys->reg_size = 0x00010000;
    sys->reg_vbase = (unsigned char *)(size_t)(sys->reg_base);
    return 0;
}

int torq_sys_close(torq_sys_t *sys)
{
    return 0;
}

int torq_wfi(torq_sys_t *sys)
{
    return 0;
}

int torq_cli(torq_sys_t *sys)
{
    return 0;
}

#endif
