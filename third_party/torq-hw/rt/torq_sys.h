// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  06/28/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#ifndef TORQ_SYS_H_
#define TORQ_SYS_H_


#include <stddef.h>

#if defined(AWS_FPGA)
#include "fpga_pci.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


typedef struct torq_sys_t {
    size_t              xram_base;
    size_t              xram_size;
    unsigned char      *xram_vbase;
    size_t              lram_base;
    size_t              lram_size;
    unsigned char      *lram_vbase;
    size_t              reg_base;
    size_t              reg_size;
    unsigned char      *reg_vbase;
#if defined(AWS_FPGA)
    pci_bar_handle_t    xram_pci_hdl;
    pci_bar_handle_t    npu_pci_hdl;
#elif defined (CMODEL)
    void               *cm;
#endif
    const char         *dump_dir;
} torq_sys_t;

int torq_sys_open(torq_sys_t *sys);
int torq_sys_close(torq_sys_t *sys);

int torq_wfi(torq_sys_t *sys);
int torq_cli(torq_sys_t *sys);


#ifdef __cplusplus
}
#endif

#endif
