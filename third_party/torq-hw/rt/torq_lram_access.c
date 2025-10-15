// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  06/28/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

//WARNING:
//  Direct accessing LRAM is not recommended!
//  It should only be used in the following scenarios:
//    (1) To load the small "bootstrap" DMA descriptor during initialization
//    (2) To backdoor load LRAM in some DV test cases
//    (3) For debug
//  Use DMA whenever possible.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "torq_log.h"
#include "torq_file_io.h"
#include "torq_sys.h"
#include "torq_lram_access.h"

#if defined(COSIM)

#include "cosim.h"

static void _lram_write32(torq_sys_t *sys, size_t addr, uint32_t data)
{
    cosim_ahb_write32(sys->lram_base + addr, data);
}

static uint32_t _lram_read32(torq_sys_t *sys, size_t addr)
{
    return cosim_ahb_read32(sys->lram_base + addr);
}

#elif defined(SV_TEST)

#include "fpga_pci_sv.h"

static void _lram_write32(torq_sys_t *sys, size_t addr, uint32_t data)
{
    int r;
    r = fpga_pci_bar1_poke(0L, sys->lram_base + addr, data); xdie(r);
}

static uint32_t _lram_read32(torq_sys_t *sys, size_t addr)
{
    int r;
    uint32_t data;
    r = fpga_pci_bar1_peek(0L, sys->lram_base + addr, &data); xdie(r);
    return data;
}

#elif defined(CMODEL)

#include "torq_cm.h"

static void _lram_write32(torq_sys_t *sys, size_t addr, uint32_t data)
{
    int r;
    r = torq_cm_write32(sys->cm, sys->lram_base + addr, &data); xdie(r);
}

static uint32_t _lram_read32(torq_sys_t *sys, size_t addr)
{
    int r;
    uint32_t data;
    r = torq_cm_read32(sys->cm, sys->lram_base + addr, &data); xdie(r);
    return data;
}

#else

static void _lram_write32(torq_sys_t *sys, size_t addr, uint32_t data)
{
    volatile uint32_t *p = (volatile uint32_t *)(sys->lram_vbase + addr);
    *p = data;
}

static uint32_t _lram_read32(torq_sys_t *sys, size_t addr)
{
    uint32_t data;
    volatile uint32_t *p = (volatile uint32_t *)(sys->lram_vbase + addr);
    data = *p;
    return data;
}

#endif

int torq_lram_write(torq_sys_t *sys, size_t addr, size_t size, uint8_t *data)
{
    size_t i;
    const size_t rmw_n = 4;
    const size_t rmw_m = rmw_n-1;
    uint32_t buf;
    size_t a[3], n[3];
    a[0] = addr;
    n[0] = (a[0]&rmw_m)?rmw_n-(a[0]&rmw_m):0;
    if (n[0]>size) n[0] = size;
    size -= n[0];
    a[1] = a[0]+n[0];
    n[1] = size&~rmw_m;
    size -= n[1];
    a[2] = a[1]+n[1];
    n[2] = size;
    for (i=0; i<3; i++) {
        if (!n[i]) continue;
        if (i!=1) { //read-modify-write
            buf = _lram_read32(sys, a[i]&~rmw_m);
            memcpy(((uint8_t *)&buf)+(a[i]&rmw_m), data, n[i]);
            _lram_write32(sys, a[i]&~rmw_m, buf);
            data += n[i];
        }
        else {
            while (n[i]) {
                memcpy(&buf, data, rmw_n);
                _lram_write32(sys, a[i], buf);
                a[i] += rmw_n;
                data += rmw_n;
                n[i] -= rmw_n;
            }
        }
    }
    return 0;
}

int torq_lram_read(torq_sys_t *sys, size_t addr, size_t size, uint8_t *data)
{
    size_t i;
    const size_t rmw_n = 4;
    const size_t rmw_m = rmw_n-1;
    uint32_t buf;
    size_t a[3], n[3];
    a[0] = addr;
    n[0] = (a[0]&rmw_m)?rmw_n-(a[0]&rmw_m):0;
    if (n[0]>size) n[0] = size;
    size -= n[0];
    a[1] = a[0]+n[0];
    n[1] = size&~rmw_m;
    size -= n[1];
    a[2] = a[1]+n[1];
    n[2] = size;
    for (i=0; i<3; i++) {
        if (!n[i]) continue;
        if (i!=1) { //read-modify-write
            buf = _lram_read32(sys, a[i]&~rmw_m);
            memcpy(data, ((uint8_t *)&buf)+(a[i]&rmw_m), n[i]);
            data += n[i];
        }
        else {
            while (n[i]) {
                buf = _lram_read32(sys, a[i]);
                memcpy(data, &buf, rmw_n);
                a[i] += rmw_n;
                data += rmw_n;
                n[i] -= rmw_n;
            }
        }
    }
    return 0;
}

#ifndef TORQ_RT_NO_FILE_IO

int torq_lram_load_from_file(torq_sys_t *sys, size_t addr, size_t word_size, size_t count, const char *fmt, const char *fname)
{
    int r;
    size_t n;
    uint8_t buf[1024];
    FILE *fp = 0L;
    xdie(sizeof(buf)%word_size);
    count *= word_size;
    fp = torq_fopen(fname, "r", fmt);
    xdie_v(!fp, "unable to open file: %s\n", fname);
    if (!fp) return -1;
    while (count) {
        n = count>sizeof(buf)?sizeof(buf):count;
        r = torq_fread(buf, word_size, n/word_size, fp, fmt); xdie(r<1);
        r = torq_lram_write(sys, addr, n, buf); xdie(r<0);
        addr += n;
        count -= n;
    }
    fclose(fp);
    return 0;
}

int torq_lram_save_to_file(torq_sys_t *sys, size_t addr, size_t word_size, size_t count, const char *fmt, const char *fname)
{
    int r;
    size_t n;
    uint8_t buf[1024];
    FILE *fp = 0L;
    xdie(sizeof(buf)%word_size);
    count *= word_size;
    fp = torq_fopen(fname, "w", fmt);
    xdie_v(!fp, "unable to open file: %s\n", fname);
    if (!fp) return -1;
    while (count) {
        n = count>sizeof(buf)?sizeof(buf):count;
        r = torq_lram_read(sys, addr, n, buf); xdie(r<0);
        r = torq_fwrite(buf, word_size, n/word_size, fp, fmt); xdie(r<1);
        addr += n;
        count -= n;
    }
    fclose(fp);
    return 0;
}

#endif
