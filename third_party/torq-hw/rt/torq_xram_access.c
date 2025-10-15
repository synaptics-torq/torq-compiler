// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  06/28/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "torq_log.h"
#include "torq_file_io.h"
#include "torq_sys.h"
#include "torq_xram_access.h"

#if defined(SV_TEST)

static int _xram_write(torq_sys_t *sys, size_t addr, size_t size, uint8_t *data)
{
    const size_t a_lsh = 2; //log2(64/16)
    sv_fpga_start_buffer_to_cl(0, 2, (size<<a_lsh), (uint64_t)data, sys->xram_base + (addr<<a_lsh));
    return 0;
}

static int _xram_read(torq_sys_t *sys, size_t addr, size_t size, uint8_t *data)
{
    const size_t a_lsh = 2; //log2(64/16)
    sv_fpga_start_cl_to_buffer(0, 2, (size<<a_lsh), (uint64_t)data, sys->xram_base + (addr<<a_lsh));
    return 0;
}

#elif defined(AWS_FPGA)

static int _xram_write(torq_sys_t *sys, size_t addr, size_t size, uint8_t *data)
{
    const size_t a_lsh = 2; //log2(64/16)
    memcpy(sys->xram_vbase + (addr<<a_lsh), data, size);
    //memcpy(sys->xram_vbase + (addr<<a_lsh), data, (size<<a_lsh));
    return 0;
}

static int _xram_read(torq_sys_t *sys, size_t addr, size_t size, uint8_t *data)
{
    const size_t a_lsh = 2; //log2(64/16)
    memcpy(data, sys->xram_vbase + (addr<<a_lsh), size);
    //memcpy(data, sys->xram_vbase + (addr<<a_lsh), (size<<a_lsh));
    return 0;
}

#endif

#if defined(SV_TEST) || defined(AWS_FPGA)

int torq_xram_write(torq_sys_t *sys, size_t addr, size_t size, uint8_t *data)
{
    int r;
    size_t i;
    const size_t rmw_n = 16;
    const size_t rmw_m = rmw_n-1;
    const size_t a_lsh = 2; //log2(64/16)
    uint8_t buf[rmw_n<<a_lsh];
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
            r = _xram_read(sys, a[i]&~rmw_m, rmw_n, buf); xdie(r<0);
            memcpy(&buf[a[i]&rmw_m], data, n[i]);
            r = _xram_write(sys, a[i]&~rmw_m, rmw_n, buf); xdie(r<0);
            data += n[i];
        }
        else {
            while (n[i]) {
                memcpy(buf, data, rmw_n);
                r = _xram_write(sys, a[i], rmw_n, buf); xdie(r<0);
                a[i] += rmw_n;
                data += rmw_n;
                n[i] -= rmw_n;
            }
        }
    }
    return 0;
}

int torq_xram_read(torq_sys_t *sys, size_t addr, size_t size, uint8_t *data)
{
    int r;
    size_t i;
    const size_t rmw_n = 16;
    const size_t rmw_m = rmw_n-1;
    const size_t a_lsh = 2; //log2(64/16)
    uint8_t buf[rmw_n<<a_lsh];
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
            r = _xram_read(sys, a[i]&~rmw_m, rmw_n, buf); xdie(r<0);
            memcpy(data, &buf[a[i]&rmw_m], n[i]);
            data += n[i];
        }
        else {
            while (n[i]) {
                r = _xram_read(sys, a[i], rmw_n, buf); xdie(r<0);
                memcpy(data, buf, rmw_n);
                a[i] += rmw_n;
                data += rmw_n;
                n[i] -= rmw_n;
            }
        }
    }
    return 0;
}

#ifndef TORQ_RT_NO_FILE_IO

int torq_xram_load_from_file(torq_sys_t *sys, size_t addr, size_t word_size, size_t count, const char *fmt, const char *fname)
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
        r = torq_xram_write(sys, addr, n, buf); xdie(r<0);
        addr += n;
        count -= n;
    }
    fclose(fp);
    return 0;
}

int torq_xram_save_to_file(torq_sys_t *sys, size_t addr, size_t word_size, size_t count, const char *fmt, const char *fname)
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
        r = torq_xram_read(sys, addr, n, buf); xdie(r<0);
        r = torq_fwrite(buf, word_size, n/word_size, fp, fmt); xdie(r<1);
        addr += n;
        count -= n;
    }
    fclose(fp);
    return 0;
}

#endif

#else

int torq_xram_write(torq_sys_t *sys, size_t addr, size_t size, uint8_t *data)
{
    uint8_t *p = sys->xram_vbase + addr;
    memcpy(p, data, size);
    return 0;
}

int torq_xram_read(torq_sys_t *sys, size_t addr, size_t size, uint8_t *data)
{
    uint8_t *p = sys->xram_vbase + addr;
    memcpy(data, p, size);
    return 0;
}

#ifndef TORQ_RT_NO_FILE_IO

int torq_xram_load_from_file(torq_sys_t *sys, size_t addr, size_t word_size, size_t word_count, const char *fmt, const char *fname)
{
    int r;
    FILE *fp = 0L;
    uint8_t *p = sys->xram_vbase + addr;
    fp = torq_fopen(fname, "r", fmt);
    xdie_v(!fp, "unable to open file: %s\n", fname);
    if (!fp) return -1;
    r = torq_fread(p, word_size, word_count, fp, fmt); xdie(r<1);
    torq_fclose(fp);
    return 0;
}

int torq_xram_save_to_file(torq_sys_t *sys, size_t addr, size_t word_size, size_t word_count, const char *fmt, const char *fname)
{
    int r;
    FILE *fp = 0L;
    uint8_t *p = sys->xram_vbase + addr;
    fp = torq_fopen(fname, "w", fmt);
    xdie_v(!fp, "unable to open file: %s\n", fname);
    if (!fp) return -1;
    r = torq_fwrite(p, word_size, word_count, fp, fmt); xdie(r<1);
    torq_fclose(fp);
    return 0;
}

#endif

#endif
