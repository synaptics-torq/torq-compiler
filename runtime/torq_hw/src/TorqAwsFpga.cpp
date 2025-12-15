// Copyright 2024 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TorqAwsFpga.h"

#include "fpga_pci.h"

#include "reg/torq_regs_host_view.h"
#include "reg/torq_nss_regs.h"
#include "reg/torq_css_regs.h"

#include <iostream>
#include <cstring>
#include <mutex>

using namespace std;

namespace synaptics {

static mutex _fpga_lock;

bool TorqAwsFpga::open() {
    _fpga_lock.lock();
    const int slot_id = 0;
    if (fpga_pci_attach(slot_id, 0, 4, 0, &xram_pci_hdl) != 0) {
        cerr << "Failed to attach xram_pci_hdl" << endl;
        return false;
    }
    const uint64_t xram_base = 0x800000000; //ddr_c
    const uint64_t xram_size = 0x400000000;
    if (fpga_pci_get_address(xram_pci_hdl, xram_base, xram_size, (void **)&_xramVBase) != 0) {
        cerr << "Failed to get xram_vbase" << endl;
        return false;
    }

    if (fpga_pci_attach(slot_id, 0, 0, 0, &npu_pci_hdl) != 0) {
        cerr << "Failed to attach npu_pci_hdl" << endl;
        return false;
    }
    const uint64_t lram_base = 0x00000000;
    const uint64_t lram_size  = 0x00080000;
    if (fpga_pci_get_address(npu_pci_hdl, lram_base, lram_size, (void **)&_lramVBase) != 0) {
        cerr << "Failed to get lram_vbase" << endl;
        return false;
    }
    const uint64_t reg_base = 0x00000000;
    const uint64_t reg_size = 0x00010000;
    if (fpga_pci_get_address(npu_pci_hdl, reg_base, reg_size, (void **)&_regVBase) != 0) {
        cerr << "Failed to get reg_vbase" << endl;
        return false;
    }

    _open_timer.start();
    return true;
}

bool TorqAwsFpga::close() {
    _fpga_lock.unlock();
    if (fpga_pci_detach(xram_pci_hdl) != 0) {
        cerr << "Failed to detach xram_pci_hdl" << endl;
        return false;
    }
    if (fpga_pci_detach(npu_pci_hdl) != 0) {
        cerr << "Failed to detach npu_pci_hdl" << endl;
        return false;
    }
    return true;
}

bool TorqAwsFpga::wfi() {
    return true;
}

bool TorqAwsFpga::cli() {
    return true;
}

static void write32(uint8_t *base, uint32_t addr, uint32_t data)
{
    volatile uint32_t *p = (volatile uint32_t *)(base + addr);
    *p = data;
}

static uint32_t read32(const uint8_t *base, uint32_t addr)
{
    volatile uint32_t *p = (volatile uint32_t *)(base + addr);
    return *p;
}

bool TorqAwsFpga::writeReg32(uint32_t addr, uint32_t data) {
    write32(_regVBase, addr, data);
    return true;
}

bool TorqAwsFpga::readReg32(uint32_t addr, uint32_t & data) const {
    data = read32(_regVBase, addr);
    return true;
}

bool TorqAwsFpga::writeLram32(uint32_t addr, uint32_t data) {
    write32(_lramVBase, addr, data);
    return true;
}

bool TorqAwsFpga::readLram32(uint32_t addr, uint32_t & data) const {
    data = read32(_lramVBase, addr);
    return true;
}

static const size_t rmw_n = 16;
static const size_t rmw_m = rmw_n-1;
static const size_t a_lsh = 2;

static void xramWrite(uint8_t * base, uint32_t addr, size_t size, uint8_t *data)
{
    memcpy(base + (addr<<a_lsh), data, size);
}

static void xramRead(uint8_t * base, uint32_t addr, size_t size, uint8_t *data)
{
    memcpy(data, base + (addr<<a_lsh), size);
}

bool TorqAwsFpga::writeXram(uint32_t addr, size_t size, const void *dataIn) {
    if (addr + size > _xramSize + _xramStartAddr) {
        cerr << "Out of range" << endl;
        return false;
    }
    auto data = (const uint8_t *) dataIn;
    uint8_t buf[rmw_n << a_lsh];
    size_t a[3], n[3];
    a[0] = addr;
    n[0] = (a[0] & rmw_m) ? rmw_n - (a[0] & rmw_m) : 0;
    if (n[0] > size)
        n[0] = size;
    size -= n[0];
    a[1] = a[0] + n[0];
    n[1] = size & ~rmw_m;
    size -= n[1];
    a[2] = a[1] + n[1];
    n[2] = size;
    for (size_t i = 0; i < 3; i++) {
        if (!n[i])
            continue;
        if (i!=1) { //read-modify-write
            xramRead(_xramVBase, a[i]&~rmw_m, rmw_n, buf);
            memcpy(&buf[a[i]&rmw_m], data, n[i]);
            xramWrite(_xramVBase, a[i]&~rmw_m, rmw_n, buf);
            data += n[i];
        }
        else {
            while (n[i]) {
                memcpy(buf, data, rmw_n);
                xramWrite(_xramVBase, a[i], rmw_n, buf);
                a[i] += rmw_n;
                data += rmw_n;
                n[i] -= rmw_n;
            }
        }
    }
    return true;
}

bool TorqAwsFpga::readXram(uint32_t addr, size_t size, void *dataOut) const {
    if (addr + size > _xramSize + _xramStartAddr) {
        cerr << "Out of range" << endl;
        return false;
    }
    auto data = (uint8_t *)dataOut;
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
    for (size_t i=0; i<3; i++) {
        if (!n[i])
            continue;
        if (i!=1) { //read-modify-write
            xramRead(_xramVBase, a[i]&~rmw_m, rmw_n, buf);
            memcpy(data, &buf[a[i]&rmw_m], n[i]);
            data += n[i];
        }
        else {
            while (n[i]) {
                xramRead(_xramVBase, a[i], rmw_n, buf);
                memcpy(data, buf, rmw_n);
                a[i] += rmw_n;
                data += rmw_n;
                n[i] -= rmw_n;
            }
        }
    }
    return true;
}

const void * TorqAwsFpga::startXramReadAccess(uint32_t addr) const {
    return nullptr;
}

bool TorqAwsFpga::endXramReadAccess() {
    return false;
}

void * TorqAwsFpga::startXramWriteAccess(uint32_t addr) {
    return nullptr;
}

bool TorqAwsFpga::endXramWriteAccess() {
    return false;
}


}  // synaptics namespace
