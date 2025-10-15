// Copyright 2025 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TorqSoCFpga.h"
#include "reg/torq_regs_host_view.h"
#include "reg/torq_nss_regs.h"
#include "reg/torq_css_regs.h"

#include <iostream>
#include <cstring>
#include <mutex>

extern "C" {
  #include <sys/types.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <unistd.h>
}

using namespace std;

namespace synaptics {

static mutex _fpga_lock;
int devNode;
bool TorqSoCFpga::open() {
    _fpga_lock.lock();

    if ((_xramStartAddr + _xramSize) > DDR_MAP_REGION) {
        cerr << "Requested xram region is out of mapped region" << endl;
        return false;
    }

    devNode = ::open(TORQ_NODE, O_RDWR | O_SYNC);
    if (devNode <= 0) {
        cerr << "torq node not available" << endl;
        return false;
    }

    const uint64_t reg_offset = 0;        /*offset 0 maps regspace (including lvramspace) */
    _regVBase = (uint8_t *)mmap(NULL, REG_SIZE__TORQ_HV, PROT_READ | PROT_WRITE, MAP_SHARED, devNode, reg_offset);
    if (!_regVBase) {
        cerr << "error mapping torq regspace" << endl;
        ::close(devNode);
        devNode = 0;
        return false;
    }
    _lramVBase = _regVBase;

    _xramVBase = (uint8_t *)mmap(NULL, DDR_MAP_REGION, PROT_READ | PROT_WRITE, MAP_SHARED, devNode, MMAP_DDR_OFFSET);
    if (!_xramVBase) {
        cerr << "error mapping ddr region" << endl;
        munmap(_regVBase, REG_SIZE__TORQ_HV);
        _regVBase = NULL;
        _lramVBase = NULL;
        ::close(devNode);
        devNode = 0;
        return false;
    }
    return true;
}

bool TorqSoCFpga::close() {
    _fpga_lock.unlock();

    if (_xramVBase)
        munmap(_xramVBase, DDR_MAP_REGION);
    if (_regVBase)
        munmap(_regVBase, REG_SIZE__TORQ_HV);
    if (devNode)
        ::close(devNode);
    return true;
}

bool TorqSoCFpga::wfi() {
    return true;
}

bool TorqSoCFpga::cli() {
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

bool TorqSoCFpga::writeReg32(uint32_t addr, uint32_t data) {
    write32(_regVBase, addr, data);
    return true;
}

bool TorqSoCFpga::readReg32(uint32_t addr, uint32_t & data) const {
    data = read32(_regVBase, addr);
    return true;
}

bool TorqSoCFpga::writeLram32(uint32_t addr, uint32_t data) {
    write32(_lramVBase, addr, data);
    return true;
}

bool TorqSoCFpga::readLram32(uint32_t addr, uint32_t & data) const {
    data = read32(_lramVBase, addr);
    return true;
}

bool TorqSoCFpga::writeXram(uint32_t addr, size_t size, const void *dataIn) {
    const uint8_t *p = _xramVBase + addr;
    memcpy((void*)p, dataIn, size);
    return true;
}

bool TorqSoCFpga::readXram(uint32_t addr, size_t size, void *dataOut) const {
    const uint8_t *p = _xramVBase + addr;
    memcpy(dataOut, p, size);
    return true;
}

}// synaptics namespace
