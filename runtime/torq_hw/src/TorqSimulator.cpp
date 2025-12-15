// Copyright 2024 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TorqSimulator.h"
#include "KelvinCSSSimulation.h"
#include "QemuCSSSimulation.h"

#include "torq_cm.h"
#include "css_sw/common/css_sw_reg_inc.h"
#include "reg/torq_css_regs.h"
#include "reg/torq_nss_regs.h"
#include "reg/torq_regs_host_view.h"

#include "iree/hal/local/executable_library.h"
#include "iree/base/internal/flags.h"

#include <cstring>
#include <iostream>

IREE_FLAG(bool, torq_enable_kelvin, false, "Enable Kelvin CSS simulation")

using namespace std;

namespace synaptics {

bool TorqSimulator::open() {
    cm = torq_cm_open(_xram.data(), _xramStartAddr, _xram.size());

    if (!_dump_dir.empty()) {
        _job_dump_dir = _dump_dir + "/job0";
        torq_cm_set_dump(cm, _job_dump_dir.c_str());
    }

#ifdef TORQ_KELVIN_SIMULATOR
    auto simulator = FLAG_torq_enable_kelvin ? &run_cpu_kelvin_binary : &run_cpu_qemu_binary;
#else
    auto simulator = &run_cpu_qemu_binary;
#endif

    torq_cm__set_css_cpu_code(cm, simulator);
    
    if (!cm) {
        cerr << "Failed to open CModel" << endl;
        return false;
    }

    _open_timer.start();
    return true;
}

bool TorqSimulator::start(uint32_t lramAddr) {
    if (!_dump_dir.empty()) {
        _job_dump_dir = _dump_dir + "/job" + to_string(job_id);
        torq_cm_set_dump(cm, _job_dump_dir.c_str());
    }

    job_id++;
    return TorqHw::start(lramAddr);
}

bool TorqSimulator::close() { return torq_cm_close(cm) >= 0; }

bool TorqSimulator::wfi() { return true; }

bool TorqSimulator::cli() { return true; }

bool TorqSimulator::writeReg32(uint32_t addr, uint32_t data) {
    return torq_cm_write32(cm, addr, &data) >= 0;
}

bool TorqSimulator::readReg32(uint32_t addr, uint32_t &data) const {
    return torq_cm_read32(cm, addr, &data) >= 0;
}

bool TorqSimulator::writeLram32(uint32_t addr, uint32_t data) {
    return torq_cm_write32(cm, addr, &data) >= 0;
}

bool TorqSimulator::readLram32(uint32_t addr, uint32_t &data) const {
    return torq_cm_read32(cm, addr, &data) >= 0;
}

bool TorqSimulator::writeXram(uint32_t addr, size_t size, const void *dataIn) {
    if (addr + size > _xram.size() + _xramStartAddr) {
        cerr << "Out of range write" << " addr " << addr << " size " << size << " _xram.size() "
             << _xram.size() << " _xramStartAddr " << _xramStartAddr << endl;
        return false;
    }
    uint8_t *p = _xram.data() + (size_t)addr - _xramStartAddr;
    memcpy(p, dataIn, size);
    return true;
}

bool TorqSimulator::readXram(uint32_t addr, size_t size, void *dataOut) const {
    if (addr + size > _xram.size() + _xramStartAddr) {
        cerr << "Out of range read " << " addr " << addr << " size " << size << " _xram.size() "
             << _xram.size() << " _xramStartAddr " << _xramStartAddr << endl;
        return false;
    }
    const uint8_t *p = _xram.data() + addr - _xramStartAddr;
    memcpy(dataOut, p, size);
    return true;
}


const void * TorqSimulator::startXramReadAccess(uint32_t addr) const {
    return _xram.data() + addr - _xramStartAddr;
}

bool TorqSimulator::endXramReadAccess() {
    return true;
}

void * TorqSimulator::startXramWriteAccess(uint32_t addr) {
    return _xram.data() + addr - _xramStartAddr;
}

bool TorqSimulator::endXramWriteAccess() {
    return true;
}

} // namespace synaptics
