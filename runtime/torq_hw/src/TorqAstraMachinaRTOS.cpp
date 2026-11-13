// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TorqAstraMachina.h"
#include "TorqAstraMachinaRTOS.h"
#include "TorqUtils.h"
#include "torq_regs.h"

/* Forward declarations for driver-level functions (torq.c, C linkage).
 * Declared here directly to avoid a build-time dependency on torq.h. */
extern "C" {
    int      torq_get_hw_id(uint32_t *hw_id);
    void     torq_write_reg(uint32_t addr, uint32_t val);
    uint32_t torq_read_reg(uint32_t addr);
    void     torq_open(void);
    void     torq_close(void);
    void     torq_load(void);
    void     torq_release(void);
    void     torq_start(uint32_t lram_addr);
    int      torq_wait(int nss_cfg, int xr_cfg, int xw_cfg, int slc0_cfg, int slc1_cfg);
    void     torq_write_xram(uint32_t addr, size_t size, const void *data);
    void     torq_read_xram(uint32_t addr, size_t size, void *data);
    void     torq_write_lram(uint32_t addr, size_t size, const void *data);
    void     torq_read_lram(uint32_t addr, size_t size, void *data);
}

using namespace std;

namespace synaptics {

std::unique_ptr<TorqHw> getAstraMachina(uint32_t xramStartAddr, size_t xramSize) {
    return std::make_unique<TorqAstraMachinaRTOS>(xramStartAddr, xramSize);
}

TorqAstraMachinaRTOS::~TorqAstraMachinaRTOS() {
    close();
}

TorqAstraMachinaRTOS::TorqAstraMachinaRTOS(uint32_t xramStartAddr, size_t xramSize)
    : TorqHw(Type::ASTRA_MACHINA), _xramStartAddr(xramStartAddr), _xramSize(xramSize)
{
    LOGD << "TorqAstraMachinaRTOS Constructor: xramStartAddr: 0x" << xramStartAddr << " xramSize: " << xramSize;
}

bool TorqAstraMachinaRTOS::open() {
    LOGD << "TorqAstraMachinaRTOS::open";
    ::torq_open();
    return true;
}

bool TorqAstraMachinaRTOS::close() {
    ::torq_close();
    return true;
}

bool TorqAstraMachinaRTOS::acquire() {
    LOGD << "TorqAstraMachinaRTOS::acquire";
    if (!TorqHw::acquire()) {
        return false;
    }
    ::torq_load();
    return true;
}

bool TorqAstraMachinaRTOS::start(uint32_t lramAddr) {
    LOGD << "TorqAstraMachinaRTOS::start lramAddr=0x" << lramAddr;
    ::torq_start(lramAddr);
    return true;
}

bool TorqAstraMachinaRTOS::wait(bool nssCfg, bool slice1Cfg, bool slice2Cfg, bool dmaInCfg, bool dmaOutCfg) {
    int done = ::torq_wait(nssCfg, dmaInCfg, dmaOutCfg, slice1Cfg, slice2Cfg);
    if (!done) {
        LOGE << "TorqAstraMachinaRTOS::wait timeout";
        return false;
    }
    LOGD << "TorqAstraMachinaRTOS::wait completed";
    return true;
}

bool TorqAstraMachinaRTOS::end() {
    return true;
}

bool TorqAstraMachinaRTOS::wfi() {
    return true;
}

bool TorqAstraMachinaRTOS::cli() {
    return true;
}

bool TorqAstraMachinaRTOS::writeReg32(uint32_t addr, uint32_t data) {
    ::torq_write_reg(TORQ_BASE_ADDR + addr, data);
    return true;
}

bool TorqAstraMachinaRTOS::readReg32(uint32_t addr, uint32_t & data) const {
    data = ::torq_read_reg(TORQ_BASE_ADDR + addr);
    return true;
}

bool TorqAstraMachinaRTOS::writeLram32(uint32_t addr, uint32_t data) {
    ::torq_write_lram(addr, sizeof(data), &data);
    return true;
}

bool TorqAstraMachinaRTOS::readLram32(uint32_t addr, uint32_t & data) const {
    ::torq_read_lram(addr, sizeof(data), &data);
    return true;
}

bool TorqAstraMachinaRTOS::writeXram(uint32_t addr, size_t size, const void *dataIn) {
    ::torq_write_xram(addr, size, dataIn);
    return true;
}

bool TorqAstraMachinaRTOS::readXram(uint32_t addr, size_t size, void *dataOut) const {
    ::torq_read_xram(addr, size, dataOut);
    return true;
}

bool TorqAstraMachinaRTOS::writeLram(uint32_t addr, size_t size, const void *dataIn) {
    ::torq_write_lram(addr, size, dataIn);
    return true;
}

bool TorqAstraMachinaRTOS::readLram(uint32_t addr, size_t size, void *dataOut) const {
    ::torq_read_lram(addr, size, dataOut);
    return true;
}

const void * TorqAstraMachinaRTOS::startXramReadAccess(uint32_t xramAddr) const {
    return (const void *)(uintptr_t)xramAddr;
}

bool TorqAstraMachinaRTOS::endXramReadAccess() {
    return true;
}

void * TorqAstraMachinaRTOS::startXramWriteAccess(uint32_t xramAddr) {
    return (void *)(uintptr_t)xramAddr;
}

bool TorqAstraMachinaRTOS::endXramWriteAccess() {
    return true;
}

bool TorqAstraMachinaRTOS::release() {
    if (!TorqHw::release()) {
        return false;
    }
    ::torq_release();
    return true;
}

bool TorqAstraMachinaRTOS::isHwCompatible(int32_t hwId) const {
    uint32_t currentHwId = 0;
    if (::torq_get_hw_id(&currentHwId) != 0) {
        LOGE << "Cannot verify hardware compatibility - hardware ID query failed";
        return false;
    }

    bool isCompatible = (currentHwId == static_cast<uint32_t>(hwId));
    if (!isCompatible) {
        LOGE << "Hardware mismatch: executable requires hw_id=" << hwId
             << ", but current hardware has hw_id=" << currentHwId;
    }
    return isCompatible;
}

} // namespace synaptics
