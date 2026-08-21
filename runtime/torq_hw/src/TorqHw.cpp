// Copyright 2024 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TorqUtils.h"
#include "TorqHw.h"
#ifdef ENABLE_SIMULATOR
#include "TorqSimulator.h"
#endif
#ifdef ENABLE_AWS_FPGA
#include "TorqAwsFpga.h"
#endif
#ifdef ENABLE_SOC_FPGA
#include "TorqSoCFpga.h"
#endif
#ifdef ENABLE_ASTRA_MACHINA
#include "TorqAstraMachina.h"
#endif

#include "reg/torq_regs_host_view.h"
#include "reg/torq_nss_regs.h"
#include "reg/torq_css_regs.h"
#include "reg/torq_reg_util.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <vector>
#include <thread>

using namespace std;

namespace synaptics {

TorqHw::TorqHw(Type type) : _type(type) {}

void TorqHw::printNssRegs() {

    uint32_t reg{};
    if (readReg32(RA_(NSS,STATUS), reg)) {
        LOGV << "NSS_STATUS (0x" << std::hex << RA_(NSS,STATUS) << " ) Value: 0x" << reg << " [ "
            << "NSS:" << RF_FMSK_RSH(NSS, STATUS_NSS, reg)
            << " XR:" << RF_FMSK_RSH(NSS, STATUS_XR, reg)
            << " XW:" << RF_FMSK_RSH(NSS, STATUS_XW, reg)
            << " SLC0:" << RF_FMSK_RSH(NSS, STATUS_SLC0, reg)
            << " SLC1:" << RF_FMSK_RSH(NSS, STATUS_SLC1, reg) << " ]";
    }

    uint32_t cfg{};
    if (readReg32(RA_(NSS,CFG), cfg)) {
        LOGV << "NSS_CFG (0x" << std::hex << RA_(NSS,CFG) << " ) Value: 0x" << std::hex << cfg << " [ "
            << "CFG_DESC: 0x" << RF_FMSK_RSH(NSS, CFG_DESC, cfg)
            << " CFG_LINK_EN:" << RF_FMSK_RSH(NSS, CFG_LINK_EN, cfg) << " ]";
    } else {
        LOGV << "Cannot read from NSS CFG";
    }

    uint32_t ctrl{};
    if (readReg32(RA_(NSS,CTRL), ctrl)) {
        LOGV << "NSS_CTRL (0x" << std::hex << RA_(NSS,CTRL) << " ) Value: 0x" << std::hex << ctrl << " [ "
            << "IEN_NSS:" << RF_FMSK_RSH(NSS, CTRL_IEN_NSS, ctrl)
            << " IEN_XR:" << RF_FMSK_RSH(NSS, CTRL_IEN_XR, ctrl)
            << " IEN_XW:" << RF_FMSK_RSH(NSS, CTRL_IEN_XW, ctrl)
            << " IEN_SLC0:" << RF_FMSK_RSH(NSS, CTRL_IEN_SLC0, ctrl)
            << " IEN_SLC1:" << RF_FMSK_RSH(NSS, CTRL_IEN_SLC1, ctrl) << " ]";
    } else {
        LOGV << "Cannot read from NSS CTRL";
    }

    uint32_t start{};
    if (readReg32(RA_(NSS,START), start)) {
        LOGV << "NSS_START (0x" << std::hex << RA_(NSS,START) << " ) Value: 0x" << std::hex << start << " [ "
            << "NSS:" << RF_FMSK_RSH(NSS, START_NSS, start)
            << " XR:" << RF_FMSK_RSH(NSS, START_XR, start)
            << " XW:" << RF_FMSK_RSH(NSS, START_XW, start)
            << " SLC0:" << RF_FMSK_RSH(NSS, START_SLC0, start)
            << " SLC1:" << RF_FMSK_RSH(NSS, START_SLC1, start) << " ]";
    } else {
        LOGV << "Cannot read from NSS START";
    }
}

std::unique_ptr<TorqHw> newTorqHw(std::string hw_type, uint32_t xram_start_addr, size_t xram_size) {
#ifdef ENABLE_SIMULATOR
    if (hw_type == "sim") {
        return std::unique_ptr<TorqHw>(new TorqSimulator(xram_start_addr, xram_size));
    }
#endif
#ifdef ENABLE_AWS_FPGA
    if (hw_type == "aws_fpga") {
        return std::unique_ptr<TorqHw>(new TorqAwsFpga(xram_start_addr, xram_size));
    }
#endif
#ifdef ENABLE_SOC_FPGA
    if (hw_type == "soc_fpga") {
        return std::unique_ptr<TorqHw>(new TorqSoCFpga(xram_start_addr, xram_size));
    }
#endif
#ifdef ENABLE_ASTRA_MACHINA
    if (hw_type == "astra_machina") {
        return std::unique_ptr<TorqHw>(new TorqAstraMachina(xram_start_addr, xram_size));
    }
#endif
    assert(false && "Unsupported TorqHw type");
    LOGE << hw_type << ": Torq Hardware not supported";
    return nullptr;
}

bool TorqHw::attachBinding(
    const TorqDeviceBuffer &buffer, uint32_t xramAddr, size_t dataOffset, size_t size
) {
    auto *data = static_cast<const uint8_t *>(buffer.mapped) + dataOffset;
    return writeXram(xramAddr, size, data);
}

bool TorqHw::detachBinding(
    const TorqDeviceBuffer &buffer, uint32_t xramAddr, size_t dataOffset, size_t size
) {
    auto *data = static_cast<uint8_t *>(buffer.mapped) + dataOffset;
    return readXram(xramAddr, size, data);
}

constexpr uint32_t cssDebugBufferDataSize = 16 - 8; // 16 bytes total, 8 bytes for read/write counters, 8 bytes for payload
constexpr uint32_t cssDebugBufferDtcmAddr = REG_SIZE__TORQ_HV_DTCM - cssDebugBufferDataSize - 8; // 8 bytes for read/write counters
constexpr uint32_t cssDebugBufferWriteAddr = cssDebugBufferDtcmAddr;
constexpr uint32_t cssDebugBufferReadAddr = cssDebugBufferDtcmAddr + 4;
constexpr uint32_t cssDebugBufferDataAddr = cssDebugBufferDtcmAddr + 8;

void TorqHw::setupCssDebugBuffer() {
    uint32_t zero = 0;
    writeDtcm(cssDebugBufferWriteAddr, sizeof(uint32_t), &zero);
    writeDtcm(cssDebugBufferReadAddr, sizeof(uint32_t), &zero);
    _css_debug_buffer_read_count = 0;
}

bool TorqHw::start(uint32_t lramAddr) {
#ifdef TORQ_DEVICE_DEBUG
    uint32_t reg{};
    uint32_t cfg{};
    if (!readReg32(RA_(NSS,STATUS), reg)) {
        LOGE << "Cannot read from NSS STATUS";
        return false;
    }
    if (!readReg32(RA_(NSS,CFG), cfg)) {
        LOGE << "Cannot read from NSS CFG";
        return false;
    }
    printf("Before START NSS_STATUS: %08x CFG: %08x\n", reg, cfg);
#endif

    if (isCssDebugBufferCallbackEnabled()) {                
        setupCssDebugBuffer();        
    }

    _start_timer.start();

    writeReg32(RA_(NSS,CFG), RF_LSH(NSS,CFG_LINK_EN, 1) | RF_BMSK_LSH(NSS,CFG_DESC, lramAddr));  // set NSS CFG descriptor address
    writeReg32(RA_(NSS,CTRL), RF_LSH(NSS,CTRL_IEN_NSS, 1));  // enable NSS interrupt (source)
    writeReg32(RA_(CSS,IEN_HST), RF_LSH(CSS,IEN_HST_NSS, 1));  // enable NSS interrupt (for host)
    // Ensure previous memory operations are visible to the device before starting the device
    std::atomic_thread_fence(std::memory_order_seq_cst);
    writeReg32(RA_(NSS,START), RF_LSH(NSS,START_NSS, 1));  // kick off NSS CFG agent
    // Ensure the device has started before continuing
    std::atomic_thread_fence(std::memory_order_seq_cst);
    LOGD << "TorqHw::start OK";
    return true;
}

void TorqHw::consumeCssDebugBuffer() {
    
    uint32_t debugWriteCount = 0;

    if (!readDtcm(cssDebugBufferWriteAddr, sizeof(debugWriteCount), &debugWriteCount)) {
        LOGE << "Failed to read debug write counter";        
        return;
    }
    
    const int available = debugWriteCount - _css_debug_buffer_read_count;

    if (available == 0) {
        return;
    }

    // the write pointer wrapped, we first read everything from the read pointer to the end of the buffer
    if (available < 0) {
        // read all data from read counter to end of the buffer
        const int bytesToRead = cssDebugBufferDataSize - _css_debug_buffer_read_count;
        std::vector<uint8_t> payload(bytesToRead);
        if (!readDtcm(cssDebugBufferDataAddr + _css_debug_buffer_read_count, bytesToRead, payload.data())) {
            LOGE << "Failed to read debug buffer (first chunk)";
            return;
        }
        logCssMessage(reinterpret_cast<const char *>(payload.data()), bytesToRead);
        _css_debug_buffer_read_count = 0;
    }
    
    // read all data from the read counter to the write counter (we now are sure it's below the write pointer)
    if (available != 0) {
        const int bytesToRead = debugWriteCount - _css_debug_buffer_read_count;
        std::vector<uint8_t> payload(bytesToRead);
        if (!readDtcm(cssDebugBufferDataAddr + _css_debug_buffer_read_count, bytesToRead, payload.data())) {
            LOGE << "Failed to read debug buffer (second chunk)";
            return;
        }
        logCssMessage(reinterpret_cast<const char *>(payload.data()), bytesToRead);
        _css_debug_buffer_read_count += bytesToRead;
    }

    if (!writeDtcm(cssDebugBufferReadAddr, sizeof(_css_debug_buffer_read_count), &_css_debug_buffer_read_count)) {
        LOGE << "Failed to update debug read counter";
    }

}

void TorqHw::startCssDebugBufferPolling() {

    if (!isCssDebugBufferCallbackEnabled()) {
        return;
    }

    _css_debug_buffer_polling = true;

    // we need a thread because in simulation readReg32 hangs till the device is done
    _css_debug_buffer_thread = std::thread([this]() {

        while (_css_debug_buffer_polling) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            consumeCssDebugBuffer();
        }

        // make sure we consume any remaining logs after the wait is finished
        consumeCssDebugBuffer();
    });
}

void TorqHw::stopCssDebugBufferPolling() {

    if (!_css_debug_buffer_thread.joinable()) {
        return;
    }

    _css_debug_buffer_polling = false;
    _css_debug_buffer_thread.join();
}


bool TorqHw::wait(bool nssCfg, bool slice1Cfg, bool slice2Cfg, bool dmaInCfg, bool dmaOutCfg) {    

    _wait_timer.start();

    // wait for interrupt
    if (!wfi()) {
        LOGE << "Cannot wait for interrupt";
        return false;
    }

    LOGV << "Waiting for: " 
         << (nssCfg ? "NSS " : "")
         << (slice1Cfg ? "Slice1 " : "")
         << (slice2Cfg ? "Slice2 " : "")
         << (dmaInCfg ? "DMA In " : "")
         << (dmaOutCfg ? "DMA Out " : "");

    const auto timeout = waitTimeout();

    startCssDebugBufferPolling();

    bool result = true;

    uint32_t reg{};
    while (1) {        

        if (!readReg32(RA_(NSS,STATUS), reg)) {
            LOGE << "Cannot read from NSS STATUS";
            result = false;
            break;
        }

        auto wait_duration = getTimeSinceWait();
        if (wait_duration > timeout) {
            LOGE << wait_duration << "(us) Timeout waiting for interrupt";
            printNssRegs();
            result = false;
            break;
        }

        // poll status
        bool status = true;

        if (nssCfg) {
            status = status && RF_FMSK_RSH(NSS,STATUS_NSS, reg);
        }

        if (slice1Cfg) {
            status = status && RF_FMSK_RSH(NSS,STATUS_SLC0, reg);
        }

        if (slice2Cfg) {
            status = status && RF_FMSK_RSH(NSS,STATUS_SLC1, reg);
        }

        if (dmaInCfg) {
            status = status && RF_FMSK_RSH(NSS,STATUS_XR, reg);
        }

        if (dmaOutCfg) {
            status = status && RF_FMSK_RSH(NSS,STATUS_XW, reg);
        }

        if (status) {
            LOGD << "TorqHw::wait OK, NSS_STATUS: 0x" << std::hex << reg << std::dec;
            printNssRegs();
            break;
        }

    }

    // drain any remaining logs
    stopCssDebugBufferPolling();

    return result;
}

bool TorqHw::end() {
    // clear NSS status
    // This will only clear out the NSS status bit, not the other bits in the register which
    // may indicate that other HW threads are still running.
    if (!writeReg32(RA_(NSS,STATUS), 1)) {
        LOGE << "Cannot write to NSS STATUS";
        return false;
    }
    // clear interrupt
    if (!cli()) {
        LOGE << "Cannot clear interrupt";
        return false;
    }
    LOGD << "TorqHw::end OK";
    return true;
}


bool TorqHw::writeLram(uint32_t addr, size_t size, const void *dataIn)
{
    auto data = (const uint8_t *)dataIn;
    const size_t rmw_n = 4;
    const size_t rmw_m = rmw_n-1;
    uint32_t buf;
    uint32_t a[3], n[3];
    a[0] = addr;
    n[0] = (a[0] & rmw_m) ? rmw_n - (a[0] & rmw_m) : 0;
    if (n[0]>size)
        n[0] = size;
    size -= n[0];
    a[1] = a[0]+n[0];
    n[1] = size & ~rmw_m;
    size -= n[1];
    a[2] = a[1] + n[1];
    n[2] = size;
    for (size_t i=0; i<3; i++) {
        if (!n[i]) continue;
        if (i!=1) { //read-modify-write
            if (!readLram32(a[i]&~rmw_m, buf)) {
                LOGE << "Cannot read from LRAM";
                return false;
            }
            memcpy(((uint8_t *)&buf)+(a[i]&rmw_m), data, n[i]);
            if (!writeLram32(a[i]&~rmw_m, buf)) {
                LOGE << "Cannot write to LRAM";
                return false;
            }
            data += n[i];
        }
        else {
            while (n[i]) {
                memcpy(&buf, data, rmw_n);
                if (!writeLram32(a[i], buf)) {
                    LOGE << "Cannot write to LRAM";
                    return false;
                }
                a[i] += rmw_n;
                data += rmw_n;
                n[i] -= rmw_n;
            }
        }
    }
    return true;
}

bool TorqHw::readLram(uint32_t addr, size_t size, void *dataOut) const
{
    auto data = (uint8_t *)dataOut;
    const size_t rmw_n = 4;
    const size_t rmw_m = rmw_n-1;
    size_t a[3], n[3];
    a[0] = addr;
    n[0] = (a[0] & rmw_m) ? rmw_n - (a[0] & rmw_m) : 0;
    if (n[0]>size)
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
        uint32_t buf;
        if (i != 1) { //read-modify-write
            if (!readLram32(a[i] & ~rmw_m, buf)) {
                LOGE << "Cannot read from LRAM";
                return false;
            }
            memcpy(data, ((uint8_t *)&buf) + (a[i] & rmw_m), n[i]);
            data += n[i];
        }
        else {
            while (n[i]) {
                if (!readLram32(a[i], buf)) {
                    LOGE << "Cannot read from LRAM";
                    return false;
                }
                memcpy(data, &buf, rmw_n);
                a[i] += rmw_n;
                data += rmw_n;
                n[i] -= rmw_n;
            }
        }
    }
    return true;
}

bool TorqHw::writeDtcm(uint32_t addr, size_t size, const void *dataIn) {
    assert(addr + size <= REG_SIZE__TORQ_HV_DTCM && "DTCM address out of range");
    return writeLram(REG_ADDR__TORQ_HV_DTCM + addr, size, dataIn);
}

bool TorqHw::readDtcm(uint32_t addr, size_t size, void *dataOut) const {
    assert(addr + size <= REG_SIZE__TORQ_HV_DTCM && "DTCM address out of range");
    return readLram(REG_ADDR__TORQ_HV_DTCM + addr, size, dataOut);
}

bool TorqHw::writeItcm(uint32_t addr, size_t size, const void *dataIn) {
    assert(addr + size <= REG_SIZE__TORQ_HV_ITCM && "ITCM address out of range");
    return writeLram(REG_ADDR__TORQ_HV_ITCM + addr, size, dataIn);
}

bool TorqHw::readItcm(uint32_t addr, size_t size, void *dataOut) const {
    assert(addr + size <= REG_SIZE__TORQ_HV_ITCM && "ITCM address out of range");
    return readLram(REG_ADDR__TORQ_HV_ITCM + addr, size, dataOut);
}

std::optional<TorqDeviceBuffer> TorqHw::allocateDeviceBuffer(size_t size) {
    TorqDeviceBuffer buffer{};
    iree_status_t status =
        torq_hw_device_buffer_allocate(TORQ_HW_DEVICE_BUFFER_MODE_MALLOC, size, &buffer);
    if (!iree_status_is_ok(status)) {
        LOGE << "Failed to allocate device buffer: "
             << iree_status_code_string(iree_status_code(status));
        iree_status_ignore(status);
        return std::nullopt;
    }
    return buffer;
}

void TorqHw::freeDeviceBuffer(TorqDeviceBuffer &buffer) {
    iree_status_t status = torq_hw_device_buffer_free(&buffer);
    if (!iree_status_is_ok(status)) {
        LOGE << "Failed to free device buffer: "
             << iree_status_code_string(iree_status_code(status));
        iree_status_ignore(status);
    }
}

}  // synaptics namespace
