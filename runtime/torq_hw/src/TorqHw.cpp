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

#include <atomic>
#include <cstring>
#include <iostream>
#include <thread>

using namespace std;

namespace synaptics {

TorqHw::TorqHw(Type type, TorqEventLog* eventLog)
    : _type(type), _eventLog(eventLog) {}

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

std::unique_ptr<TorqHw> newTorqHw(std::string hw_type, uint32_t xram_start_addr, size_t xram_size, std::string dump_dir, TorqEventLog* eventLog) {
#ifdef ENABLE_SIMULATOR
    if (hw_type == "sim") {
        return std::unique_ptr<TorqHw>(new TorqSimulator(xram_start_addr, xram_size, dump_dir, eventLog));
    }
#endif
#ifdef ENABLE_AWS_FPGA
    if (hw_type == "aws_fpga") {
        return std::unique_ptr<TorqHw>(new TorqAwsFpga(xram_start_addr, xram_size, eventLog));
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
    cerr << hw_type << ": Torq Hardware not supported" << endl;
    return nullptr;
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

    _start_timer.start();

    writeReg32(RA_(NSS,CFG), RF_LSH(NSS,CFG_LINK_EN, 1) | RF_BMSK_LSH(NSS,CFG_DESC, lramAddr));  // set NSS CFG descriptor address
    writeReg32(RA_(NSS,CTRL), RF_LSH(NSS,CTRL_IEN_NSS, 1));  // enable NSS interrupt (source)
    writeReg32(RA_(CSS,IEN_HST), RF_LSH(CSS,IEN_HST_NSS, 1));  // enable NSS interrupt (for host)
    // Ensure previous memory operations are visible to the device before starting the device
    std::atomic_thread_fence(std::memory_order_seq_cst);
    writeReg32(RA_(NSS,START), RF_LSH(NSS,START_NSS, 1));  // kick off NSS CFG agent
    // Ensure the device has started before continuing
    std::atomic_thread_fence(std::memory_order_seq_cst);
    LOGD << "TorqHw::start OK" << endl;
    return true;
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

    uint32_t reg{};
    while (1) {
        if (!readReg32(RA_(NSS,STATUS), reg)) {
            LOGE << "Cannot read from NSS STATUS";
            return false;
        }

        auto wait_duration = getTimeSinceWait();
        if (wait_duration > timeout) {
            LOGE << wait_duration << "(us) Timeout waiting for interrupt";
            printNssRegs();
            return false;
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
    return true;
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
    return 0;
}

bool TorqHw::readDtcm(uint32_t addr, size_t size, void *dataOut) const {
    assert(addr + size <= REG_SIZE__TORQ_HV_DTCM && "DTCM address out of range");
    return readLram(REG_ADDR__TORQ_HV_DTCM + addr, size, dataOut);
}

bool TorqHw::readItcm(uint32_t addr, size_t size, void *dataOut) const {
    assert(addr + size <= REG_SIZE__TORQ_HV_ITCM && "ITCM address out of range");
    return readLram(REG_ADDR__TORQ_HV_ITCM + addr, size, dataOut);
}

}  // synaptics namespace
