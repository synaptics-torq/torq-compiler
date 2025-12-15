// Copyright 2024 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "Timer.h"
#include "TorqEventLog.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <atomic>

#ifdef ENABLE_ASTRA_MACHINA
#define DEF_HW_TYPE "astra_machina"
#else
#define DEF_HW_TYPE "sim"
#endif

namespace synaptics {

/// Virtual interface to lowest-level Torq driver
class TorqHw {
  public:
    enum class Type {
        UNDEFINED,
        SIMULATOR,
        AWS_FPGA,
        SOC_FPGA,
        ASTRA_MACHINA
    };

    virtual ~TorqHw() {}
    TorqHw(Type type, TorqEventLog* eventLog = nullptr);

    Type getType() const { return _type; }

    /// open the device
    virtual bool open() = 0;
    /// close the device
    virtual bool close() = 0;
    /// wait time-out
    virtual Timer::Duration waitTimeout() = 0;

    /// load resources required for the jobs
    virtual bool load() = 0;
    /// release the loaded resources
    virtual bool release() = 0;
    /// start device execution
    virtual bool start(uint32_t lramAddr);
    /// wait device to complete
    virtual bool wait(bool nssCfg = true, bool slice1Cfg = false, bool slice2Cfg = false, bool dmaInCfg = false, bool dmaOutCfg = false);
    /// end for the device execution to complete
    virtual bool end();
    /// write dataIn to LRAM, only for debugging purpose
    virtual bool writeLram(uint32_t addr, size_t size, const void *dataIn);
    /// read data from LRAM, only for debugging purpose
    virtual bool readLram(uint32_t addr, size_t size, void *dataOut) const;
    /// read data from DTCM, only for debugging purpose
    bool readDtcm(uint32_t addr, size_t size, void *dataOut) const;
    /// read data from ITCM, only for debugging purpose
    bool readItcm(uint32_t addr, size_t size, void *dataOut) const;

    /// start XRAM read access
    virtual const void * startXramReadAccess(uint32_t xramAddr) const = 0;
    /// end XRAM read access
    virtual bool endXramReadAccess() = 0;
    /// start XRAM write access
    virtual void * startXramWriteAccess(uint32_t xramAddr) = 0;
    /// end XRAM write access
    virtual bool endXramWriteAccess() = 0;

    /// write data to XRAM
    virtual bool writeXram(uint32_t addr, size_t size, const void *dataIn) = 0;
    /// read data from XRAM
    virtual bool readXram(uint32_t addr, size_t size, void *dataOut) const = 0;

    // Get elapsed time since last start()
    /// @return duration in microseconds
    Timer::Duration getTimeSinceStart() const { return _start_timer.get(); }

    // Get elapsed time since last open()
    /// @return duration in microseconds
    Timer::Duration getTimeSinceOpen() const { return _open_timer.get(); }

    // Get elapsed time since last wait() to read register
    /// @return duration in microseconds
    Timer::Duration getTimeSinceWait() const { return _wait_timer.get(); }

    void printNssRegs();

  private:
    /// wait for interrupt
    virtual bool wfi() = 0;
    /// clear interrupt
    virtual bool cli() = 0;
    /// write 32-bit data to the register
    virtual bool writeReg32(uint32_t addr, uint32_t data) = 0;
    /// write 32-bit data to the register
    virtual bool readReg32(uint32_t addr, uint32_t &data) const = 0;
    /// write data to LRAM
    virtual bool writeLram32(uint32_t addr, uint32_t data) = 0;
    /// read data from XRAM
    virtual bool readLram32(uint32_t addr, uint32_t &data) const = 0;

  protected:
    /// Torq hardware type
    const Type _type{Type::UNDEFINED};
  
    /// Job Timer reset at each start() call.
    Timer _start_timer;
    /// Job Timer reset at each open() call.
    Timer _open_timer;
    /// Job Timer reset at each wait() call.
    Timer _wait_timer;

    TorqEventLog* _eventLog = nullptr;
};

std::unique_ptr<TorqHw> newTorqHw(std::string hw_type, uint32_t xram_start_addr, size_t xram_size, std::string dump_dir = "", TorqEventLog* eventLog = nullptr);

} // namespace synaptics
