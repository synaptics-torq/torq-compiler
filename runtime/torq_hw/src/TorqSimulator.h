// Copyright 2024 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "TorqHw.h"
#include <string>
#include <vector>

namespace synaptics {

class TorqSimulator: public TorqHw {
  public:
    TorqSimulator(uint32_t xram_start_addr, size_t xram_size, std::string dump_dir = "", TorqEventLog* eventLog = nullptr)
        // Init memory to some non-null value. Kernels must not rely on uninitialized memory being at 0.
        // We use 0x77 which is visually visible and a big value as both an int and a float exponent
        // so it is more likely to show up in the results if used by mistake.
        : TorqHw(Type::SIMULATOR, eventLog), _xramStartAddr(xram_start_addr), _xram(xram_size, (uint8_t)0x77), _dump_dir{dump_dir} {}

    bool open() override;
    bool close() override;
    Timer::Duration waitTimeout() override { return Timer::Duration(100000000); }
    bool start(uint32_t lramAddr) override;

    bool writeXram(uint32_t addr, size_t size, const void *dataIn) override;
    bool readXram(uint32_t addr, size_t size, void *dataOut) const override;
    const void * startXramReadAccess(uint32_t xramAddr) const override;
    bool endXramReadAccess() override;
    void * startXramWriteAccess(uint32_t xramAddr) override;
    bool endXramWriteAccess() override;

    bool load() override { return true; };
    bool release() override { return true; };

  private:
    /// CModel handle
    void * cm{};

    /// XRAM start address
    const uint32_t _xramStartAddr;
    /// XRAM content
    std::vector<uint8_t> _xram;
    /// directory where to dump the CModel data
    std::string _dump_dir;
    /// current job dump dir (we need to keep a reference here
    /// because the cmodel is not going to copy it)
    std::string _job_dump_dir{};

    /// how many times start has been called
    int job_id{0};

    bool wfi() override;
    bool cli() override;
    bool writeReg32(uint32_t addr, uint32_t data) override;
    bool readReg32(uint32_t addr, uint32_t & data) const override;
    bool writeLram32(uint32_t addr, uint32_t data) override;
    bool readLram32(uint32_t addr, uint32_t & data) const override;
};

} // namespace synaptics

