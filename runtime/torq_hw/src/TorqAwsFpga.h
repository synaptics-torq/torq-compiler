// Copyright 2024 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "TorqHw.h"

namespace synaptics {

class TorqAwsFpga: public TorqHw {
  public:
    TorqAwsFpga(uint32_t xramStartAddr, size_t xramSize, TorqEventLog* eventLog = nullptr)
        : TorqHw(Type::AWS_FPGA, eventLog), _xramStartAddr(xramStartAddr), _xramSize(xramSize) {}

    bool open() override;
    bool close() override;
    Timer::Duration waitTimeout() override { return Timer::Duration(10000000); }
    
    bool writeXram(uint32_t addr, size_t size, const void *dataIn) override;
    bool readXram(uint32_t addr, size_t size, void *dataOut) const override;
    const void * startXramReadAccess(uint32_t xramAddr) const override;
    bool endXramReadAccess() override;
    void * startXramWriteAccess(uint32_t xramAddr) override;
    bool endXramWriteAccess() override;
    bool load() override { return true; };
    bool release() override { return true; };

  private:
    /// XRAM start address
    const uint32_t _xramStartAddr;
    /// XRAM size
    const size_t _xramSize;

    typedef int PciHandle;
    PciHandle xram_pci_hdl{-1};
    PciHandle npu_pci_hdl{-1};
    uint8_t *_regVBase{};
    uint8_t *_lramVBase{};
    uint8_t *_xramVBase{};

    bool wfi() override;
    bool cli() override;
    bool writeReg32(uint32_t addr, uint32_t data) override;
    bool readReg32(uint32_t addr, uint32_t & data) const override;
    bool writeLram32(uint32_t addr, uint32_t data) override;
    bool readLram32(uint32_t addr, uint32_t & data) const override;
};

} // namespace synaptics

