// Copyright 2025 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "TorqHw.h"

#define TORQ_NODE "/dev/torq"
#define DDR_MAP_REGION 0x6000000   /*DDR region to be mapped*/
#define MMAP_DDR_OFFSET 0x1000

namespace synaptics {

class TorqSoCFpga: public TorqHw {
  public:
    TorqSoCFpga(uint32_t xramStartAddr, size_t xramSize): _xramStartAddr(xramStartAddr), _xramSize(xramSize) {}

    bool open() override;
    bool close() override;
    Timer::Duration waitTimeout() override { return Timer::Duration(10000000); }
    bool writeXram(uint32_t addr, size_t size, const void *dataIn) override;
    bool readXram(uint32_t addr, size_t size, void *dataOut) const override;
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

