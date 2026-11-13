// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "TorqHw.h"

namespace synaptics {

class TorqAstraMachinaRTOS: public TorqHw {
  public:
    TorqAstraMachinaRTOS(uint32_t xramStartAddr, size_t xramSize);
    ~TorqAstraMachinaRTOS();

    bool open() override;
    bool close() override;
    Timer::Duration waitTimeout() override { return Timer::Duration(100000000); }
    bool writeXram(uint32_t addr, size_t size, const void *dataIn) override;
    bool readXram(uint32_t addr, size_t size, void *dataOut) const override;
    bool writeLram(uint32_t addr, size_t size, const void *dataIn) override;
    bool readLram(uint32_t addr, size_t size, void *dataOut) const override;
    const void * startXramReadAccess(uint32_t xramAddr) const override;
    bool endXramReadAccess() override;
    void * startXramWriteAccess(uint32_t xramAddr) override;
    bool endXramWriteAccess() override;

    bool acquire() override;
    bool release() override;
    bool start(uint32_t lramAddr) override;
    bool wait(bool nssCfg = true, bool slice1Cfg = false, bool slice2Cfg = false, bool dmaInCfg = false, bool dmaOutCfg = false) override;
    bool end() override;

    bool isHwCompatible(int32_t hwId) const override;

  private:
    const uint32_t _xramStartAddr;
    const size_t _xramSize;

    bool wfi() override;
    bool cli() override;
    bool writeReg32(uint32_t addr, uint32_t data) override;
    bool readReg32(uint32_t addr, uint32_t & data) const override;
    bool writeLram32(uint32_t addr, uint32_t data) override;
    bool readLram32(uint32_t addr, uint32_t & data) const override;

};

} // namespace synaptics