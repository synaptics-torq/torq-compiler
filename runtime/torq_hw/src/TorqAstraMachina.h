// Copyright 2025 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "TorqHw.h"
#include "torq_kernel_uapi.h"

#define TORQ_NODE "/dev/torq"

#define DMABUF_USE_UNCACHED
#if defined (DMABUF_USE_UNCACHED)
#define DMABUF_NODE "/dev/dma_heap/system-cust-uncached"
#else
#define DMABUF_NODE "/dev/dma_heap/system"
#endif

namespace synaptics {

#define ALIGN_4K 4096

#define TORQ_IOCTL_WAIT_BITMASK_NSS 0
#define TORQ_IOCTL_WAIT_BITMASK_DMA_IN 1
#define TORQ_IOCTL_WAIT_BITMASK_DMA_OUT 2
#define TORQ_IOCTL_WAIT_BITMASK_SLC_0 3
#define TORQ_IOCTL_WAIT_BITMASK_SLC_1 4

class TorqAstraMachina: public TorqHw {
  public:
    TorqAstraMachina(uint32_t xramStartAddr, size_t xramSize);
    ~TorqAstraMachina();

    bool open() override;
    bool close() override;
    Timer::Duration waitTimeout() override { return Timer::Duration(10000000); }
    bool writeXram(uint32_t addr, size_t size, const void *dataIn) override;
    bool readXram(uint32_t addr, size_t size, void *dataOut) const override;
    bool writeLram(uint32_t addr, size_t size, const void *dataIn) override;
    bool readLram(uint32_t addr, size_t size, void *dataOut) const override;

    bool load() override;
    bool release() override;
    bool start(uint32_t lramAddr) override;
    bool wait(bool nssCfg = true, bool slice1Cfg = false, bool slice2Cfg = false, bool dmaInCfg = false, bool dmaOutCfg = false) override;
    bool end() override;

  private:
    int _torqDevNode;
    int _dmabufDevNode;
    const uint32_t _xramStartAddr;
    const size_t _xramSize;

    uint32_t _alignOffset;
    uint32_t _xramStartAligned;
    size_t _xramSizeAligned;
    int _dmabufHandle;
    uint8_t *_xramVBase{};

    uint32_t _networkId;
    bool _networkActive;

    bool wfi() override;
    bool cli() override;
    bool writeReg32(uint32_t addr, uint32_t data) override;
    bool readReg32(uint32_t addr, uint32_t & data) const override;
    bool writeLram32(uint32_t addr, uint32_t data) override;
    bool readLram32(uint32_t addr, uint32_t & data) const override;
    void alignXram();

    bool createNetwork();
    bool startNetwork();
    bool runNetwork(uint32_t codeEntry);
    bool waitNetwork(uint32_t waitBits);
    bool stopNetwork();
    bool destroyNetwork();

    bool setupXramSpace();
    bool startXramAccess() const;
    bool endXramAccess() const;

};

} // namespace synaptics

