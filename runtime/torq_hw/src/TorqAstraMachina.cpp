// Copyright 2025 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TorqAstraMachina.h"
#include "TorqUtils.h"
#include "reg/torq_regs_host_view.h"
#include "reg/torq_nss_regs.h"
#include "reg/torq_css_regs.h"
#include <linux/dma-heap.h>
#include <linux/dma-buf.h>

#include <iostream>
#include <cstring>
#include <mutex>

extern "C" {
  #include <sys/types.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <unistd.h>
}

using namespace std;

namespace synaptics {

static mutex _session_lock;

TorqAstraMachina::~TorqAstraMachina() {
    // Clean up network if still active
    if (_networkActive) {
        stopNetwork();
        destroyNetwork();
    }
    close();
}

TorqAstraMachina::TorqAstraMachina(uint32_t xramStartAddr, size_t xramSize): _xramStartAddr(xramStartAddr), _xramSize(xramSize) {
    _xramVBase = NULL;
    _dmabufHandle = 0;
    _networkId = 0;
    _networkActive = false;
    alignXram();
}

void TorqAstraMachina::alignXram() {
    /* align xram start to 4K for mapping IOVA region */
    _xramStartAligned = _xramStartAddr;
    _xramSizeAligned = _xramSize;
    _alignOffset = _xramStartAddr%ALIGN_4K;

    if (_alignOffset) {
       _xramSizeAligned += _alignOffset;
    }
    _xramStartAligned -= _alignOffset;
    LOGD << "incoming xramstart " << _xramStartAddr << " size " << _xramSize << " adjusted: "
         << _xramStartAligned << " size " << _xramSizeAligned << " alignOffset " << _alignOffset << "\n";
}

bool TorqAstraMachina::open() {
    _session_lock.lock();

    _torqDevNode = ::open(TORQ_NODE, O_RDWR | O_SYNC);
    if (_torqDevNode <= 0) {
        cerr << "torq node not available" << endl;
        close();
        return false;
    }

    if (!setupXramSpace()) {
        cerr << "Failed to configure xram memory" << endl;
        close();
        return false;
    }

    if (!createNetwork()) {
        cerr << "Failed to create network during initialization" << endl;
        close();
        return false;
    }

    LOGD << "Network " << _networkId << " created on open()";
    return true;
}

bool TorqAstraMachina::close() {
    // Clean up network if still active
    if (_networkActive) {
        stopNetwork();
    }
    if (_networkId != 0) {
        destroyNetwork();
    }
    if ((_xramVBase != MAP_FAILED) && (_xramVBase != NULL)) {
        munmap(_xramVBase, _xramSize);
        _xramVBase = NULL;
    }
    if (_dmabufHandle) {
        ::close(_dmabufHandle);
        _dmabufHandle = 0;
    }
    if (_dmabufDevNode) {
        ::close(_dmabufDevNode);
        _dmabufDevNode = 0;
    }
    if (_torqDevNode) {
        ::close(_torqDevNode);
        _torqDevNode = 0;
    }
    _session_lock.unlock();
    return true;
}

bool TorqAstraMachina::wfi() {
    return true;
}

bool TorqAstraMachina::cli() {
    return true;
}

static void write32(uint8_t *base, uint32_t addr, uint32_t data)
{
    volatile uint32_t *p = (volatile uint32_t *)(base + addr);
    *p = data;
}

static uint32_t read32(const uint8_t *base, uint32_t addr)
{
    volatile uint32_t *p = (volatile uint32_t *)(base + addr);
    return *p;
}

bool TorqAstraMachina::writeReg32(uint32_t addr, uint32_t data) {
    /* register access should go via kernel module */
    return false;
}

bool TorqAstraMachina::readReg32(uint32_t addr, uint32_t & data) const {
    /* register access should go via kernel module */
    return false;
}

bool TorqAstraMachina::writeLram32(uint32_t addr, uint32_t data) {
    /* LRAM access should go via kernel module */
    return false;
}

bool TorqAstraMachina::readLram32(uint32_t addr, uint32_t & data) const {
    /* LRAM access should go via kernel module */
    return false;
}

bool TorqAstraMachina::startXramAccess() const {
#if defined (DMABUF_USE_UNCACHED)
    return true;
#else
    struct dma_buf_sync sync = {0};
    sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW;
    if (ioctl(_dmabufHandle, DMA_BUF_IOCTL_SYNC, &sync)) {
        cerr << "error in dmabuf start sync ioctl\n";
        return false;
    }
    return true;
#endif
}

bool TorqAstraMachina::endXramAccess() const {
#if defined (DMABUF_USE_UNCACHED)
    return true;
#else
    struct dma_buf_sync sync = {0};
    sync.flags = DMA_BUF_SYNC_END  | DMA_BUF_SYNC_RW;
    if (ioctl(_dmabufHandle, DMA_BUF_IOCTL_SYNC, &sync)) {
        cerr << "error in dmabuf end sync ioctl\n";
        return false;
    }
    return true;
#endif
}

bool TorqAstraMachina::writeXram(uint32_t addr, size_t size, const void *dataIn) {
    const uint8_t *p = _xramVBase + _alignOffset + (addr - _xramStartAddr);
    if ((p < _xramVBase) || (p + size) > (_xramVBase + _xramSizeAligned)) {
        cerr << "xram_write:acessing memory out of bounds\n";
        return false;
    }

    if (!startXramAccess()) {
        return false;
    }
    memcpy((void*)p, dataIn, size);
    return endXramAccess();
}

bool TorqAstraMachina::readXram(uint32_t addr, size_t size, void *dataOut) const {
    const uint8_t *p = _xramVBase + _alignOffset + (addr - _xramStartAddr);
    if ((p < _xramVBase) || (p + size) > (_xramVBase + _xramSizeAligned)) {
        cerr << "xram_read:acessing memory out of bounds\n";
        return false;
    }

    if (!startXramAccess()) {
         return false;
    }
    memcpy(dataOut, p, size);
    return endXramAccess();
}

bool TorqAstraMachina::writeLram(uint32_t addr, size_t size, const void *dataIn) {
    if (!_networkActive) {
        cerr << "Network not active, cannot write LRAM" << endl;
        return false;
    }

    struct torq_write_lram_req req;
    req.network_id = _networkId;
    req.addr = addr;
    req.size = size;
    req.data = (void *)dataIn;

    if (ioctl(_torqDevNode, TORQ_IOCTL_WRITE_LRAM, &req) < 0) {
        cerr << "LRAM write IOCTL failed: " << strerror(errno) << endl;
        return false;
    }

    return true;
}

bool TorqAstraMachina::readLram(uint32_t addr, size_t size, void *dataOut) const {
    if (!_networkActive) {
        cerr << "Network not active, cannot read LRAM" << endl;
        return false;
    }

    struct torq_read_lram_req req;
    req.network_id = _networkId;
    req.addr = addr;
    req.size = size;
    req.data = dataOut;

    if (ioctl(_torqDevNode, TORQ_IOCTL_READ_LRAM, &req) < 0) {
        cerr << "LRAM read IOCTL failed: " << strerror(errno) << endl;
        return false;
    }

    return true;
}

bool TorqAstraMachina::setupXramSpace() {
    _dmabufDevNode = ::open(DMABUF_NODE, O_RDWR);
    if (_dmabufDevNode <= 0) {
        cerr << "dmabuf node not available for xram buffer" << endl;
        return false;
    }

    struct dma_heap_allocation_data bufferReq = {0};
    bufferReq.len = _xramSizeAligned;
    bufferReq.fd_flags = O_RDWR;

    if (ioctl(_dmabufDevNode, DMA_HEAP_IOCTL_ALLOC, &bufferReq) < 0) {
        cerr << "error allocating buffer from heap " << DMABUF_NODE << ", check kernel config\n";
        return false;
    }

    _dmabufHandle = bufferReq.fd;
    LOGD << "(xram) dmabuf fd " << _dmabufHandle << ": heap " << DMABUF_NODE << ": size "<< _xramSize << "\n";

    _xramVBase = (uint8_t *)mmap(NULL, _xramSizeAligned, PROT_READ | PROT_WRITE, MAP_SHARED, _dmabufHandle, 0);
    if (_xramVBase == MAP_FAILED) {
        cerr << "error mapping ddr region" << endl;
        return false;
    }
    return true;
}

bool TorqAstraMachina::createNetwork() {
    torq_create_network_req createReq;
    createReq.xram_start = _xramStartAddr;
    createReq.dmabuf_fd = _dmabufHandle;
    createReq.network_id = 0;

    int ret = ioctl(_torqDevNode, TORQ_IOCTL_CREATE_NETWORK, &createReq);
    if (ret < 0) {
        LOGD << "Failed to create network via IOCTL: " << ret;
        return false;
    }

    _networkId = createReq.network_id;
    return true;
}

bool TorqAstraMachina::startNetwork() {
    torq_start_network_req startReq;
    startReq.network_id = _networkId;

    LOGD << "Starting network " << _networkId;

    int kMaxRetries = 5;
    int kRetryDelayMs = 20;
    int ret = -1;
    for (int i = 0; i < kMaxRetries; ++i) {
        ret = ioctl(_torqDevNode, TORQ_IOCTL_START_NETWORK, &startReq);
        if (ret == 0) {
            break;
        }
        if (errno != EBUSY) {
            cerr << "Failed to start network via IOCTL: " << strerror(errno) << endl;
            return false;
        }

        LOGD << "device busy, retrying in " << kRetryDelayMs << "ms...";
        usleep(kRetryDelayMs * 1000);
        if (i == kMaxRetries - 1) {
            cerr << "Failed to start network, device busy after " << kMaxRetries << " retries." << endl;
            return false;
        }
    }

    LOGD << "Network " << _networkId << " started successfully";
    return true;
}

bool TorqAstraMachina::runNetwork(uint32_t codeEntry) {
    torq_run_network_req runReq;
    runReq.network_id = _networkId;
    runReq.code_entry = codeEntry;

    LOGD << "Running job on network " << _networkId << " with code entry 0x" << hex << codeEntry << dec;

    int ret = ioctl(_torqDevNode, TORQ_IOCTL_RUN_NETWORK, &runReq);
    if (ret < 0) {
        cerr << "Failed to run network job via IOCTL: " << ret << endl;
        return false;
    }

    LOGD << "Job started on network " << _networkId;
    return true;
}

bool TorqAstraMachina::waitNetwork(uint32_t waitBits) {
    torq_wait_network_req waitReq;
    waitReq.network_id = _networkId;
    waitReq.wait_bits = waitBits;

    LOGD << "Waiting for job completion on network " << _networkId << " with wait bits 0x" << hex << waitBits << dec;

    int ret = ioctl(_torqDevNode, TORQ_IOCTL_WAIT_NETWORK, &waitReq);
    if (ret < 0) {
        cerr << "Failed to wait for network job via IOCTL: " << ret << endl;
        return false;
    }

    LOGD << "Job completed on network " << _networkId;
    return true;
}

bool TorqAstraMachina::stopNetwork() {
    if (_networkId == 0) {
        return true;
    }

    torq_stop_network_req stopReq;
    stopReq.network_id = _networkId;

    int ret = ioctl(_torqDevNode, TORQ_IOCTL_STOP_NETWORK, &stopReq);
    if (ret < 0) {
        LOGD << "Failed to stop network " << _networkId << " via IOCTL: " << ret;
        return false;
    }

    LOGD << "Network " << _networkId << " stopped successfully";
    return true;
}

bool TorqAstraMachina::destroyNetwork() {
    if (_networkId == 0) {
        return true;
    }

    torq_destroy_network_req destroyReq;
    destroyReq.network_id = _networkId;

    int ret = ioctl(_torqDevNode, TORQ_IOCTL_DESTROY_NETWORK, &destroyReq);
    if (ret < 0) {
        LOGD << "Failed to destroy network " << _networkId << " via IOCTL: " << ret;
        return false;
    }

    LOGD << "Network " << _networkId << " destroyed successfully";
    _networkId = 0;
    _networkActive = false;
    return true;
}

bool TorqAstraMachina::load() {
    if (!_networkId) {
        cerr << "unable to load, network not setup yet\n";
        return false;
    }
    if (!startNetwork()) {
        cerr << "Failed to start network for ExecutionContext\n";
        return false;
    }
    _networkActive = true;
    return true;
}

bool TorqAstraMachina::release() {
    if (_networkActive) {
        stopNetwork();
        _networkActive = false;
    }
    return true;
}

bool TorqAstraMachina::start(uint32_t lramAddr) {
    if (!_networkActive) {
        LOGD << " network is not active yet\n";
        return false;
    }

    if (!runNetwork(lramAddr)) {
        cerr << "Failed to run network job" << endl;
        return false;
    }

    _start_timer.start();
    return true;
}

bool TorqAstraMachina::wait(bool nssCfg, bool slice1Cfg, bool slice2Cfg, bool dmaInCfg, bool dmaOutCfg) {
    uint32_t waitBits = 0;
    if (nssCfg) waitBits |= (1 << TORQ_IOCTL_WAIT_BITMASK_NSS);
    if (dmaInCfg) waitBits |= (1 << TORQ_IOCTL_WAIT_BITMASK_DMA_IN);
    if (dmaOutCfg) waitBits |= (1 << TORQ_IOCTL_WAIT_BITMASK_DMA_OUT);
    if (slice1Cfg) waitBits |= (1 << TORQ_IOCTL_WAIT_BITMASK_SLC_0);
    if (slice2Cfg) waitBits |= (1 << TORQ_IOCTL_WAIT_BITMASK_SLC_1);

    if (!waitNetwork(waitBits)) {
        cerr << "Failed to wait for network job completion" << endl;
        return false;
    }

    return true;
}

bool TorqAstraMachina::end() {
    //  wait() already clears interrupt status
    return true;
}

}// synaptics namespace
