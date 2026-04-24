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
#include <linux/dma-buf.h>

#include <iostream>
#include <cstring>

extern "C" {
  #include <sys/types.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <unistd.h>
}

using namespace std;


#define TORQ_NODE "/dev/torq"

#define DMABUF_USE_UNCACHED

namespace synaptics {

TorqAstraMachina::~TorqAstraMachina() {
    // Clean up network if still active
    if (_networkActive) {
        stopNetwork();
        destroyNetwork();
    }
    close();
}

TorqAstraMachina::TorqAstraMachina(uint32_t xramStartAddr, size_t xramSize): TorqHw(Type::ASTRA_MACHINA), _xramStartAddr(xramStartAddr), _xramSize(xramSize) {
    _xramVBase = NULL;
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
    _torqDevNode = ::open(TORQ_NODE, O_RDWR | O_SYNC);
    if (_torqDevNode == kInvalidFd) {
        cerr << "torq node not available" << endl;
        close();
        return false;
    }

    iree_status_t node_status = torq_hw_dma_heap_node_acquire();
    if (!iree_status_is_ok(node_status)) {
        cerr << "failed to acquire dma heap nodes: "
             << iree_status_code_string(iree_status_code(node_status)) << endl;
        iree_status_ignore(node_status);
        close();
        return false;
    }
    _dmaHeapNodeAcquired = true;

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

    _open_timer.start();
    LOGD << "Network " << _networkId << " created on open()";
    return true;
}

bool TorqAstraMachina::close() {
    // Clean up network if still active
    if (_networkActive) {
        stopNetwork();
    }
    if (_networkId != kInvalidNetworkId) {
        destroyNetwork();
    }
    TorqHw::freeDeviceBuffer(_xramBuffer);
    _dmabufHandle = kInvalidFd;
    _xramVBase = nullptr;
    if (_dmaHeapNodeAcquired) {
        torq_hw_dma_heap_node_release();
        _dmaHeapNodeAcquired = false;
    }
    if (_torqDevNode != kInvalidFd) {
        ::close(_torqDevNode);
        _torqDevNode = kInvalidFd;
    }
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
    if (_xramAccessActive) {
        return true;
    }

    struct dma_buf_sync sync = {0};
    sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW;
    if (ioctl(_dmabufHandle, DMA_BUF_IOCTL_SYNC, &sync)) {
        cerr << "error in dmabuf start sync ioctl\n";
        return false;
    }
    _xramAccessActive = true;
    return true;
#endif
}

bool TorqAstraMachina::endXramAccess() const {
#if defined (DMABUF_USE_UNCACHED)
    return true;
#else
    if (!_xramAccessActive) {
        return true;
    }
    struct dma_buf_sync sync = {0};
    sync.flags = DMA_BUF_SYNC_END  | DMA_BUF_SYNC_RW;
    if (ioctl(_dmabufHandle, DMA_BUF_IOCTL_SYNC, &sync)) {
        cerr << "error in dmabuf end sync ioctl\n";
        return false;
    }
    _xramAccessActive = false;
    return true;
#endif
}

const void * TorqAstraMachina::startXramReadAccess(uint32_t addr) const {
    // TODO check bounds with size and use DMA_BUF_SYNC_READ for read only access
    if (!startXramAccess()) {
        return nullptr;
    }
    return reinterpret_cast<const void*>(_xramVBase + _alignOffset + (addr - _xramStartAddr));
}

bool TorqAstraMachina::endXramReadAccess() {
    // TODO use DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ for read only access
    return endXramAccess();
}

void * TorqAstraMachina::startXramWriteAccess(uint32_t addr) {
    // TODO check bounds and use DMA_BUF_SYNC_WRITE for write access
    if (!startXramAccess()) {
        return nullptr;
    }
    return reinterpret_cast<void*>(_xramVBase + _alignOffset + (addr - _xramStartAddr));
}

bool TorqAstraMachina::endXramWriteAccess() {
    // TODO use DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE for write access
    return endXramAccess();
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
    // don't *need* a temp copy, defensive move in-case `torq_hw_device_buffer_allocate() fails
    TorqDeviceBuffer xramBuffer{};
    iree_status_t status = torq_hw_device_buffer_allocate(
        TORQ_HW_DEVICE_BUFFER_MODE_DMA_HEAP_UNCACHED, _xramSizeAligned, &xramBuffer
    );
    if (!iree_status_is_ok(status)) {
        cerr << "error allocating buffer from heap " << DMABUF_NODE_UNCACHED
             << ", check kernel config\n";
        iree_status_ignore(status);
        return false;
    }
    _xramBuffer = xramBuffer;
    _dmabufHandle = _xramBuffer.handle;
    _xramVBase = (uint8_t *)_xramBuffer.mapped;
    LOGD << "(xram) dmabuf fd " << _dmabufHandle << ": size " << _xramSize << "\n";
    return true;
}

std::optional<TorqDeviceBuffer> TorqAstraMachina::allocateDeviceBuffer(size_t size) {
    TorqDeviceBuffer buffer{};
    iree_status_t status = torq_hw_device_buffer_allocate(
        TORQ_HW_DEVICE_BUFFER_MODE_DMA_HEAP, size, &buffer);
    if (!iree_status_is_ok(status)) {
        LOGE << "Failed to allocate device buffer: "
             << iree_status_code_string(iree_status_code(status));
        iree_status_ignore(status);
        return std::nullopt;
    }
    return buffer;
}

bool TorqAstraMachina::attachBinding(
    const TorqDeviceBuffer &buffer, uint32_t xramAddr, size_t dataOffset, size_t size
) {
    if (!_networkActive || _networkId == kInvalidNetworkId || buffer.handle == kInvalidFd) {
        return TorqHw::attachBinding(buffer, xramAddr, dataOffset, size);
    }

    struct torq_attach_binding_req req = {};
    req.network_id = _networkId;
    req.dmabuf_fd = buffer.handle;
    req.xram_addr = xramAddr;
    req.data_offset = dataOffset;
    req.binding_size = size;
    return ioctl(_torqDevNode, TORQ_IOCTL_ATTACH_BINDING, &req) == 0;
}

bool TorqAstraMachina::detachBinding(
    const TorqDeviceBuffer &buffer, uint32_t xramAddr, size_t dataOffset, size_t size
) {
    if (!_networkActive || _networkId == kInvalidNetworkId || buffer.handle == kInvalidFd) {
        return TorqHw::detachBinding(buffer, xramAddr, dataOffset, size);
    }

    struct torq_detach_binding_req req = {};
    req.network_id = _networkId;
    req.xram_addr = xramAddr;
    req.data_offset = dataOffset;
    req.binding_size = size;
    return ioctl(_torqDevNode, TORQ_IOCTL_DETACH_BINDING, &req) == 0;
}

bool TorqAstraMachina::createNetwork() {
    torq_create_network_req createReq;
    createReq.xram_start = _xramStartAddr;
    createReq.dmabuf_fd = _dmabufHandle;
    createReq.network_id = kInvalidNetworkId;

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
    if (_networkId == kInvalidNetworkId) {
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
    if (_networkId == kInvalidNetworkId) {
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
    _networkId = kInvalidNetworkId;
    _networkActive = false;
    return true;
}

bool TorqAstraMachina::acquire() {
    if (_networkId == kInvalidNetworkId) {
        cerr << "unable to load, network not setup yet\n";
        return false;
    }

    if (!TorqHw::acquire()) {
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

    if (!TorqHw::release()) {
        return false;
    }

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
