// SPDX-License-Identifier: GPL-2.0
// Copyright 2025 Synaptics Incorporated

#include "torq_core_linux.h"
#include "torq_kernel_log.h"
#include "torq_reg_define.h"

#define REQUIRE_DEVICE_LOCK(cmd)  ((cmd == TORQ_IOCTL_START_NETWORK) || \
               (cmd == TORQ_IOCTL_RUN_NETWORK) || \
               (cmd == TORQ_IOCTL_WAIT_NETWORK) || \
               (cmd == TORQ_IOCTL_STOP_NETWORK) || \
               (cmd == TORQ_IOCTL_DESTROY_NETWORK))

#define SYNA_NPU_DEV_NAME "torq"
#define IOVA_ALIGN 4096
#define NPU_RESET_INTERVAL 10

static int torq_power_on(struct torq_module *torq_dev)
{
    int ret;

    LOG_ENTER();
    ret = clk_prepare_enable(torq_dev->core_clk);
    if (ret) {
       KLOGE("error enabling clock");
       return ret;
    }
    return ret;
}

static int torq_power_off(struct torq_module *torq_dev)
{
    LOG_ENTER();
    clk_disable_unprepare(torq_dev->core_clk);
    return 0;
}

static int torq_detach_network_domain(struct torq_module *torq_dev, struct torq_network *net)
{
    int ret;

    /* Verify this network is the one currently attached */
    if (torq_dev->active_network != net) {
        KLOGE("Network %d not currently active (active: %d)", net->network_id,
              torq_dev->active_network ? torq_dev->active_network->network_id : 0);
        return -EPERM;
    }

    /* Detach network domain from hardware and attach dma domain back */
    iommu_detach_device(net->domain, torq_dev->iommu_device);
    ret = iommu_attach_device(torq_dev->default_domain, torq_dev->iommu_device);
    if (ret < 0) {
        KLOGE("Critical: Failed to revert to default domain for network %d: %d", net->network_id, ret);
        /* Continue cleanup */
    }

    torq_dev->active_network = NULL;
    KLOGI("hardware released from network %d usage", net->network_id);

    return ret;
}

static void torq_cleanup_network_resources(struct torq_module *torq_dev, struct torq_network *net)
{
    struct torq_lram_segment *segment, *tmp;
    list_for_each_entry_safe(segment, tmp, &net->lram_segments, list) {
        list_del(&segment->list);
        kfree(segment->data);
        kfree(segment);
    }

    if (torq_dev->active_network == net) {
        torq_detach_network_domain(torq_dev, net);
    }

    if (net->domain) {
        if (net->xram_start && net->size) {
            iommu_unmap(net->domain, net->xram_start, net->size);
            net->iova_mapped = false;
            net->xram_start = 0;
            net->size = 0;
        }
        iommu_domain_free(net->domain);
        net->domain = NULL;
    }

    if (net->attachment && net->sgt) {
        dma_buf_unmap_attachment(net->attachment, net->sgt, DMA_BIDIRECTIONAL);
        dma_buf_detach(net->dmabuf, net->attachment);
        net->attachment = NULL;
        net->sgt = NULL;
    }

    if (net->dmabuf) {
        dma_buf_put(net->dmabuf);
        net->dmabuf = NULL;
    }
}

static struct torq_network *torq_find_network_in_instance(struct torq_file_inst *inst, uint32_t network_id)
{
    return xa_load(&inst->networks_xa, network_id);
}

static struct torq_lram_segment *torq_find_lram_segment(struct torq_network *net, uint32_t addr)
{
    struct torq_lram_segment *segment;

    list_for_each_entry(segment, &net->lram_segments, list) {
        if (segment->addr == addr) {
            return segment;
        }
    }
    return NULL;
}

static void torq_commit_lram_to_hw(void __iomem *lram_base, uint32_t addr,
                                   const uint8_t *data, size_t size)
{
    int index = 0;
    size_t bytes_remaining = size;
    size_t memcpy_bytes;
    size_t bytes_for_read_modify;
    uint32_t lram_update;
    uint32_t aligned_addr, offset, mask;

    /* Handle unaligned start */
    if (addr & 0x3) {
        aligned_addr = addr & ~0x03;
        offset = addr & 0x3;
        bytes_for_read_modify = min(bytes_remaining, (size_t)(4 - offset));
        mask = ((1U << (bytes_for_read_modify * 8)) - 1) << (offset * 8);

        lram_update = readl(lram_base + aligned_addr);
        lram_update &= ~mask;

        for (index = 0; index < bytes_for_read_modify; index++) {
            lram_update |= ((uint32_t)data[index]) << ((offset + index) * 8);
        }

        writel(lram_update, lram_base + aligned_addr);
        bytes_remaining -= bytes_for_read_modify;
        addr += bytes_for_read_modify;
        data += bytes_for_read_modify;
    }

    if (bytes_remaining >= 4) {
        memcpy_bytes = bytes_remaining & ~0x03;
        memcpy_toio(lram_base + addr, data, memcpy_bytes);
        bytes_remaining -= memcpy_bytes;
        addr += memcpy_bytes;
        data += memcpy_bytes;
    }

    /* Handle unaligned end */
    if (bytes_remaining) {
        lram_update = readl(lram_base + addr);
        mask = (1U << (bytes_remaining * 8)) - 1;
        lram_update &= ~mask;

        for (index = 0; index < bytes_remaining; index++) {
            lram_update |= ((uint32_t)data[index]) << (index * 8);
        }

        writel(lram_update, lram_base + addr);
    }
}

static void torq_commit_network_lram(struct torq_module *torq_dev, struct torq_network *net)
{
    struct torq_lram_segment *segment;
    int segment_count = 0;

    if (!list_empty(&net->lram_segments)) {
        KLOGD("Loading LRAM segments to hardware for network %d", net->network_id);
        list_for_each_entry(segment, &net->lram_segments, list) {
            KLOGI("Loading segment %d: %zu bytes at offset 0x%x",
                  segment_count, segment->size, segment->addr);
            /* Copy segment data to hardware at the specified address */
            torq_commit_lram_to_hw(torq_dev->reg_map, segment->addr, segment->data, segment->size);
            segment_count++;
        }

        KLOGD("Successfully loaded %d LRAM segments for network %d",
              segment_count, net->network_id);
    }
}

static int torq_add_lram_segment(struct torq_network *net, unsigned int addr,
                                 size_t size, const void *data)
{
    struct torq_lram_segment *segment;

    segment = kzalloc(sizeof(*segment), GFP_KERNEL);
    if (!segment) {
        KLOGE("Failed to allocate LRAM segment");
        return -ENOMEM;
    }

    segment->data = kmalloc(size, GFP_KERNEL);
    if (!segment->data) {
        KLOGE("Failed to allocate LRAM segment data");
        kfree(segment);
        return -ENOMEM;
    }

    segment->addr = addr;
    segment->size = size;
    memcpy(segment->data, data, size);
    list_add_tail(&segment->list, &net->lram_segments);

    KLOGD("Created new LRAM segment at addr 0x%x, size %zu in network:%d", addr, size, net->network_id);
    return 0;
}

static int torq_create_network(struct torq_file_inst *inst, struct torq_create_network_req *req)
{
    struct torq_module *torq_dev = inst->torq_device;
    struct torq_network *net;
    struct dma_buf *dmabuf;
    struct iommu_domain *domain;
    uint32_t network_id;
    int ret;

    if (!inst || !req) {
        KLOGE("Invalid parameters: inst=%p, req=%p", inst, req);
        return -EINVAL;
    }

    if (req->dmabuf_fd < 0) {
        KLOGE("Invalid dmabuf_fd: %d", req->dmabuf_fd);
        return -EINVAL;
    }

    if (req->xram_start & (IOVA_ALIGN - 1)) {
        KLOGE("IOMMU address 0x%x not aligned to %d bytes", req->xram_start, IOVA_ALIGN);
        return -EINVAL;
    }

    KLOGD("(%d) Creating network with dmabuf_fd=%d, xram_start=0x%x",
          inst->pid, req->dmabuf_fd, req->xram_start);

    dmabuf = dma_buf_get(req->dmabuf_fd);
    if (IS_ERR(dmabuf)) {
        KLOGE("Failed to get DMA buffer from fd %d: %ld", req->dmabuf_fd, PTR_ERR(dmabuf));
        return PTR_ERR(dmabuf);
    }

    /* Create new unmanged IOMMU domain to manage networks address space */
    domain = iommu_domain_alloc(torq_dev->pdev->dev.bus);
    if (!domain) {
        KLOGE("Failed to allocate IOMMU domain for network");
        dma_buf_put(dmabuf);
        return -ENOMEM;
    }

    net = kzalloc(sizeof(*net), GFP_KERNEL);
    if (!net) {
        KLOGE("Failed to allocate network structure");
        iommu_domain_free(domain);
        dma_buf_put(dmabuf);
        return -ENOMEM;
    }

    net->dmabuf = dmabuf;
    net->dmabuf_fd = req->dmabuf_fd;
    net->domain = domain;
    net->xram_start = req->xram_start;
    net->size = dmabuf->size;
    INIT_LIST_HEAD(&net->lram_segments);

    /* store instance in xarry and use allocated id to refer network*/
    ret = xa_alloc(&inst->networks_xa, &network_id, net, XA_LIMIT(1, UINT_MAX), GFP_KERNEL);
    if (ret) {
        KLOGE("Failed to allocate network ID in XArray: %d", ret);
        iommu_domain_free(domain);
        dma_buf_put(dmabuf);
        kfree(net);
        return ret;
    }

    net->network_id = network_id;
    req->network_id = net->network_id;

    KLOGI("Created network Id %d for pid %d", net->network_id, inst->pid);
    return 0;
}

static int torq_start_network(struct torq_file_inst *inst, struct torq_start_network_req *req)
{
    struct torq_module *torq_dev = inst->torq_device;
    struct torq_network *net;
    struct dma_buf_attachment *attachment;
    struct sg_table *sgt;
    int ret;

    if (!inst || !req) {
        KLOGE("Invalid parameters: inst=%p, req=%p", inst, req);
        return -EINVAL;
    }

    net = torq_find_network_in_instance(inst, req->network_id);
    if (!net) {
        KLOGE("Network %d not found in instance", req->network_id);
        return -ENOENT;
    }

    KLOGD("Starting network:%d (pid:%d)",req->network_id, inst->pid);

    if (torq_dev->active_network) {
        KLOGE("Hardware busy with network %d",
              torq_dev->active_network->network_id);
        return -EBUSY; /* application can retry on EBUSY return */
    }

    if (!net->attachment) {
        KLOGD("mapping network (%d:%d) xram space to device dma domain", inst->pid, req->network_id);
        attachment = dma_buf_attach(net->dmabuf, torq_dev->iommu_device);
        if (IS_ERR(attachment)) {
            KLOGE("Failed to attach DMA buffer: %ld", PTR_ERR(attachment));
            return PTR_ERR(attachment);
        }

        /* Map DMA buffer to get sgt which is later mapped to new domain */
        sgt = dma_buf_map_attachment(attachment, DMA_BIDIRECTIONAL);
        if (IS_ERR(sgt)) {
            KLOGE("Failed to map DMA buffer attachment: %ld", PTR_ERR(sgt));
            dma_buf_detach(net->dmabuf, attachment);
            return PTR_ERR(sgt);
        }

        net->attachment = attachment;
        net->sgt = sgt;
    }

    KLOGI("attach iova domain from network %d:%d to device", inst->pid, req->network_id);
    /* Attach network's domain to hardware device */
    ret = iommu_attach_device(net->domain, torq_dev->iommu_device);
    if (ret < 0) {
        KLOGE("Failed to attach network domain to device: %d", ret);
        return ret;
    }

    if (!net->iova_mapped) {
        /* Map DMA buffer to the network's IOMMU domain */
        KLOGD("Mapping DMA buffer to network's domain at 0x%x", net->network_id, net->xram_start);
        ret = iommu_map_sg(net->domain, net->xram_start, net->sgt->sgl,
                           net->sgt->orig_nents, IOMMU_READ | IOMMU_WRITE, GFP_KERNEL);
        if (ret < 0) {
            KLOGE("Failed to map SG table to network IOMMU domain: %d", ret);
            iommu_detach_device(net->domain, torq_dev->iommu_device);
            iommu_attach_device(torq_dev->default_domain, torq_dev->iommu_device);
            return ret;
        }
        net->iova_mapped = true;
    }

    /* Mark network as attached and set as active */
    torq_dev->active_network = net;

    /* write the network specific LRAM space, LRAM updates for network expected to be serialized */
    torq_commit_network_lram(torq_dev, net);

    /* reset the module before starting new network */
    reset_control_assert(torq_dev->core_rst);
    udelay(NPU_RESET_INTERVAL);
    reset_control_deassert(torq_dev->core_rst);

    return 0;
}

static int torq_run_network(struct torq_file_inst *inst, struct torq_run_network_req *req)
{
    struct torq_module *torq_dev = inst->torq_device;
    struct torq_network *net;

    if (!inst || !req) {
        KLOGE("Invalid parameters: inst=%p, req=%p", inst, req);
        return -EINVAL;
    }

    net = torq_find_network_in_instance(inst, req->network_id);
    if (!net) {
        KLOGE("Network %d not found in instance", req->network_id);
        return -ENOENT;
    }

    if (torq_dev->active_network != net) {
        KLOGE("Network %d not attached to hardware", req->network_id);
        return -ENODEV;
    }

    KLOGD("Running job on network %d with code entry 0x%x", req->network_id, req->code_entry);

    writel(RF_LSH(NSS, CFG_LINK_EN, 1) | RF_BMSK_LSH(NSS, CFG_DESC, req->code_entry),
           torq_dev->reg_map + RA_(NSS,CFG));

    writel(RF_LSH(NSS, CTRL_IEN_NSS, 1), torq_dev->reg_map + RA_(NSS,CTRL));

    writel(RF_LSH(CSS, IEN_HST_NSS, 1), torq_dev->reg_map + RA_(CSS,IEN_HST));

    wmb();

    writel(RF_LSH(NSS, START_NSS, 1), torq_dev->reg_map + RA_(NSS,START));

    wmb();

    return 0;
}

static int torq_wait_network(struct torq_file_inst *inst, struct torq_wait_network_req *req)
{
    struct torq_module *torq_dev = inst->torq_device;
    struct torq_network *net;
    int poll_count = 0;
    int timeout_ms = 1000;
    int poll_interval_us = 100;
    int max_polls = (timeout_ms * 1000) / poll_interval_us;
    uint32_t reg_val;

    if (!inst || !req) {
        KLOGE("Invalid parameters: inst=%p, req=%p", inst, req);
        return -EINVAL;
    }

    net = torq_find_network_in_instance(inst, req->network_id);
    if (!net) {
        KLOGE("Network %d not found in instance", req->network_id);
        return -ENOENT;
    }

    if (torq_dev->active_network != net) {
        KLOGE("Network %d not attached to hardware", req->network_id);
        return -ENODEV;
    }

    while (poll_count < max_polls) {
        reg_val = readl(torq_dev->reg_map + RA_(NSS,STATUS));

        bool job_complete = true;

        if (req->wait_bits & (1 << TORQ_IOCTL_WAIT_BITMASK_NSS)) {
            job_complete = job_complete && (reg_val & (1 << REG_FIELD_POS__SYNPU_NSS_REGS_STATUS_NSS));
        }
        if (req->wait_bits & (1 << TORQ_IOCTL_WAIT_BITMASK_DMA_IN)) {
            job_complete = job_complete && (reg_val & (1 << REG_FIELD_POS__SYNPU_NSS_REGS_STATUS_XR));
        }
        if (req->wait_bits & (1 << TORQ_IOCTL_WAIT_BITMASK_DMA_OUT)) {
            job_complete = job_complete && (reg_val & (1 << REG_FIELD_POS__SYNPU_NSS_REGS_STATUS_XW));
        }
        if (req->wait_bits & (1 << TORQ_IOCTL_WAIT_BITMASK_SLC_0)) {
            job_complete = job_complete && (reg_val & (1 << REG_FIELD_POS__SYNPU_NSS_REGS_STATUS_SLC0));
        }
        if (req->wait_bits & (1 << TORQ_IOCTL_WAIT_BITMASK_SLC_1)) {
            job_complete = job_complete && (reg_val & (1 << REG_FIELD_POS__SYNPU_NSS_REGS_STATUS_SLC1));
        }

        if (job_complete) {
            KLOGD("Job completed on network %d, NSS_STATUS: 0x%x", req->network_id, reg_val);
            break;
        }

        udelay(poll_interval_us);
        poll_count++;
    }

    if (poll_count >= max_polls) {
        KLOGE("Job timeout on network %d after %d ms, NSS_STATUS: 0x%x", req->network_id, timeout_ms, reg_val);
        return -ETIMEDOUT;
    }

    /* Clear status */
    writel(1, torq_dev->reg_map + RA_(NSS,STATUS));
    return 0;
}

static int torq_stop_network(struct torq_file_inst *inst, struct torq_stop_network_req *req)
{
    struct torq_module *torq_dev = inst->torq_device;
    struct torq_network *net;
    int ret;

    if (!inst || !req) {
        KLOGE("Invalid parameters: inst=%p, req=%p", inst, req);
        return -EINVAL;
    }

    KLOGD("Stopping network %d", req->network_id);

    net = torq_find_network_in_instance(inst, req->network_id);
    if (!net) {
        KLOGE("Network %d not found in instance", req->network_id);
        return -ENOENT;
    }

    ret = torq_detach_network_domain(torq_dev, net);
    return ret;
}

static int torq_destroy_network(struct torq_file_inst *inst, struct torq_destroy_network_req *req)
{
    struct torq_module *torq_dev = inst->torq_device;
    struct torq_network *net;

    if (!inst || !req) {
        KLOGE("Invalid parameters: inst=%p, req=%p", inst, req);
        return -EINVAL;
    }

    KLOGD("Destroying network %d", req->network_id);

    net = torq_find_network_in_instance(inst, req->network_id);
    if (!net) {
        KLOGE("Network %d not found in instance", req->network_id);
        return -ENOENT;
    }

    if (torq_dev->active_network == net) {
        KLOGE("Cannot destroy network %d: still attached to hardware", req->network_id);
        return -EBUSY;
    }

    torq_cleanup_network_resources(torq_dev, net);

    xa_erase(&inst->networks_xa, req->network_id);
    kfree(net);

    KLOGI("Network %d destroyed successfully", req->network_id);
    return 0;
}


static int torq_write_lram(struct torq_file_inst *inst, struct torq_write_lram_req *req)
{
    struct torq_module *torq_dev = inst->torq_device;
    struct torq_network *net;
    void *lram_data;
    int ret;

    if (!req->data || req->size == 0) {
        KLOGE("Invalid LRAM write parameters");
        return -EINVAL;
    }

    lram_data = kmalloc(req->size, GFP_KERNEL);
    if (!lram_data) {
        KLOGE("Failed to allocate kernel buffer for LRAM write");
        return -ENOMEM;
    }

    if (copy_from_user(lram_data, req->data, req->size)) {
        KLOGE("Failed to copy LRAM data from user space");
        kfree(lram_data);
        return -EFAULT;
    }

    net = xa_load(&inst->networks_xa, req->network_id);
    if (!net) {
        KLOGE("Network %d not found for LRAM write", req->network_id);
        kfree(lram_data);
        return -ENOENT;
    }

    ret = torq_add_lram_segment(net, req->addr, req->size, lram_data);

    /* commit to HW if the network is already loaded */
    if (torq_dev->active_network == net) {
        mutex_lock(&torq_dev->device_lock);
        torq_commit_lram_to_hw(torq_dev->reg_map, req->addr, lram_data, req->size);
        mutex_unlock(&torq_dev->device_lock);
    }

    kfree(lram_data);
    return ret;
}

static int torq_read_lram(struct torq_file_inst *inst, struct torq_read_lram_req *req)
{
    struct torq_network *net;
    struct torq_lram_segment *segment;
    size_t copy_size;

    if (!req->data || req->size == 0) {
        KLOGE("Invalid LRAM read parameters");
        return -EINVAL;
    }

    net = xa_load(&inst->networks_xa, req->network_id);
    if (!net) {
        KLOGE("Network %d not found for LRAM read", req->network_id);
        return -ENOENT;
    }

    segment = torq_find_lram_segment(net, req->addr);
    if (!segment) {
        KLOGE("LRAM segment at addr 0x%x not found", req->addr);
        return -ENOENT;
    }

    copy_size = min(req->size, segment->size);
    if (copy_to_user(req->data, segment->data, copy_size)) {
        KLOGE("Failed to copy LRAM data to user space");
        return -EFAULT;
    }

    return 0;
}

static long torq_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    unsigned int dir;
    union torq_ioctl_arg data;
    struct torq_file_inst *inst = filp->private_data;
    int ret = 0;

    if (inst == NULL) {
        KLOGE("error getting torq_file_inst from file data");
        return -EFAULT;
    }

    dir = _IOC_DIR(cmd);
    if (_IOC_SIZE(cmd) > sizeof(data))
        return -EINVAL;

    if (!(dir & _IOC_WRITE))
        memset(&data, 0, sizeof(data));
    else if (copy_from_user(&data, (void __user *)arg, _IOC_SIZE(cmd)))
        return -EFAULT;

    mutex_lock(&inst->inst_mutex);
    if (REQUIRE_DEVICE_LOCK(cmd)) {
        mutex_lock(&inst->torq_device->device_lock);
    }

    switch (cmd) {

        case TORQ_IOCTL_CREATE_NETWORK:
            KLOGD("torq_ioctl create network:(pid:%d)", inst->pid);
            ret = torq_create_network(inst, &data.create_network_request);
            if (ret)
                break;

            if (copy_to_user((void __user *)arg, &data, _IOC_SIZE(cmd))) {
                struct torq_network *net = xa_erase(&inst->networks_xa, data.create_network_request.network_id);
                if (net) {
                    iommu_domain_free(net->domain);
                    dma_buf_put(net->dmabuf);
                    kfree(net);
                }
                KLOGE("error copying network details back to user");
                ret = -EFAULT;
            }
        break;

        case TORQ_IOCTL_START_NETWORK:
            ret = torq_start_network(inst, &data.start_network_request);
        break;

        case TORQ_IOCTL_RUN_NETWORK:
            KLOGD("torq_ioctl run network:(pid:%d)", inst->pid);
            ret = torq_run_network(inst, &data.run_network_request);
        break;

        case TORQ_IOCTL_WAIT_NETWORK:
            KLOGD("torq_ioctl wait network:(pid:%d)", inst->pid);
            ret = torq_wait_network(inst, &data.wait_network_request);
        break;

        case TORQ_IOCTL_STOP_NETWORK:
            KLOGD("torq_ioctl stop network:(pid:%d)", inst->pid);
            ret = torq_stop_network(inst, &data.stop_network_request);
        break;

        case TORQ_IOCTL_DESTROY_NETWORK:
            KLOGD("torq_ioctl destroy network:(pid:%d)", inst->pid);
            ret = torq_destroy_network(inst, &data.destroy_network_request);
        break;

        case TORQ_IOCTL_WRITE_LRAM:
            KLOGD("torq_ioctl write lram:%d 0x%x:0x%x", inst->pid, TORQ_IOCTL_WRITE_LRAM, TORQ_IOCTL_WRITE_LRAM_32);
            ret = torq_write_lram(inst, &data.lram_write_request);
        break;

        case TORQ_IOCTL_READ_LRAM:
            KLOGD("torq_ioctl read lram:%d", inst->pid);
            ret = torq_read_lram(inst, &data.lram_read_request);
        break;

        default:
            KLOGE("unknown ioctl cmd::0x%x", cmd);
            ret = -EINVAL;
    }

    if (REQUIRE_DEVICE_LOCK(cmd)) {
        mutex_unlock(&inst->torq_device->device_lock);
    }
    mutex_unlock(&inst->inst_mutex);
    return ret;
}

static long torq_compat_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct torq_file_inst *inst = filp->private_data;
    struct torq_write_lram_req lram_write_64;
    struct torq_write_lram_req_32compat lram_write_32;
    struct torq_read_lram_req lram_read_64;
    struct torq_read_lram_req_32compat lram_read_32;
    int ret = 0;

    switch (cmd) {
        case TORQ_IOCTL_WRITE_LRAM_32:
            if (copy_from_user(&lram_write_32, (void __user *)arg, _IOC_SIZE(cmd))) {
                KLOGE("error copying lram data from user\n");
                return -EFAULT;
            }
            lram_write_64.network_id = lram_write_32.network_id;
            lram_write_64.addr = lram_write_32.addr;
            lram_write_64.size = lram_write_32.size;
            lram_write_64.data = compat_ptr(lram_write_32.data);
            mutex_lock(&inst->inst_mutex);
            ret = torq_write_lram(inst, &lram_write_64);
            mutex_unlock(&inst->inst_mutex);
        break;

        case TORQ_IOCTL_READ_LRAM_32:
            if (copy_from_user(&lram_read_32, (void __user *)arg, _IOC_SIZE(cmd))) {
                KLOGE("error copying lram data from user\n");
                return -EFAULT;
            }
            lram_read_64.network_id = lram_read_32.network_id;
            lram_read_64.addr = lram_read_32.addr;
            lram_read_64.size = lram_read_32.size;
            lram_read_64.data = compat_ptr(lram_read_32.data);
            mutex_lock(&inst->inst_mutex);
            ret = torq_read_lram(inst, &lram_read_64);
            mutex_unlock(&inst->inst_mutex);
        break;

        default:
            ret = torq_ioctl(filp, cmd, arg);
    }

    return ret;
}

static int torq_open(struct inode *inode, struct file *filp)
{
    struct torq_module *torq_dev;
    struct torq_file_inst *inst = NULL;

    LOG_ENTER();

    torq_dev = container_of(filp->private_data, struct torq_module, misc_dev);

    inst = kzalloc(sizeof(struct torq_file_inst), GFP_KERNEL);
    if (!inst) {
        KLOGE("alloc torq_file_inst failed");
        return -ENOMEM;
    }

    inst->torq_device = torq_dev;
    inst->pid = current->pid;

    xa_init_flags(&inst->networks_xa, XA_FLAGS_TRACK_FREE);
    mutex_init(&inst->inst_mutex);

    mutex_lock(&torq_dev->files_mutex);

    /* Power on when first file instance opens */
    if (list_empty(&torq_dev->files)) {
        torq_power_on(torq_dev);
    }

    list_add(&inst->list, &torq_dev->files);
    mutex_unlock(&torq_dev->files_mutex);

    filp->private_data = inst;
    return 0;
}

static int torq_release(struct inode *inode, struct file *file)
{
    struct torq_file_inst *inst = file->private_data;
    struct torq_module *torq_dev = inst->torq_device;

    LOG_ENTER();

    /* Clean up all networks owned by this file instance */
    mutex_lock(&inst->inst_mutex);
    mutex_lock(&torq_dev->device_lock);

    /* Iterate through XArray and cleanup networks */
    struct torq_network *net;
    unsigned long index;
    xa_for_each(&inst->networks_xa, index, net) {
        if (torq_dev->active_network == net) {
            /* Force stop active networks */
            KLOGI("instance closed without releasing job resources.. force close");
            torq_detach_network_domain(inst->torq_device, net);
        }
        torq_cleanup_network_resources(inst->torq_device, net);
        xa_erase(&inst->networks_xa, net->network_id);
        kfree(net);
    }
    mutex_unlock(&torq_dev->device_lock);
    mutex_unlock(&inst->inst_mutex);

    mutex_lock(&inst->torq_device->files_mutex);
    list_del(&inst->list);

    /* Power off when last file instance closes */
    if (list_empty(&inst->torq_device->files)) {
        torq_power_off(inst->torq_device);
    }

    mutex_unlock(&inst->torq_device->files_mutex);

    kfree(inst);
    return 0;
}

static const struct file_operations torq_fops = {
    .owner = THIS_MODULE,
    .open = torq_open,
    .release = torq_release,
    .unlocked_ioctl = torq_ioctl,
    .compat_ioctl = torq_compat_ioctl,
};

static void torq_remove(struct platform_device *pdev)
{
    struct torq_module *torq_dev = platform_get_drvdata(pdev);
    LOG_ENTER();

    if (torq_dev->default_domain) {
        iommu_domain_free(torq_dev->default_domain);
    }

    if (torq_dev->misc_registered) {
        misc_deregister(&torq_dev->misc_dev);
    }

    platform_set_drvdata(pdev, NULL);
}

static int torq_probe(struct platform_device *pdev)
{
    struct torq_module *torq_dev;
    struct resource *res;
    /* temp: override axi prot for accessing non-secure MMU space */
    void __iomem *npu_axi_prot_map;

    LOG_ENTER();
    torq_dev = devm_kzalloc(&pdev->dev, sizeof(*torq_dev), GFP_KERNEL);
    if (!torq_dev)
        return -ENOMEM;

    /* map npu register space */
    res = platform_get_resource_byname(pdev, IORESOURCE_MEM, "npu_region");
    if (!res)
        return -ENODEV;

    torq_dev->reg_map = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(torq_dev->reg_map)) {
        KLOGE("error in remapping register space");
        return PTR_ERR(torq_dev->reg_map);
    }

    torq_dev->reg_size = resource_size(res);
    torq_dev->reg_base = res->start;

    /* temp: map npu axi prot control to override to use non-secure space in iommu */
    res = platform_get_resource_byname(pdev, IORESOURCE_MEM, "axi_prot_override");
    if (!res) {
        KLOGE("axi prot override not present");
        return -ENODEV;
    }
    npu_axi_prot_map = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(npu_axi_prot_map)) {
        KLOGE("error in mapping axiprot space");
        return PTR_ERR(npu_axi_prot_map);
    }
    writel(0xf, npu_axi_prot_map); //override npu axi txn to non-secure

    torq_dev->core_clk = devm_clk_get(&pdev->dev, "core");
    if (IS_ERR(torq_dev->core_clk)) {
        KLOGE("devm_clk_get failed in probe %d", PTR_ERR(torq_dev->core_clk));
        return PTR_ERR(torq_dev->core_clk);
    }

    torq_dev->core_rst = devm_reset_control_get(&pdev->dev, "core_rst");
    if (IS_ERR(torq_dev->core_rst)) {
        KLOGE("devm_reset_control_get failed in probe %d", PTR_ERR(torq_dev->core_rst));
        return PTR_ERR(torq_dev->core_rst);
    }

    mutex_init(&torq_dev->files_mutex);
    mutex_init(&torq_dev->device_lock);
    INIT_LIST_HEAD(&torq_dev->files);

    torq_dev->pdev = pdev;
    torq_dev->iommu_device = &torq_dev->pdev->dev;

    /* get default dma domain of connected smmu */
    torq_dev->default_domain = iommu_get_domain_for_dev(torq_dev->iommu_device);
    if (!torq_dev->default_domain) {
        KLOGE("No default IOMMU domain found, check iommu config in dts");
        return -ENODEV;
    }

    dma_set_max_seg_size(&torq_dev->pdev->dev, UINT_MAX);
    platform_set_drvdata(pdev, torq_dev);

    torq_dev->misc_dev.name = SYNA_NPU_DEV_NAME;
    torq_dev->misc_dev.groups = NULL;
    torq_dev->misc_dev.parent = &pdev->dev;
    torq_dev->misc_dev.fops = &torq_fops;
    torq_dev->misc_dev.minor = MISC_DYNAMIC_MINOR;

    if (misc_register(&torq_dev->misc_dev) < 0) {
        KLOGE("cannot register character device");
        torq_remove(pdev);
        return -1;
    }

    torq_dev->misc_registered = true;
    KLOGI("NPU driver loaded");
    return 0;
}

static const struct of_device_id torq_of_match[] = {
    { .compatible = "syna,npu", },
    {},
};

MODULE_DEVICE_TABLE(of, torq_of_match);

static struct platform_driver npu_driver = {
    .probe = torq_probe,
    .remove = torq_remove,
    .driver = {
        .name = SYNA_NPU_DEV_NAME,
        .of_match_table = torq_of_match,
        .owner = THIS_MODULE,
    },
};

module_platform_driver(npu_driver);
MODULE_IMPORT_NS(DMA_BUF);
MODULE_LICENSE("GPL");
