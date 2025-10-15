/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2025 Synaptics Incorporated */

#ifndef __TORQ_CORE_LINUX__
#define __TORQ_CORE_LINUX__

#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <linux/miscdevice.h>
#include <linux/xarray.h>
#include <linux/iommu.h>
#include <linux/dma-buf.h>
#include <linux/dma-mapping.h>
#include <linux/dma-map-ops.h>
#include <linux/delay.h>
#include <linux/clk.h>
#include <linux/reset.h>
#include <linux/compat.h>
#include <linux/list.h>
#include <linux/mutex.h>

#include "torq_kernel_uapi.h"

#define TORQ_IOCTL_WAIT_BITMASK_NSS 0
#define TORQ_IOCTL_WAIT_BITMASK_DMA_IN 1
#define TORQ_IOCTL_WAIT_BITMASK_DMA_OUT 2
#define TORQ_IOCTL_WAIT_BITMASK_SLC_0 3
#define TORQ_IOCTL_WAIT_BITMASK_SLC_1 4

struct torq_lram_segment {
    struct list_head list;      /* List node */
    unsigned int addr;          /* Start offset for segment */
    unsigned int size;          /* Size of segment */
    void *data;                 /* Segment data */
};

union torq_ioctl_arg {
    struct torq_create_network_req create_network_request;
    struct torq_start_network_req start_network_request;
    struct torq_run_network_req run_network_request;
    struct torq_wait_network_req wait_network_request;
    struct torq_stop_network_req stop_network_request;
    struct torq_destroy_network_req destroy_network_request;
    struct torq_write_lram_req lram_write_request;
    struct torq_read_lram_req lram_read_request;
};

struct torq_module {
    /* this is the device that is registered in the DTS */
    struct platform_device *pdev;

    struct list_head files;
    struct mutex files_mutex;
    struct mutex device_lock; /* Mutex for device level access from instances */

    /* device register base and size */
    void __iomem *reg_map;
    resource_size_t reg_base;
    resource_size_t reg_size;

    /* clock for npu configured from dts */
    struct clk *core_clk;

    /* reset for npu configured from dts */
    struct reset_control *core_rst;

    /* iommu device and domains */
    struct device *iommu_device;
    struct iommu_domain *default_domain; /* Default domain to revert to */

    /* misc device registration */
    struct miscdevice misc_dev;
    bool misc_registered;

    struct torq_network *active_network; /* Currently attached network */
};

struct torq_network {
    unsigned int network_id;
    unsigned int xram_start;
    size_t size;

    /* dmabuf fd holding the xram space from user */
    int dmabuf_fd;
    /* dmabuf mapping to default domain of device */
    struct dma_buf *dmabuf;
    struct dma_buf_attachment *attachment;
    struct sg_table *sgt;

    /* network specific unmanaged domain which hold its IOVA space */
    struct iommu_domain *domain;
    /* lram segments for the network, committed to shared LRAM when network loaded*/
    struct list_head lram_segments;
    /* check if xram region is mapped in networks domain */
    bool iova_mapped;
};

struct torq_file_inst {
    struct torq_module *torq_device;
    struct list_head list;
    struct xarray networks_xa;
    struct mutex inst_mutex;
    pid_t pid;
};

#endif
