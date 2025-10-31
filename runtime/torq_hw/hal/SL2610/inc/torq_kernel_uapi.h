/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2025 Synaptics Incorporated */

#ifndef __TORQ_KERNEL_UAPI__
#define __TORQ_KERNEL_UAPI__

#ifdef __KERNEL__
#include <linux/ioctl.h>
#else
#include <sys/ioctl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TORQ_IOCTL_MAGIC 'S'

/* IOCTL commands */
#define TORQ_IOCTL_CREATE_NETWORK  _IOWR(TORQ_IOCTL_MAGIC, 1, struct torq_create_network_req)
#define TORQ_IOCTL_START_NETWORK   _IOWR(TORQ_IOCTL_MAGIC, 2, struct torq_start_network_req)
#define TORQ_IOCTL_RUN_NETWORK     _IOWR(TORQ_IOCTL_MAGIC, 3, struct torq_run_network_req)
#define TORQ_IOCTL_WAIT_NETWORK    _IOWR(TORQ_IOCTL_MAGIC, 4, struct torq_wait_network_req)
#define TORQ_IOCTL_STOP_NETWORK    _IOWR(TORQ_IOCTL_MAGIC, 5, struct torq_stop_network_req)
#define TORQ_IOCTL_DESTROY_NETWORK _IOWR(TORQ_IOCTL_MAGIC, 6, struct torq_destroy_network_req)
#define TORQ_IOCTL_WRITE_LRAM      _IOWR(TORQ_IOCTL_MAGIC, 7, struct torq_write_lram_req)
#define TORQ_IOCTL_READ_LRAM       _IOWR(TORQ_IOCTL_MAGIC, 8, struct torq_read_lram_req)

/* 32bit compact IOCTL commands */
#define TORQ_IOCTL_WRITE_LRAM_32   _IOWR(TORQ_IOCTL_MAGIC, 7, struct torq_write_lram_req_32compat)
#define TORQ_IOCTL_READ_LRAM_32    _IOWR(TORQ_IOCTL_MAGIC, 8, struct torq_read_lram_req_32compat)

/* IOCTL Data Structures */
struct torq_create_network_req {
    unsigned int xram_start;        /* XRAM start address */
    int dmabuf_fd;                  /* DMA buffer file descriptor */
    unsigned int network_id;        /* Output: allocated network ID */
};

struct torq_start_network_req {
    unsigned int network_id;        /* Network ID to start */
};

struct torq_run_network_req {
    unsigned int network_id;        /* Network ID */
    unsigned int code_entry;        /* Job code entry address */
};

struct torq_wait_network_req {
    unsigned int network_id;        /* Network ID */
    unsigned int wait_bits;         /* Bits to wait for completion */
};

struct torq_stop_network_req {
    unsigned int network_id;        /* Network ID to stop and cleanup */
};

struct torq_destroy_network_req {
    unsigned int network_id;        /* Network ID to destroy */
};

struct torq_write_lram_req {
    unsigned int network_id;       /* Network ID */
    unsigned int addr;             /* LRAM address offset */
    unsigned int size;             /* Data size */
    void *data;                    /* Data pointer */
};

struct torq_read_lram_req {
    unsigned int network_id;       /* Network ID */
    unsigned int addr;             /* LRAM address offset */
    unsigned int size;             /* Data size */
    void *data;                    /* Data pointer */
};

struct torq_write_lram_req_32compat {
    unsigned int network_id;       /* Network ID */
    unsigned int addr;             /* LRAM address offset */
    unsigned int size;             /* Data size */
    unsigned int data;             /* 32bit data pointer value */
};

struct torq_read_lram_req_32compat {
    unsigned int network_id;       /* Network ID */
    unsigned int addr;             /* LRAM address offset */
    unsigned int size;             /* Data size */
    unsigned int data;             /* 32bit data pointer value */
};

#ifdef __cplusplus
}
#endif

#endif
