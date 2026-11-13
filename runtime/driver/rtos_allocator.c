// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <cache.h>
#include <zephyr/init.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/util.h>

#include "iree/base/api.h"

#define TORQ_HOST_ALIGNMENT 64

typedef union torq_host_allocation_header_t {
    size_t size;
    uint8_t padding[TORQ_HOST_ALIGNMENT];
} torq_host_allocation_header_t;

static uint8_t runtime_heap_storage[CONFIG_TORQ_HEAP_SIZE]
    Z_GENERIC_SECTION(APSRAM_TORQ_HEAP) __aligned(TORQ_HOST_ALIGNMENT);
static struct k_heap runtime_heap;

static int initialize_runtime_heap(void) {
    k_heap_init(&runtime_heap, runtime_heap_storage, sizeof(runtime_heap_storage));
    return 0;
}
SYS_INIT(initialize_runtime_heap, APPLICATION, CONFIG_KERNEL_INIT_PRIORITY_DEFAULT);

IREE_API_EXPORT iree_status_t rtos_allocator_system_ctl(
    void *self, iree_allocator_command_t command, const void *parameters, void **pointer
) {
    (void)self;
    if (!pointer) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
    if (command == IREE_ALLOCATOR_COMMAND_FREE) {
        if (*pointer) k_heap_free(&runtime_heap, (torq_host_allocation_header_t *)*pointer - 1);
        *pointer = NULL;
        return iree_ok_status();
    }
    if (command != IREE_ALLOCATOR_COMMAND_MALLOC && command != IREE_ALLOCATOR_COMMAND_CALLOC &&
        command != IREE_ALLOCATOR_COMMAND_REALLOC) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
    }
    if (!parameters) return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
    size_t size = ((const iree_allocator_alloc_params_t *)parameters)->byte_length;
    if (size > UINT32_MAX ||
        size > SIZE_MAX - sizeof(torq_host_allocation_header_t) - (TORQ_HOST_ALIGNMENT - 1)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    torq_host_allocation_header_t *previous =
        command == IREE_ALLOCATOR_COMMAND_REALLOC && *pointer
            ? (torq_host_allocation_header_t *)*pointer - 1 : NULL;
    if (!size) {
        if (previous) k_heap_free(&runtime_heap, previous);
        *pointer = NULL;
        return iree_ok_status();
    }
    size_t allocation_size = ROUND_UP(sizeof(torq_host_allocation_header_t) + size,
                                      TORQ_HOST_ALIGNMENT);
    torq_host_allocation_header_t *header = k_heap_aligned_alloc(
        &runtime_heap, TORQ_HOST_ALIGNMENT, allocation_size, K_NO_WAIT);
    if (!header) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
    header->size = size;
    void *data = header + 1;
    if (command == IREE_ALLOCATOR_COMMAND_CALLOC || command == IREE_ALLOCATOR_COMMAND_REALLOC) {
        memset(data, 0, size);
    }
    if (previous) memcpy(data, *pointer, MIN(size, previous->size));
    if (cache_clean_addr(data, (uint32_t)size) != 0) {
        k_heap_free(&runtime_heap, header);
        return iree_make_status(IREE_STATUS_INTERNAL);
    }
    if (previous) k_heap_free(&runtime_heap, previous);
    *pointer = data;
    return iree_ok_status();
}