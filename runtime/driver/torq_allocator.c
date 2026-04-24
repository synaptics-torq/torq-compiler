// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq_allocator.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "TorqDeviceBufferAPI.h"
#include "iree/base/api.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/resource.h"

typedef struct iree_hal_torq_buffer_t {
    iree_hal_buffer_t base;
    iree_allocator_t host_allocator;
    torq_hw_device_buffer_t device_buffer;
} iree_hal_torq_buffer_t;

typedef struct iree_hal_torq_allocator_t {
    iree_hal_resource_t resource;
    iree_allocator_t host_allocator;
    iree_string_view_t identifier;
    int dma_heap_node_acquired;
} iree_hal_torq_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_torq_allocator_vtable;
static const iree_hal_buffer_vtable_t iree_hal_torq_buffer_vtable;

static iree_hal_torq_allocator_t *
iree_hal_torq_allocator_cast(iree_hal_allocator_t *IREE_RESTRICT base_value) {
    IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_torq_allocator_vtable);
    return (iree_hal_torq_allocator_t *)base_value;
}

static iree_hal_torq_buffer_t *iree_hal_torq_buffer_cast(iree_hal_buffer_t *base_value) {
    IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_torq_buffer_vtable);
    return (iree_hal_torq_buffer_t *)base_value;
}

static const iree_hal_torq_buffer_t *
iree_hal_torq_buffer_const_cast(const iree_hal_buffer_t *base_value) {
    IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_torq_buffer_vtable);
    return (const iree_hal_torq_buffer_t *)base_value;
}

bool iree_hal_torq_buffer_isa(const iree_hal_buffer_t *buffer) {
    return buffer && iree_hal_resource_is((const iree_hal_resource_t *)&buffer->resource,
                                          &iree_hal_torq_buffer_vtable);
}

const torq_hw_device_buffer_t *iree_hal_torq_buffer_device_buffer(const iree_hal_buffer_t *buffer) {
    if (!iree_hal_torq_buffer_isa(buffer)) {
        return NULL;
    }
    return &iree_hal_torq_buffer_const_cast(buffer)->device_buffer;
}

static void iree_hal_torq_buffer_destroy(iree_hal_buffer_t *base_buffer) {
    iree_hal_torq_buffer_t *buffer = iree_hal_torq_buffer_cast(base_buffer);
    iree_allocator_t host_allocator = buffer->host_allocator;
    IREE_IGNORE_ERROR(torq_hw_device_buffer_free(&buffer->device_buffer));
    iree_allocator_free(host_allocator, buffer);
}

static iree_status_t iree_hal_torq_buffer_map_range(
    iree_hal_buffer_t *base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t *mapping
) {
    iree_hal_torq_buffer_t *buffer = iree_hal_torq_buffer_cast(base_buffer);
    uint8_t *data = (uint8_t *)torq_hw_device_buffer_data(&buffer->device_buffer) + local_byte_offset;
    mapping->contents = iree_make_byte_span(data, local_byte_length);
#ifndef NDEBUG
    if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_DISCARD)) {
        memset(mapping->contents.data, 0xCD, local_byte_length);
    }
#endif // !NDEBUG
    return iree_ok_status();
}

static iree_status_t iree_hal_torq_buffer_unmap_range(
    iree_hal_buffer_t *base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t *mapping
) {
    (void)base_buffer;
    (void)local_byte_offset;
    (void)local_byte_length;
    (void)mapping;
    return iree_ok_status();
}

static iree_status_t iree_hal_torq_buffer_invalidate_range(
    iree_hal_buffer_t *base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length
) {
    iree_hal_torq_buffer_t *buffer = iree_hal_torq_buffer_cast(base_buffer);
    return torq_hw_device_buffer_invalidate_range(
        &buffer->device_buffer, (size_t)local_byte_offset, (size_t)local_byte_length
    );
}

static iree_status_t iree_hal_torq_buffer_flush_range(
    iree_hal_buffer_t *base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length
) {
    iree_hal_torq_buffer_t *buffer = iree_hal_torq_buffer_cast(base_buffer);
    return torq_hw_device_buffer_flush_range(
        &buffer->device_buffer, (size_t)local_byte_offset, (size_t)local_byte_length
    );
}

iree_status_t iree_hal_torq_allocator_create(
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    iree_hal_allocator_t **out_allocator
) {
    IREE_ASSERT_ARGUMENT(out_allocator);
    *out_allocator = NULL;

    iree_hal_torq_allocator_t *allocator = NULL;
    iree_host_size_t total_size = iree_sizeof_struct(*allocator) + identifier.size;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size, (void **)&allocator));

    iree_hal_resource_initialize(&iree_hal_torq_allocator_vtable, &allocator->resource);
    allocator->host_allocator = host_allocator;
    allocator->dma_heap_node_acquired = 0;
    iree_string_view_append_to_buffer(
        identifier, &allocator->identifier, (char *)allocator + iree_sizeof_struct(*allocator)
    );

#ifdef ENABLE_ASTRA_MACHINA
    iree_status_t node_status = torq_hw_dma_heap_node_acquire();
    if (!iree_status_is_ok(node_status)) {
        iree_allocator_free(host_allocator, allocator);
        return node_status;
    }
    allocator->dma_heap_node_acquired = 1;
#endif // ENABLE_ASTRA_MACHINA

    *out_allocator = (iree_hal_allocator_t *)allocator;
    return iree_ok_status();
}

static void iree_hal_torq_allocator_destroy(iree_hal_allocator_t *IREE_RESTRICT base_allocator) {
    iree_hal_torq_allocator_t *allocator = iree_hal_torq_allocator_cast(base_allocator);
#ifdef ENABLE_ASTRA_MACHINA
    if (allocator->dma_heap_node_acquired) {
        torq_hw_dma_heap_node_release();
        allocator->dma_heap_node_acquired = 0;
    }
#endif // ENABLE_ASTRA_MACHINA
    iree_allocator_free(allocator->host_allocator, allocator);
}

static iree_allocator_t
iree_hal_torq_allocator_host_allocator(const iree_hal_allocator_t *IREE_RESTRICT base_allocator) {
    const iree_hal_torq_allocator_t *allocator = (const iree_hal_torq_allocator_t *)base_allocator;
    return allocator->host_allocator;
}

static iree_status_t iree_hal_torq_allocator_trim(iree_hal_allocator_t *IREE_RESTRICT base_allocator
) {
    (void)base_allocator;
    return iree_ok_status();
}

static void iree_hal_torq_allocator_query_statistics(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t *IREE_RESTRICT out_statistics
) {
    (void)base_allocator;
    memset(out_statistics, 0, sizeof(*out_statistics));
}

static iree_status_t iree_hal_torq_allocator_query_memory_heaps(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator, iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t *IREE_RESTRICT heaps, iree_host_size_t *IREE_RESTRICT out_count
) {
    (void)base_allocator;
    const iree_host_size_t count = 1;
    if (out_count)
        *out_count = count;
    if (capacity < count) {
        return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
    }
    heaps[0] = (iree_hal_allocator_memory_heap_t){
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                IREE_HAL_MEMORY_TYPE_HOST_CACHED,
        .allowed_usage =
            IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH |
            IREE_HAL_BUFFER_USAGE_SHARING_EXPORT | IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE |
            IREE_HAL_BUFFER_USAGE_SHARING_CONCURRENT | IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE |
            IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED | IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
            IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL | IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
            IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE,
        .max_allocation_size = ~(iree_device_size_t)0,
        .min_alignment = IREE_HAL_HEAP_BUFFER_ALIGNMENT,
    };
    return iree_ok_status();
}

static iree_hal_buffer_compatibility_t iree_hal_torq_allocator_query_buffer_compatibility(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t *IREE_RESTRICT params,
    iree_device_size_t *IREE_RESTRICT allocation_size
) {
    (void)base_allocator;
    (void)allocation_size;
    iree_hal_buffer_compatibility_t compatibility = IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                                                    IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE |
                                                    IREE_HAL_BUFFER_COMPATIBILITY_EXPORTABLE;

    if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
        if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
            compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
        }
        if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
            compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
        }
    }

    params->type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_CACHED;
    params->type &= ~(IREE_HAL_MEMORY_TYPE_OPTIMAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT);
    params->usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                     IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
                     IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM | IREE_HAL_BUFFER_USAGE_TRANSFER;

    return compatibility;
}

static iree_status_t iree_hal_torq_allocator_allocate_buffer(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t *IREE_RESTRICT params, iree_device_size_t allocation_size,
    iree_hal_buffer_t **IREE_RESTRICT out_buffer
) {
    iree_hal_torq_allocator_t *allocator = iree_hal_torq_allocator_cast(base_allocator);

    iree_hal_buffer_params_t compat_params = *params;
    if (!iree_all_bits_set(
            iree_hal_torq_allocator_query_buffer_compatibility(
                base_allocator, &compat_params, &allocation_size
            ),
            IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE
        )) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "allocator cannot allocate a buffer with the given parameters"
        );
    }

    iree_hal_torq_buffer_t *buffer = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator->host_allocator, sizeof(*buffer), (void **)&buffer)
    );
    memset(buffer, 0, sizeof(*buffer));
    buffer->host_allocator = allocator->host_allocator;
    buffer->device_buffer.handle = TORQ_HW_INVALID_FD;

    torq_hw_device_buffer_mode_t allocation_mode = TORQ_HW_DEVICE_BUFFER_MODE_MALLOC;
#ifdef ENABLE_ASTRA_MACHINA
    allocation_mode = TORQ_HW_DEVICE_BUFFER_MODE_DMA_HEAP;
#endif // ENABLE_ASTRA_MACHINA

    iree_status_t alloc_status =
        torq_hw_device_buffer_allocate(allocation_mode, allocation_size, &buffer->device_buffer);
    if (!iree_status_is_ok(alloc_status)) {
        iree_allocator_free(allocator->host_allocator, buffer);
        return alloc_status;
    }

    const iree_hal_buffer_placement_t placement = {
        .device = NULL,
        .queue_affinity = compat_params.queue_affinity ? compat_params.queue_affinity
                                                       : IREE_HAL_QUEUE_AFFINITY_ANY,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    iree_hal_buffer_initialize(
        placement, &buffer->base, allocation_size, 0, allocation_size, compat_params.type,
        compat_params.access, compat_params.usage, &iree_hal_torq_buffer_vtable, &buffer->base
    );
    *out_buffer = &buffer->base;
    return iree_ok_status();
}

static void iree_hal_torq_allocator_deallocate_buffer(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator, iree_hal_buffer_t *IREE_RESTRICT base_buffer
) {
    (void)base_allocator;
    iree_hal_buffer_destroy(base_buffer);
}

static iree_status_t iree_hal_torq_allocator_import_buffer(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t *IREE_RESTRICT params,
    iree_hal_external_buffer_t *IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t **IREE_RESTRICT out_buffer
) {
    iree_hal_torq_allocator_t *allocator = iree_hal_torq_allocator_cast(base_allocator);

    iree_hal_buffer_params_t compat_params = *params;
    iree_device_size_t allocation_size = external_buffer->size;
    if (!iree_all_bits_set(
            iree_hal_torq_allocator_query_buffer_compatibility(
                base_allocator, &compat_params, &allocation_size
            ),
            IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE
        )) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "allocator cannot import a buffer with the given parameters"
        );
    }

    void *ptr = NULL;
    switch (external_buffer->type) {
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION:
        ptr = external_buffer->handle.host_allocation.ptr;
        break;
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION:
        ptr = (void *)((intptr_t)external_buffer->handle.device_allocation.ptr);
        break;
    default:
        return iree_make_status(IREE_STATUS_UNAVAILABLE, "external buffer type not supported");
    }

    const iree_hal_buffer_placement_t placement = {
        .device = NULL,
        .queue_affinity = compat_params.queue_affinity ? compat_params.queue_affinity
                                                       : IREE_HAL_QUEUE_AFFINITY_ANY,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    return iree_hal_heap_buffer_wrap(
        placement, compat_params.type, compat_params.access, compat_params.usage,
        external_buffer->size, iree_make_byte_span(ptr, external_buffer->size), release_callback,
        allocator->host_allocator, out_buffer
    );
}

static iree_status_t iree_hal_torq_allocator_export_buffer(
    iree_hal_allocator_t *IREE_RESTRICT base_allocator, iree_hal_buffer_t *IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t *IREE_RESTRICT out_external_buffer
) {
    (void)base_allocator;
    if (requested_type != IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION &&
        requested_type != IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION) {
        return iree_make_status(IREE_STATUS_UNAVAILABLE, "external buffer type not supported");
    }

    iree_hal_buffer_mapping_t mapping;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_PERSISTENT, iree_hal_buffer_allowed_access(buffer), 0,
        IREE_HAL_WHOLE_BUFFER, &mapping
    ));

    out_external_buffer->type = requested_type;
    out_external_buffer->flags = requested_flags;
    out_external_buffer->size = mapping.contents.data_length;
    if (requested_type == IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION) {
        out_external_buffer->handle.host_allocation.ptr = mapping.contents.data;
    }
    else if (requested_type == IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION) {
        out_external_buffer->handle.device_allocation.ptr =
            (uint64_t)(uintptr_t)mapping.contents.data;
    }
    return iree_ok_status();
}

static const iree_hal_allocator_vtable_t iree_hal_torq_allocator_vtable = {
    .destroy = iree_hal_torq_allocator_destroy,
    .host_allocator = iree_hal_torq_allocator_host_allocator,
    .trim = iree_hal_torq_allocator_trim,
    .query_statistics = iree_hal_torq_allocator_query_statistics,
    .query_memory_heaps = iree_hal_torq_allocator_query_memory_heaps,
    .query_buffer_compatibility = iree_hal_torq_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_torq_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_torq_allocator_deallocate_buffer,
    .import_buffer = iree_hal_torq_allocator_import_buffer,
    .export_buffer = iree_hal_torq_allocator_export_buffer,
};

static const iree_hal_buffer_vtable_t iree_hal_torq_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_torq_buffer_destroy,
    .map_range = iree_hal_torq_buffer_map_range,
    .unmap_range = iree_hal_torq_buffer_unmap_range,
    .invalidate_range = iree_hal_torq_buffer_invalidate_range,
    .flush_range = iree_hal_torq_buffer_flush_range,
};
