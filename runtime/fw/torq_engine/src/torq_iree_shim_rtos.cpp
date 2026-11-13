// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file    torq_iree_shim_rtos.cpp
 * \brief   POSIX shim stubs required by the IREE runtime on Zephyr.
 * \details Provides weak definitions of \c gettimeofday and the linker-script
 *          heap symbols so the IREE runtime links cleanly on Zephyr targets.
 */

#include <sys/time.h>

#include <zephyr/kernel.h>

#define TORQ_HEAP_STUB_SIZE  4U
#define TORQ_HEAP_STUB_ALIGN 8U

/**
 * \brief Weak POSIX \c gettimeofday backed by the Zephyr kernel tick counter.
 *
 * Provides a monotonic wall-clock approximation using \c k_uptime_get().
 * The timezone argument is ignored.
 *
 * \param[out] tv  Timeval structure to fill; ignored if NULL.
 * \return Always 0.
 */
extern "C" int __attribute__((weak)) gettimeofday(struct timeval *tv, void *)
{
    if (tv) {
        uint64_t ms = k_uptime_get();

        tv->tv_sec  = ms / MSEC_PER_SEC;
        tv->tv_usec = (ms % MSEC_PER_SEC) * USEC_PER_MSEC;
    }
    return 0;
}

extern "C" {
    alignas(TORQ_HEAP_STUB_ALIGN) char __attribute__((weak)) __heap_start[TORQ_HEAP_STUB_SIZE];
    alignas(TORQ_HEAP_STUB_ALIGN) char __attribute__((weak)) __heap_end[TORQ_HEAP_STUB_SIZE];
}
