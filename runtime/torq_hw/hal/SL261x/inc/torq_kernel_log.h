// SPDX-License-Identifier: GPL-2.0
/* Copyright (C) 2025 Synaptics Incorporated */

#pragma once

#include <linux/kernel.h>

static inline void torq_print(const char* level, const char* function, int line, const char *msg, ...)
{
    char strbuf[256] = {'\0'};
    va_list varg;
    va_start (varg, msg);
    vsnprintf(strbuf, 256, msg, varg);
    va_end(varg);

    printk("%s npu:[%s():%d] %s\n", level, function, line, strbuf);
}

#define KLOGH(...)      torq_print(KERN_CRIT, __func__, __LINE__, __VA_ARGS__);

/*
   log print out
   when set to 1, only enable error logs.
   when set to 2, enable info and error logs.
*/
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 1
#endif

#if (DEBUG_LEVEL == 3)
#define LOG_ENTER()   torq_print(KERN_INFO, __func__, __LINE__, "entering");
#define KLOGI(...)    torq_print(KERN_INFO, __func__, __LINE__, __VA_ARGS__);
#define KLOGE(...)    torq_print(KERN_ERR, __func__, __LINE__, __VA_ARGS__);
#define KLOGD(...)    torq_print(KERN_DEBUG, __func__, __LINE__, __VA_ARGS__);
#elif (DEBUG_LEVEL == 2)
#define LOG_ENTER()   torq_print(KERN_INFO, __func__, __LINE__, "entering");
#define KLOGI(...)    torq_print(KERN_INFO, __func__, __LINE__, __VA_ARGS__);
#define KLOGE(...)    torq_print(KERN_ERR, __func__, __LINE__, __VA_ARGS__);
#define KLOGD(...)
#elif (DEBUG_LEVEL == 1)
#define LOG_ENTER()
#define KLOGI(...)
#define KLOGE(...)    torq_print(KERN_ERR, __func__, __LINE__, __VA_ARGS__);
#define KLOGD(...)
#else
#define LOG_ENTER()
#define KLOGI(...)
#define KLOGE(...)
#define KLOGD(...)
#endif
