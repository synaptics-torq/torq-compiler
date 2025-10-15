// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  06/28/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#ifndef TORQ_LOG_H
#define TORQ_LOG_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


#define xdbg printf
#define xmsg printf
#define xerr printf
#define xflush() fflush(stdout)

#define xdie_v(c_,...) do { if (c_) { printf("FATAL ERROR in %s() line %d: ", __func__,__LINE__); printf(__VA_ARGS__); exit(1); } } while (0)
#define xdie(c_) xdie_v((c_),"%s\n",#c_)


#ifdef __cplusplus
}
#endif

#endif
