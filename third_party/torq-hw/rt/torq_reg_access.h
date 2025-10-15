// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  06/28/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#ifndef TORQ_REG_ACCESS_H_
#define TORQ_REG_ACCESS_H_

#ifdef __cplusplus
extern "C" {
#endif


void     torq_reg_write32(torq_sys_t *sys, uint32_t addr, uint32_t data);
uint32_t torq_reg_read32(torq_sys_t *sys, uint32_t addr);


#ifdef __cplusplus
}
#endif

#endif
