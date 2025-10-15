// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  09/30/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#ifndef TORQ_HW_H_
#define TORQ_HW_H_

#ifdef __cplusplus
extern "C" {
#endif


int torq_hw_start    (torq_sys_t *sys, uint32_t start_lram_addr);
int torq_hw_wait     (torq_sys_t *sys);
int torq_hw_end      (torq_sys_t *sys);
int torq_hw_set_dump (torq_sys_t *sys, const char *dump_dir);


#ifdef __cplusplus
}
#endif

#endif
