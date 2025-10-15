// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  06/28/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#ifndef TORQ_XRAM_ACCESS_H_
#define TORQ_XRAM_ACCESS_H_

#ifdef __cplusplus
extern "C" {
#endif


int torq_xram_write(torq_sys_t *sys, size_t addr, size_t size, uint8_t *data);
int torq_xram_read(torq_sys_t *sys, size_t addr, size_t size, uint8_t *data);
int torq_xram_load_from_file(torq_sys_t *sys, size_t addr, size_t word_size, size_t count, const char *fmt, const char *fname);
int torq_xram_save_to_file(torq_sys_t *sys, size_t addr, size_t word_size, size_t count, const char *fmt, const char *fname);


#ifdef __cplusplus
}
#endif

#endif
