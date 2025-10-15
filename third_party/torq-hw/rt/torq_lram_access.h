// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  06/28/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#ifndef TORQ_LRAM_ACCESS_H_
#define TORQ_LRAM_ACCESS_H_

#ifdef __cplusplus
extern "C" {
#endif


//WARNING:
//  Direct accessing LRAM is not recommended!
//  It should only be used in the following scenarios:
//    (1) To load the small "bootstrap" DMA descriptor during initialization
//    (2) To backdoor load LRAM in some DV test cases
//    (3) For debug
//  Use DMA whenever possible.

int torq_lram_write(torq_sys_t *sys, size_t addr, size_t size, uint8_t *data);
int torq_lram_read(torq_sys_t *sys, size_t addr, size_t size, uint8_t *data);
int torq_lram_load_from_file(torq_sys_t *sys, size_t addr, size_t word_size, size_t count, const char *fmt, const char *fname);
int torq_lram_save_to_file(torq_sys_t *sys, size_t addr, size_t word_size, size_t count, const char *fmt, const char *fname);


#ifdef __cplusplus
}
#endif

#endif
