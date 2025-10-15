// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  06/28/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#ifndef TORQ_MEM_LIST_H_
#define TORQ_MEM_LIST_H_

#ifdef __cplusplus
extern "C" {
#endif


int torq_proc_mem_lst(torq_sys_t *sys, const char *lst_fname, const char *inp_dir, const char *out_dir);


#ifdef __cplusplus
}
#endif

#endif
