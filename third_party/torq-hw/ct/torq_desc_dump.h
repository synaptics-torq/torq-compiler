// clang-format off
// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  12/01/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#ifndef TORQ_DESC_DUMP_H_
#define TORQ_DESC_DUMP_H_

#include "torq_api.h"

#ifdef __cplusplus
extern "C" {
#endif


void *torq_desc_dump__open(
          const char *path
      );
int   torq_desc_dump__close(
          void *self_
      );
int   torq_desc_dump__ndl_info(
          void *self_,
          int job_id,
          int tsk_id,
          int slc_id,
          uint32_t tag,
          int set_id,
          torq_ndl_cmd_t *cmd
      );
int   torq_desc_dump__data(
          void *self_,
          int job_id,
          int stg_id,
          int tsk_id,
          int slc_id,
          uint32_t mem_tag,
          uint32_t op_tag,
          uint32_t data_tag,
          uint32_t addr_tag,
          uint32_t addr,
          uint32_t size,
          void *data
      );


#ifdef __cplusplus
}
#endif

#endif
