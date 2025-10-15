#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct torq_bitstream_segment_t {
    uint32_t lram_addr;
    uint32_t xram_addr;
    uint32_t size;
    uint8_t *data;
    struct torq_bitstream_segment_t *next;
} torq_bitstream_segment_t;

#ifdef __cplusplus
}
#endif
