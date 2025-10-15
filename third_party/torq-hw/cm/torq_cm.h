// Copyright 2023-2024 Synaptics Incorporated. All rights reserved.
// Created:  09/30/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#ifndef TORQ_CM_H_
#define TORQ_CM_H_

//NOTE: we use multichar constants intentionally
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wmultichar"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#ifndef TORQ_CM_CB_DEFINED
#define TORQ_CM_CB_DEFINED
typedef int (*cb_read_reg_t)            (void *ctx, uint32_t tag, uint32_t addr, uint32_t *data);
typedef int (*cb_write_reg_t)           (void *ctx, uint32_t tag, uint32_t addr, uint32_t *data);
typedef int (*cb_read_mem_t)            (void *ctx, uint32_t tag, size_t addr, size_t size, uint8_t *data);
typedef int (*cb_write_mem_t)           (void *ctx, uint32_t tag, size_t addr, size_t size, uint8_t *data);
typedef int (*cb_get_irq_t)             (void *ctx, uint32_t tag, uint32_t ien, uint32_t *irq);
#endif


// Note: Tag is set by the source hardware module (C model) for performance analysis.
//       If the caller is runtime software (not hardware), set tag = 0.

void *torq_cm__open                    (uint8_t *xram, uint32_t xram_base, size_t xram_size);
int   torq_cm__close                   (void *_self);
int   torq_cm__reset                   (void *_self);
int   torq_cm__set_dump                (void *_self, const char *path);
int   torq_cm__set_css_cpu_code        (void *_self, void (*func)(void *)); // for css cpu behavior model only

// slave interface for parent to access the torq register space
int   torq_cm__read_reg                (void *_self, uint32_t tag, uint32_t addr, uint32_t *data);
int   torq_cm__write_reg               (void *_self, uint32_t tag, uint32_t addr, uint32_t *data);

// irq interface for parent to sample the torq irq output pin (single bit)
int   torq_cm__get_irq                 (void *_self, uint32_t tag, uint32_t ien, uint32_t *irq);

// master interface for torq to access the external memory space
int   torq_cm__set_ext_mem_if          (void *_self, void *mem_ctx, cb_read_mem_t cb_read_mem, cb_write_mem_t cb_write_mem);


#define torq_cm_open(xram_, xram_base_, xram_size_)      torq_cm__open((xram_), (xram_base_), (xram_size_))
#define torq_cm_close(self_)                 torq_cm__close((self_))
#define torq_cm_set_dump(self_, path_)       torq_cm__set_dump((self_), (path_))
#define torq_cm_read32(self_, addr_, data_)  torq_cm__read_reg((self_), 0, (addr_), (data_))
#define torq_cm_write32(self_, addr_, data_) torq_cm__write_reg((self_), 0, (addr_), (data_))


#ifdef __cplusplus
}
#endif

#endif
