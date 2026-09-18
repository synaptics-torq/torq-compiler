// Copyright 2026 Synaptics Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Model-agnostic autoregressive (KV-cache) decode loop driver for Torq NPU
// decoder-only LLMs. Driven by a model directory:
//   --model-dir=<dir> containing:
//     config.json          num_hidden_layers, hidden_size, vocab_size,
//                          head_dim, num_key_value_heads, eos_token_id
//     <body>.vmfb          compiled KV-cache body (N+2 inputs, N+1 outputs;
//                          inputs: token_embedding, position_ids, N x past_kv)
//     lm_head*.vmfb        optional on-NPU lm-head ([1,1,H] -> [1,1,V])
//     token_embeddings.npy [vocab, hidden] bf16 embedding table (mmap'd)
//     token_id_lut.npy     optional trimmed-vocab remap (1-D i32/i64)
//
// Decode loop:
//   embed(token) -> body(hidden, present.*) -> lm-head -> argmax -> next.
// When the body vmfb ties each output to an input (iree.abi.tied), the KV
// cache is updated in place: the driver allocates the input views once and
// only rewrites the embedding/position buffers between steps. Aliasing is
// verified on the first step; on mismatch it falls back to rebuilding the
// past list from the outputs every step (--no-tied-kv forces the fallback).
// When no lm_head module is present, a tiled NEON host lm-head over the
// embedding table is used instead.

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/tooling/context_util.h"
#include "iree/vm/api.h"

#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Driver flags (parsed by us; everything else goes to iree_flags_parse).
// ---------------------------------------------------------------------------
static const char* FLAG_model_dir = NULL;
static int FLAG_max_new_tokens = 8;
static int FLAG_past_len = 256;
static int FLAG_vocab_size = 0;    // 0 = take from config/npy
static int FLAG_hidden_dim = 0;    // 0 = take from config/npy
static int FLAG_n_layer = 0;       // 0 = take from body signature/config
static int FLAG_head_dim = 0;      // 0 = take from config
static int FLAG_kv_heads = 0;      // 0 = take from config (default 1)
static int FLAG_first_token = 40;
static int FLAG_start_position = 0;
static const char* FLAG_lmhead_weight = NULL;  // legacy [hidden, vocab] bf16
static const char* FLAG_prompt_tokens = NULL;  // e.g. "2,105,2364"
static int FLAG_no_tied_kv = 0;
static int FLAG_host_lmhead = 0;
static int FLAG_one_context = 0;

// ---------------------------------------------------------------------------
// Model config (from config.json + flag overrides).
// ---------------------------------------------------------------------------
#define MAX_EOS_IDS 8
typedef struct {
  int n_layer;
  int hidden_dim;
  int vocab_size;
  int head_dim;
  int kv_heads;
  int eos_ids[MAX_EOS_IDS];
  int n_eos;
} model_config_t;
static model_config_t g_cfg = {18, 640, 262144, 256, 1, {1}, 1};

static bool config_is_eos(int32_t token) {
  for (int i = 0; i < g_cfg.n_eos; ++i)
    if (g_cfg.eos_ids[i] == token) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Minimal JSON scanner: find "key" and return a pointer just past the ':'.
// Only handles the flat scalar/array-of-int values we need; no nesting.
// ---------------------------------------------------------------------------
static const char* json_find(const char* text, const char* key) {
  char pat[128];
  snprintf(pat, sizeof(pat), "\"%s\"", key);
  const char* p = strstr(text, pat);
  if (!p) return NULL;
  p += strlen(pat);
  while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ':'))
    ++p;
  return p;
}

static bool json_int(const char* text, const char* key, int* out) {
  const char* p = json_find(text, key);
  if (!p || (*p != '-' && (*p < '0' || *p > '9'))) return false;
  *out = (int)strtol(p, NULL, 10);
  return true;
}

// eos_token_id may be a scalar or an array like [1, 106].
static void json_int_list(const char* text, const char* key, int* out, int max_n,
                          int* out_n) {
  *out_n = 0;
  const char* p = json_find(text, key);
  if (!p) return;
  if (*p == '[') {
    ++p;
    while (*p && *p != ']' && *out_n < max_n) {
      while (*p == ' ' || *p == ',' || *p == '\n' || *p == '\t') ++p;
      if (*p == '-' || (*p >= '0' && *p <= '9')) {
        out[(*out_n)++] = (int)strtol(p, (char**)&p, 10);
      } else if (*p != ']') {
        ++p;
      }
    }
  } else if (*p == '-' || (*p >= '0' && *p <= '9')) {
    out[(*out_n)++] = (int)strtol(p, NULL, 10);
  }
}

static bool load_config_json(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::string text;
  text.resize((size_t)size);
  size_t got = fread(&text[0], 1, (size_t)size, f);
  fclose(f);
  if (got != (size_t)size) return false;

  json_int(text.c_str(), "num_hidden_layers", &g_cfg.n_layer);
  json_int(text.c_str(), "hidden_size", &g_cfg.hidden_dim);
  json_int(text.c_str(), "vocab_size", &g_cfg.vocab_size);
  json_int(text.c_str(), "head_dim", &g_cfg.head_dim);
  if (!json_int(text.c_str(), "num_key_value_heads", &g_cfg.kv_heads))
    g_cfg.kv_heads = 1;
  json_int_list(text.c_str(), "eos_token_id", g_cfg.eos_ids, MAX_EOS_IDS,
                &g_cfg.n_eos);
  return true;
}

// ---------------------------------------------------------------------------
// .npy v1.0 reader (mmap). Only what's needed: descr, rank, shape, data ptr.
// ---------------------------------------------------------------------------
typedef struct {
  void* data;         // mmap'd payload
  size_t data_bytes;  // payload size in bytes
  int elem_size;      // bytes per element
  char descr[8];      // e.g. "<i8", "<V2", "<f4"
  int64_t shape[8];
  int rank;
  void* map_base;
  size_t map_len;
} npy_array_t;

static bool npy_mmap(const char* path, npy_array_t* out) {
  memset(out, 0, sizeof(*out));
  int fd = open(path, O_RDONLY);
  if (fd < 0) return false;
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    return false;
  }
  size_t len = (size_t)st.st_size;
  void* base = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (base == MAP_FAILED) return false;
  const unsigned char* p = (const unsigned char*)base;
  if (len < 10 || memcmp(p, "\x93NUMPY", 6) != 0 || p[6] != 1) {
    munmap(base, len);
    return false;  // only v1.x supported
  }
  uint16_t hlen = (uint16_t)(p[8] | (p[9] << 8));
  if (10 + (size_t)hlen > len) {
    munmap(base, len);
    return false;
  }
  std::string header((const char*)p + 10, (size_t)hlen);

  // 'descr': '<V2' etc — single-quoted key and value.
  const char* d = strstr(header.c_str(), "'descr'");
  if (d) {
    d = strchr(d + 7, '\'');
    if (d) ++d;
  }
  if (!d) {
    munmap(base, len);
    return false;
  }
  size_t di = 0;
  while (*d && *d != '\'' && di < sizeof(out->descr) - 1) out->descr[di++] = *d++;
  out->descr[di] = 0;

  const char* s = strstr(header.c_str(), "'shape'");
  if (!s || !(s = strchr(s, '('))) {
    munmap(base, len);
    return false;
  }
  ++s;
  out->rank = 0;
  int64_t elems = 1;
  while (*s && *s != ')' && out->rank < 8) {
    while (*s == ' ' || *s == ',') ++s;
    if (*s >= '0' && *s <= '9') {
      out->shape[out->rank] = strtoll(s, (char**)&s, 10);
      elems *= out->shape[out->rank];
      ++out->rank;
    } else if (*s != ')') {
      ++s;
    }
  }

  // Element size from descr tail, e.g. "<i8" -> 8, "<V2" -> 2, "|u1" -> 1.
  int es = atoi(out->descr + 2);
  if (es <= 0) es = atoi(out->descr + 1);
  if (es <= 0) {
    munmap(base, len);
    return false;
  }
  out->elem_size = es;
  out->data = (void*)(p + 10 + hlen);
  out->data_bytes = (size_t)elems * (size_t)es;
  out->map_base = base;
  out->map_len = len;
  return true;
}

static void npy_unmap(npy_array_t* a) {
  if (a->map_base) munmap(a->map_base, a->map_len);
  a->map_base = NULL;
}

// ---------------------------------------------------------------------------
// Embedding table / legacy lm-head weight (bf16).
// ---------------------------------------------------------------------------
static npy_array_t g_embeddings;       // [vocab, hidden] bf16 (descr <V2)
static bool g_has_embeddings = false;
static npy_array_t g_lut;              // [Vcompact] i32/i64
static bool g_has_lut = false;
static std::vector<int64_t> g_lut_host;

static uint16_t* g_lmhead_w = NULL;  // legacy [hidden, vocab] column table

static inline float bf16_to_f32(uint16_t b) {
  union {
    uint32_t u;
    float f;
  } cvt;
  cvt.u = ((uint32_t)b) << 16;
  return cvt.f;
}

#if defined(__aarch64__)
#include <arm_neon.h>
// logits[v] += h * wrow[v] over n elems (bf16 weight row, f32 accumulate).
static void lm_head_row_neon(const uint16_t* wrow, float h, float* logits, int n) {
  float32x4_t hv = vdupq_n_f32(h);
  int v = 0;
  for (; v + 8 <= n; v += 8) {
    uint16x8_t b16 = vld1q_u16(wrow + v);
    uint32x4_t lo32 = vshll_n_u16(vget_low_u16(b16), 16);
    uint32x4_t hi32 = vshll_n_u16(vget_high_u16(b16), 16);
    float32x4_t flo = vreinterpretq_f32_u32(lo32);
    float32x4_t fhi = vreinterpretq_f32_u32(hi32);
    float32x4_t l0 = vld1q_f32(logits + v);
    float32x4_t l1 = vld1q_f32(logits + v + 4);
    vst1q_f32(logits + v, vfmaq_f32(l0, hv, flo));
    vst1q_f32(logits + v + 4, vfmaq_f32(l1, hv, fhi));
  }
  for (; v < n; ++v) logits[v] += h * bf16_to_f32(wrow[v]);
}

// dot(hidden_f32, row_bf16) for one vocab row of the embedding table.
static float emb_row_dot_neon(const uint16_t* row, const float* hf, int n) {
  float32x4_t acc0 = vdupq_n_f32(0.f);
  float32x4_t acc1 = vdupq_n_f32(0.f);
  int d = 0;
  for (; d + 8 <= n; d += 8) {
    uint16x8_t b16 = vld1q_u16(row + d);
    uint32x4_t lo32 = vshll_n_u16(vget_low_u16(b16), 16);
    uint32x4_t hi32 = vshll_n_u16(vget_high_u16(b16), 16);
    acc0 = vfmaq_f32(acc0, vld1q_f32(hf + d), vreinterpretq_f32_u32(lo32));
    acc1 = vfmaq_f32(acc1, vld1q_f32(hf + d + 4), vreinterpretq_f32_u32(hi32));
  }
  float32x4_t acc = vaddq_f32(acc0, acc1);
  float total = vaddvq_f32(acc);
  for (; d < n; ++d) total += hf[d] * bf16_to_f32(row[d]);
  return total;
}
#endif

static inline float emb_row_dot(const uint16_t* row, const float* hf, int n) {
#if defined(__aarch64__)
  return emb_row_dot_neon(row, hf, n);
#else
  float total = 0.f;
  for (int d = 0; d < n; ++d) total += hf[d] * bf16_to_f32(row[d]);
  return total;
#endif
}

// ---------------------------------------------------------------------------
// Per-phase timing instrumentation (TORQ_DECODE_TIMING=1).
// ---------------------------------------------------------------------------
static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static double g_t_embed = 0, g_t_h2d = 0, g_t_inputs = 0, g_t_invoke = 0,
              g_t_d2h = 0, g_t_lmhead = 0, g_t_lmhead_npu = 0, g_t_feedback = 0;
static int g_t_enabled = 0;

// ---------------------------------------------------------------------------
// Device buffer view helpers.
// ---------------------------------------------------------------------------
static iree_status_t iree_decode_create_view(
    iree_hal_allocator_t* device_allocator, iree_hal_device_t* device,
    iree_allocator_t host_allocator, const void* data, iree_device_size_t byte_length,
    const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
    iree_hal_element_type_t element_type, iree_hal_buffer_view_t** out_view) {
  *out_view = NULL;
  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      device_allocator, params, byte_length, &buffer));
  iree_status_t status = iree_hal_device_transfer_h2d(
      device, data, buffer, 0, byte_length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout());
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_create(
        buffer, shape_rank, shape, element_type,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, host_allocator, out_view);
  }
  iree_hal_buffer_release(buffer);
  return status;
}

// H2D into an already-allocated view's buffer (persistent-input path).
static iree_status_t iree_decode_h2d_view(iree_hal_device_t* device,
                                          iree_hal_buffer_view_t* view,
                                          const void* data,
                                          iree_device_size_t byte_length) {
  return iree_hal_device_transfer_h2d(
      device, data, iree_hal_buffer_view_buffer(view), 0, byte_length,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
}

static iree_status_t iree_decode_push_view(iree_vm_list_t* list,
                                           iree_hal_buffer_view_t* v) {
  iree_vm_ref_t ref = iree_hal_buffer_view_retain_ref(v);
  iree_status_t status = iree_vm_list_push_ref_retain(list, &ref);
  iree_vm_ref_release(&ref);
  iree_hal_buffer_view_release(v);
  return status;
}

static iree_status_t iree_decode_get_view(iree_vm_list_t* list, iree_host_size_t i,
                                          iree_hal_buffer_view_t** out_view) {
  *out_view = iree_vm_list_get_buffer_view_retain(list, i);
  if (!*out_view) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "list element %d is not a buffer view", (int)i);
  }
  return iree_ok_status();
}

// ---------------------------------------------------------------------------
// Persistent tied-operand state.
// ---------------------------------------------------------------------------
typedef struct {
  iree_hal_buffer_view_t* emb_view;   // [1,1,hidden] bf16 (tied to output 0)
  iree_hal_buffer_view_t* pos_view;   // [1,1] si32
  std::vector<iree_hal_buffer_view_t*> kv_views;  // n_layer x [1,2kv,past,hd]
  iree_vm_list_t* inputs;             // [emb, pos, kv...] reused every step
} tied_state_t;

static void tied_state_release(tied_state_t* t) {
  if (t->emb_view) iree_hal_buffer_view_release(t->emb_view);
  if (t->pos_view) iree_hal_buffer_view_release(t->pos_view);
  for (auto* v : t->kv_views) iree_hal_buffer_view_release(v);
  t->kv_views.clear();
  if (t->inputs) iree_vm_list_release(t->inputs);
  t->inputs = NULL;
  t->emb_view = t->pos_view = NULL;
}

static iree_status_t tied_state_init(tied_state_t* t, iree_hal_allocator_t* da,
                                     iree_hal_device_t* device,
                                     iree_allocator_t host_allocator, int n_layer,
                                     int past_len) {
  t->emb_view = NULL;
  t->pos_view = NULL;
  t->inputs = NULL;
  iree_status_t status = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             2 + n_layer, host_allocator, &t->inputs);
  if (!iree_status_is_ok(status)) return status;

  // token_embedding [1,1,hidden] bf16 (contents rewritten each step).
  const iree_hal_dim_t emb_shape[3] = {1, 1, (iree_hal_dim_t)g_cfg.hidden_dim};
  std::vector<uint16_t> zeros_emb(g_cfg.hidden_dim, 0);
  status = iree_decode_create_view(da, device, host_allocator, zeros_emb.data(),
                                   (iree_device_size_t)g_cfg.hidden_dim * 2,
                                   emb_shape, 3, IREE_HAL_ELEMENT_TYPE_BFLOAT_16,
                                   &t->emb_view);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_view_retain(t->emb_view);
    status = iree_decode_push_view(t->inputs, t->emb_view);
  }

  // position_ids [1,1] si32 (rewritten each step).
  const iree_hal_dim_t scalar_shape[2] = {1, 1};
  int32_t zero_pos = 0;
  if (iree_status_is_ok(status)) {
    status = iree_decode_create_view(da, device, host_allocator, &zero_pos,
                                     sizeof(int32_t), scalar_shape, 2,
                                     IREE_HAL_ELEMENT_TYPE_SINT_32, &t->pos_view);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_view_retain(t->pos_view);
    status = iree_decode_push_view(t->inputs, t->pos_view);
  }

  // Zero past_key_values: n_layer x [1, 2*kv_heads, past, head_dim] bf16.
  const iree_hal_dim_t kv_shape[4] = {1, (iree_hal_dim_t)(2 * g_cfg.kv_heads),
                                      (iree_hal_dim_t)past_len,
                                      (iree_hal_dim_t)g_cfg.head_dim};
  size_t kv_elems = (size_t)2 * g_cfg.kv_heads * past_len * g_cfg.head_dim;
  std::vector<uint16_t> zeros_kv(kv_elems, 0);
  for (int i = 0; i < n_layer && iree_status_is_ok(status); ++i) {
    iree_hal_buffer_view_t* v = NULL;
    status = iree_decode_create_view(da, device, host_allocator, zeros_kv.data(),
                                     (iree_device_size_t)kv_elems * 2, kv_shape, 4,
                                     IREE_HAL_ELEMENT_TYPE_BFLOAT_16, &v);
    if (iree_status_is_ok(status)) {
      t->kv_views.push_back(v);
      iree_hal_buffer_view_retain(v);
      status = iree_decode_push_view(t->inputs, v);
    }
  }
  if (!iree_status_is_ok(status)) {
    tied_state_release(t);
    return status;
  }
  return iree_ok_status();
}

// Verifies that each output aliases its tied input buffer:
// output0 -> emb input, output i (>=1) -> kv input i-1.
static bool tied_state_validate(const tied_state_t* t, iree_vm_list_t* outputs,
                                int n_layer) {
  iree_host_size_t out_size = iree_vm_list_size(outputs);
  if ((int)out_size < 1 + n_layer) return false;
  for (int i = 0; i <= n_layer; ++i) {
    iree_hal_buffer_view_t* out_view = NULL;
    if (!iree_status_is_ok(iree_decode_get_view(outputs, (iree_host_size_t)i,
                                                &out_view)))
      return false;
    iree_hal_buffer_t* ob = iree_hal_buffer_view_buffer(out_view);
    iree_hal_buffer_t* eb =
        (i == 0) ? iree_hal_buffer_view_buffer(t->emb_view)
                 : iree_hal_buffer_view_buffer(t->kv_views[i - 1]);
    iree_hal_buffer_view_release(out_view);
    if (ob != eb) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Fallback (non-tied) path: fresh inputs each step, feedback from outputs.
// ---------------------------------------------------------------------------
static iree_status_t iree_decode_push_view_new(
    iree_hal_allocator_t* device_allocator, iree_hal_device_t* device,
    iree_allocator_t host_allocator, iree_vm_list_t* list, const void* data,
    iree_device_size_t byte_length, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type) {
  iree_hal_buffer_view_t* v = NULL;
  IREE_RETURN_IF_ERROR(iree_decode_create_view(device_allocator, device,
                                               host_allocator, data, byte_length,
                                               shape, shape_rank, element_type, &v));
  return iree_decode_push_view(list, v);
}

static iree_status_t iree_decode_build_inputs(
    iree_hal_allocator_t* device_allocator, iree_hal_device_t* device,
    iree_allocator_t host_allocator, const uint16_t* token_emb_bf16,
    int32_t position_id, iree_vm_list_t* past_views, int n_layer,
    iree_vm_list_t** out_inputs) {
  iree_vm_list_t* inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                           2 + n_layer, host_allocator, &inputs));

  const iree_hal_dim_t emb_shape[3] = {1, 1, (iree_hal_dim_t)g_cfg.hidden_dim};
  iree_status_t status = iree_decode_push_view_new(
      device_allocator, device, host_allocator, inputs, token_emb_bf16,
      (iree_device_size_t)g_cfg.hidden_dim * 2, emb_shape, 3,
      IREE_HAL_ELEMENT_TYPE_BFLOAT_16);

  const iree_hal_dim_t scalar_shape[2] = {1, 1};
  if (iree_status_is_ok(status)) {
    status = iree_decode_push_view_new(device_allocator, device, host_allocator,
                                       inputs, &position_id, sizeof(int32_t),
                                       scalar_shape, 2, IREE_HAL_ELEMENT_TYPE_SINT_32);
  }

  for (int i = 0; i < n_layer && iree_status_is_ok(status); ++i) {
    iree_vm_ref_t ref = iree_vm_ref_null();
    status = iree_vm_list_get_ref_retain(past_views, (iree_host_size_t)i, &ref);
    if (iree_status_is_ok(status)) status = iree_vm_list_push_ref_retain(inputs, &ref);
    iree_vm_ref_release(&ref);
  }
  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(inputs);
    return status;
  }
  *out_inputs = inputs;
  return iree_ok_status();
}

static iree_status_t iree_decode_zero_past(
    iree_hal_allocator_t* device_allocator, iree_hal_device_t* device,
    iree_allocator_t host_allocator, int n_layer, int past_len,
    iree_vm_list_t** out_past_views) {
  iree_vm_list_t* past = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                           n_layer, host_allocator, &past));
  const iree_hal_dim_t shape[4] = {1, (iree_hal_dim_t)(2 * g_cfg.kv_heads),
                                   (iree_hal_dim_t)past_len,
                                   (iree_hal_dim_t)g_cfg.head_dim};
  size_t elems = (size_t)2 * g_cfg.kv_heads * past_len * g_cfg.head_dim;
  std::vector<uint16_t> zeros(elems, 0);
  for (int i = 0; i < n_layer; ++i) {
    iree_status_t status = iree_decode_push_view_new(
        device_allocator, device, host_allocator, past, zeros.data(),
        (iree_device_size_t)elems * 2, shape, 4, IREE_HAL_ELEMENT_TYPE_BFLOAT_16);
    if (!iree_status_is_ok(status)) {
      iree_vm_list_release(past);
      return status;
    }
  }
  *out_past_views = past;
  return iree_ok_status();
}

// ---------------------------------------------------------------------------
// Legacy column-major lm-head weight (--lmhead-weight).
// ---------------------------------------------------------------------------
static iree_status_t load_lmhead_weight(void) {
  FILE* f = fopen(FLAG_lmhead_weight, "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", FLAG_lmhead_weight);
    return iree_make_status(IREE_STATUS_NOT_FOUND, "lmhead weight not found");
  }
  size_t n = (size_t)g_cfg.hidden_dim * g_cfg.vocab_size;
  g_lmhead_w = (uint16_t*)malloc(n * 2);
  if (!g_lmhead_w) {
    fclose(f);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "oom");
  }
  size_t got = fread(g_lmhead_w, 2, n, f);
  fclose(f);
  if (got != n) {
    fprintf(stderr, "lmhead weight short read: %zu/%zu\n", got, n);
    return iree_make_status(IREE_STATUS_DATA_LOSS, "lmhead weight short read");
  }
  return iree_ok_status();
}

// ---------------------------------------------------------------------------
// Flag parsing.
// ---------------------------------------------------------------------------
static void iree_decode_parse_driver_flags(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (!strncmp(a, "--model-dir=", 12)) FLAG_model_dir = a + 12;
    else if (!strncmp(a, "--max-new-tokens=", 17)) FLAG_max_new_tokens = atoi(a + 17);
    else if (!strncmp(a, "--past-len=", 11)) FLAG_past_len = atoi(a + 11);
    else if (!strncmp(a, "--vocab-size=", 13)) FLAG_vocab_size = atoi(a + 13);
    else if (!strncmp(a, "--hidden-dim=", 13)) FLAG_hidden_dim = atoi(a + 13);
    else if (!strncmp(a, "--n-layer=", 10)) FLAG_n_layer = atoi(a + 10);
    else if (!strncmp(a, "--head-dim=", 11)) FLAG_head_dim = atoi(a + 11);
    else if (!strncmp(a, "--kv-heads=", 11)) FLAG_kv_heads = atoi(a + 11);
    else if (!strncmp(a, "--first-token=", 14)) FLAG_first_token = atoi(a + 14);
    else if (!strncmp(a, "--start-position=", 17)) FLAG_start_position = atoi(a + 17);
    else if (!strncmp(a, "--lmhead-weight=", 16)) FLAG_lmhead_weight = a + 16;
    else if (!strncmp(a, "--prompt-tokens=", 16)) FLAG_prompt_tokens = a + 16;
    else if (!strcmp(a, "--no-tied-kv")) FLAG_no_tied_kv = 1;
    else if (!strcmp(a, "--host-lmhead")) FLAG_host_lmhead = 1;
    else if (!strcmp(a, "--one-context")) FLAG_one_context = 1;
  }
}

static bool args_have_module_flag(int argc, char** argv) {
  for (int i = 1; i < argc; ++i)
    if (!strncmp(argv[i], "--module=", 9)) return true;
  return false;
}

// Discovers body and lm-head vmfbs in the model dir. A file counts as a vmfb
// if its name contains ".vmfb"; "lm_head*" names go to the head slot.
static void discover_vmfbs(const char* dir, std::string* out_body,
                           std::string* out_head) {
  DIR* d = opendir(dir);
  if (!d) return;
  struct dirent* e;
  while ((e = readdir(d)) != NULL) {
    const char* n = e->d_name;
    if (!strstr(n, ".vmfb")) continue;
    std::string full = std::string(dir) + "/" + n;
    if (!strncmp(n, "lm_head", 7)) {
      if (out_head->empty()) *out_head = full;
    } else {
      if (out_body->empty()) *out_body = full;
    }
  }
  closedir(d);
}

static std::string dir_file(const char* dir, const char* name) {
  return std::string(dir) + "/" + name;
}

static bool file_exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}

// ---------------------------------------------------------------------------
// Counts arguments/results of a vmfb export from its calling convention
// string (`0<args>_<results>`, counting i/I/r chars).
// ---------------------------------------------------------------------------
static void cconv_counts(iree_vm_function_t* fn, int* out_args, int* out_results) {
  *out_args = *out_results = 0;
  iree_vm_function_signature_t sig = iree_vm_function_signature(fn);
  iree_string_view_t cc = sig.calling_convention;
  if (cc.size < 1 || cc.data[0] != '0') return;
  bool results = false;
  for (iree_host_size_t i = 1; i < cc.size; ++i) {
    char c = cc.data[i];
    if (c == '_') {
      results = true;
      continue;
    }
    if (c == 'i' || c == 'I' || c == 'r') {
      if (results) ++*out_results;
      else ++*out_args;
    }
  }
}

// Picks the body (N+2 in / N+1 out, N>=1) and head (1 in / 1 out) exported
// functions from the loaded module list, and records their modules.
static iree_status_t find_body_and_head(iree_tooling_module_list_t* module_list,
                                        iree_vm_function_t* out_body,
                                        iree_vm_function_t* out_head,
                                        int* out_body_n_kv,
                                        iree_vm_module_t** out_body_module,
                                        iree_vm_module_t** out_head_module) {
  memset(out_body, 0, sizeof(*out_body));
  memset(out_head, 0, sizeof(*out_head));
  *out_body_n_kv = 0;
  *out_body_module = NULL;
  *out_head_module = NULL;
  for (iree_host_size_t m = 0; m < module_list->count; ++m) {
    iree_vm_module_t* module = module_list->values[m];
    iree_vm_module_signature_t msig = iree_vm_module_signature(module);
    for (iree_host_size_t o = 0; o < msig.export_function_count; ++o) {
      iree_vm_function_t fn;
      iree_status_t status = iree_vm_module_lookup_function_by_ordinal(
          module, IREE_VM_FUNCTION_LINKAGE_EXPORT, o, &fn);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        continue;
      }
      int args = 0, results = 0;
      cconv_counts(&fn, &args, &results);
      if (args >= 3 && results == args - 1) {
        *out_body = fn;
        *out_body_n_kv = args - 2;
        *out_body_module = module;
      } else if (args == 1 && results == 1) {
        *out_head = fn;
        *out_head_module = module;
      }
    }
  }
  if (!out_body->module) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no KV-cache body export found (need N+2 in / N+1 out)");
  }
  return iree_ok_status();
}

int main(int argc, char** argv) {
  iree_decode_parse_driver_flags(argc, argv);

  // ---- Model dir discovery: config.json, embeddings, LUT, vmfbs. ----
  std::string body_vmfb, head_vmfb;
  if (FLAG_model_dir) {
    std::string cfg_path = dir_file(FLAG_model_dir, "config.json");
    if (load_config_json(cfg_path.c_str())) {
      fprintf(stderr, "config: %s (layers=%d hidden=%d vocab=%d head_dim=%d kv_heads=%d eos=%d...)\n",
              cfg_path.c_str(), g_cfg.n_layer, g_cfg.hidden_dim, g_cfg.vocab_size,
              g_cfg.head_dim, g_cfg.kv_heads, g_cfg.eos_ids[0]);
    } else {
      fprintf(stderr, "warning: no config.json in %s, using defaults/flags\n",
              FLAG_model_dir);
    }
    if (!args_have_module_flag(argc, argv))
      discover_vmfbs(FLAG_model_dir, &body_vmfb, &head_vmfb);

    std::string emb_path = dir_file(FLAG_model_dir, "token_embeddings.npy");
    if (file_exists(emb_path) && npy_mmap(emb_path.c_str(), &g_embeddings)) {
      if (g_embeddings.rank != 2) {
        fprintf(stderr, "token_embeddings.npy must be 2-D [vocab, hidden]\n");
        return EXIT_FAILURE;
      }
      g_has_embeddings = true;
      g_cfg.vocab_size = (int)g_embeddings.shape[0];
      g_cfg.hidden_dim = (int)g_embeddings.shape[1];
      fprintf(stderr, "embeddings: [%lld, %lld] %s mmap'd\n",
              (long long)g_embeddings.shape[0], (long long)g_embeddings.shape[1],
              g_embeddings.descr);
    }
    std::string lut_path = dir_file(FLAG_model_dir, "token_id_lut.npy");
    if (file_exists(lut_path) && npy_mmap(lut_path.c_str(), &g_lut)) {
      if (g_lut.rank == 1 &&
          (!strcmp(g_lut.descr, "<i8") || !strcmp(g_lut.descr, "<i4"))) {
        g_has_lut = true;
        g_lut_host.resize((size_t)g_lut.shape[0]);
        if (g_lut.elem_size == 8) {
          const int64_t* p = (const int64_t*)g_lut.data;
          for (int64_t i = 0; i < g_lut.shape[0]; ++i) g_lut_host[i] = p[i];
        } else {
          const int32_t* p = (const int32_t*)g_lut.data;
          for (int64_t i = 0; i < g_lut.shape[0]; ++i) g_lut_host[i] = p[i];
        }
        fprintf(stderr, "token_id_lut: %lld entries (%s)\n",
                (long long)g_lut.shape[0], g_lut.descr);
      } else {
        fprintf(stderr, "warning: ignoring malformed token_id_lut.npy\n");
      }
    }
  }
  // Flag overrides (explicit flags win over config/npy).
  if (FLAG_vocab_size) g_cfg.vocab_size = FLAG_vocab_size;
  if (FLAG_hidden_dim) g_cfg.hidden_dim = FLAG_hidden_dim;
  if (FLAG_n_layer) g_cfg.n_layer = FLAG_n_layer;
  if (FLAG_head_dim) g_cfg.head_dim = FLAG_head_dim;
  if (FLAG_kv_heads) g_cfg.kv_heads = FLAG_kv_heads;

  // ---- IREE flag parsing (with discovered modules injected). ----
  std::vector<std::string> all_args = {argv[0], "--device=torq"};
  for (int i = 1; i < argc; ++i) all_args.push_back(argv[i]);
  if (!body_vmfb.empty()) all_args.push_back("--module=" + body_vmfb);
  if (!head_vmfb.empty()) all_args.push_back("--module=" + head_vmfb);
  std::vector<char*> arg_ptrs;
  for (auto& a : all_args) arg_ptrs.push_back(const_cast<char*>(a.c_str()));
  int new_argc = (int)arg_ptrs.size();
  char** new_argv = arg_ptrs.data();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &new_argc, &new_argv);

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_vm_instance_t* instance = NULL;
  iree_status_t status = iree_tooling_create_instance(host_allocator, &instance);

  iree_tooling_module_list_t module_list;
  iree_tooling_module_list_initialize(&module_list);
  if (iree_status_is_ok(status)) {
    status = iree_tooling_load_modules_from_flags(instance, host_allocator, &module_list);
  }

  iree_vm_function_t body_fn, head_fn;
  iree_vm_module_t* body_module = NULL;
  iree_vm_module_t* head_module = NULL;
  int n_layer = g_cfg.n_layer;
  if (iree_status_is_ok(status)) {
    int body_n_kv = 0;
    status = find_body_and_head(&module_list, &body_fn, &head_fn, &body_n_kv,
                                &body_module, &head_module);
    if (iree_status_is_ok(status) && body_n_kv > 0) n_layer = body_n_kv;
  }
  bool has_head_module = head_fn.module != NULL && !FLAG_host_lmhead;

  // When a head module is present, run it in a second VM context: contexts
  // own separate HAL executable caches, hence separate XRAM arenas and kernel
  // network leases. In one context both executables share an arena keyed by
  // the device-wide cache; their baked address ranges overlap, so every
  // dispatch clobbers the other module's weight segments and the runtime
  // reloads ~200 MB per switch (~107 ms). Two contexts trade that for a
  // 1280-byte hidden-state roundtrip through the host plus cheap lease
  // preemption. --one-context keeps the old zero-copy chaining for
  // comparison.
  bool two_context = has_head_module && !FLAG_one_context;

  iree_vm_context_t* context = NULL;
  iree_hal_device_t* device = NULL;
  iree_hal_allocator_t* device_allocator = NULL;
  iree_vm_context_t* head_context = NULL;
  iree_hal_device_t* head_device = NULL;
  iree_hal_allocator_t* head_device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    if (two_context) {
      status = iree_tooling_create_context_from_flags(
          instance, 1, &body_module, iree_string_view_empty(), host_allocator,
          &context, &device, &device_allocator);
      if (iree_status_is_ok(status)) {
        status = iree_tooling_create_context_from_flags(
            instance, 1, &head_module, iree_string_view_empty(), host_allocator,
            &head_context, &head_device, &head_device_allocator);
      }
    } else {
      status = iree_tooling_create_context_from_flags(
          instance, module_list.count, module_list.values,
          iree_string_view_empty(), host_allocator, &context, &device,
          &device_allocator);
    }
  }
  iree_tooling_module_list_reset(&module_list);

  // Embedding source: token_embeddings.npy preferred, legacy weight fallback.
  if (iree_status_is_ok(status) && !g_has_embeddings) {
    if (FLAG_lmhead_weight) {
      status = load_lmhead_weight();
    } else {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "need token_embeddings.npy (model dir) or "
                                "--lmhead-weight=<path>");
    }
  }

  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "setup failed: ");
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    iree_vm_instance_release(instance);
    return EXIT_FAILURE;
  }

  fprintf(stderr, "body: %d KV layers | lm-head: %s | tied-kv: %s\n", n_layer,
          has_head_module ? "npu" : "host", FLAG_no_tied_kv ? "off" : "auto");

  // ---- Decode state. ----
  tied_state_t tied = {};
  bool use_tied = !FLAG_no_tied_kv;
  bool tied_validated = false;
  if (use_tied) {
    status = tied_state_init(&tied, device_allocator, device, host_allocator,
                             n_layer, FLAG_past_len);
  }
  iree_vm_list_t* past_views = NULL;  // fallback path only
  if (iree_status_is_ok(status) && !use_tied) {
    status = iree_decode_zero_past(device_allocator, device, host_allocator,
                                   n_layer, FLAG_past_len, &past_views);
  }

  // Prompt token list (prefill).
  std::vector<int32_t> prompt;
  if (FLAG_prompt_tokens) {
    const char* p = FLAG_prompt_tokens;
    while (*p) {
      prompt.push_back((int32_t)strtol(p, (char**)&p, 10));
      if (*p == ',') ++p;
    }
  }

  std::vector<uint16_t> hidden(g_cfg.hidden_dim);
  std::vector<uint16_t> logits_raw;
  std::vector<float> logits;
  std::vector<float> hf(g_cfg.hidden_dim);

  // Two-context head: persistent input view + input list in the head context.
  iree_hal_buffer_view_t* head_input_view = NULL;
  iree_vm_list_t* head_inputs = NULL;
  if (iree_status_is_ok(status) && two_context) {
    const iree_hal_dim_t head_shape[3] = {1, 1, (iree_hal_dim_t)g_cfg.hidden_dim};
    std::vector<uint16_t> zeros(g_cfg.hidden_dim, 0);
    status = iree_decode_create_view(head_device_allocator, head_device,
                                     host_allocator, zeros.data(),
                                     (iree_device_size_t)g_cfg.hidden_dim * 2,
                                     head_shape, 3, IREE_HAL_ELEMENT_TYPE_BFLOAT_16,
                                     &head_input_view);
    if (iree_status_is_ok(status)) {
      iree_hal_buffer_view_retain(head_input_view);
      status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                   host_allocator, &head_inputs);
    }
    if (iree_status_is_ok(status)) {
      status = iree_decode_push_view(head_inputs, head_input_view);
    }
  }

  int n_prompt = (int)prompt.size();
  int32_t token_id = n_prompt > 0 ? prompt[0] : FLAG_first_token;
  int gen_count = 0;       // decode tokens produced
  bool eos_hit = false;
  bool first_step = true;
  double decode_time = 0;  // wall time of token-producing steps only
  int decode_steps = 0;

  g_t_enabled = getenv("TORQ_DECODE_TIMING") != NULL;
  double t_loop0 = now_s();
  for (int step = 0; iree_status_is_ok(status) && !eos_hit &&
                    gen_count < FLAG_max_new_tokens;
       ++step) {
    // All prompt tokens except the final one skip the lm-head.
    bool is_prefill = step < n_prompt - 1;
    double t_step0 = now_s();

    // Host embed look-up: row `token_id` of the [vocab, hidden] table.
    double t0 = now_s();
    const uint16_t* emb_row = NULL;
    std::vector<uint16_t> emb_gather;
    if (g_has_embeddings) {
      emb_row = (const uint16_t*)g_embeddings.data + (size_t)token_id * g_cfg.hidden_dim;
    } else {
      emb_gather.resize(g_cfg.hidden_dim);
      for (int d = 0; d < g_cfg.hidden_dim; ++d)
        emb_gather[d] = g_lmhead_w[(size_t)d * g_cfg.vocab_size + token_id];
      emb_row = emb_gather.data();
    }
    g_t_embed += now_s() - t0;

    int32_t position = FLAG_start_position + step;
    iree_vm_list_t* inputs = NULL;
    bool inputs_persistent = false;
    t0 = now_s();
    if (use_tied) {
      status = iree_decode_h2d_view(device, tied.emb_view, emb_row,
                                    (iree_device_size_t)g_cfg.hidden_dim * 2);
      if (iree_status_is_ok(status)) {
        status = iree_decode_h2d_view(device, tied.pos_view, &position,
                                      sizeof(int32_t));
      }
      g_t_h2d += now_s() - t0;
      inputs = tied.inputs;
      inputs_persistent = true;
    } else {
      status = iree_decode_build_inputs(device_allocator, device, host_allocator,
                                        emb_row, position, past_views, n_layer,
                                        &inputs);
      g_t_inputs += now_s() - t0;
    }

    iree_vm_list_t* outputs = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 32,
                                   host_allocator, &outputs);
    }
    double t1 = now_s();
    if (iree_status_is_ok(status)) {
      status = iree_vm_invoke(context, body_fn, IREE_VM_INVOCATION_FLAG_NONE, NULL,
                              inputs, outputs, host_allocator);
    }
    g_t_invoke += now_s() - t1;
    if (!inputs_persistent && inputs) iree_vm_list_release(inputs);
    if (!iree_status_is_ok(status)) {
      iree_vm_list_release(outputs);
      break;
    }

    // First tied step: verify the HAL really wrote outputs into the tied
    // input buffers. If not, switch to the feedback path for the rest.
    if (use_tied && !tied_validated) {
      tied_validated = true;
      if (tied_state_validate(&tied, outputs, n_layer)) {
        if (g_t_enabled) fprintf(stderr, "tied-kv: output buffers alias inputs (zero-copy)\n");
      } else {
        fprintf(stderr, "tied-kv: aliasing check failed, falling back to feedback path\n");
        use_tied = false;
      }
    }

    // output[0] = hidden state [1,1,hidden] bf16.
    iree_hal_buffer_view_t* hidden_view = NULL;
    status = iree_decode_get_view(outputs, 0, &hidden_view);

    if (!is_prefill && iree_status_is_ok(status)) {
      if (has_head_module) {
        // On-NPU lm-head: body hidden -> head -> logits -> host argmax.
        if (two_context) {
          // Cross-context handoff: D2H the 1280-byte hidden state and H2D it
          // into the head context's persistent input view.
          double t2 = now_s();
          status = iree_hal_device_transfer_d2h(
              device, iree_hal_buffer_view_buffer(hidden_view), 0, hidden.data(),
              (iree_device_size_t)g_cfg.hidden_dim * sizeof(uint16_t),
              IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
          g_t_d2h += now_s() - t2;
          t2 = now_s();
          if (iree_status_is_ok(status)) {
            status = iree_decode_h2d_view(head_device, head_input_view,
                                          hidden.data(),
                                          (iree_device_size_t)g_cfg.hidden_dim * 2);
          }
          g_t_h2d += now_s() - t2;
        } else {
          // Single context: zero-copy — wrap the body's hidden view directly.
          if (head_inputs) iree_vm_list_release(head_inputs);
          head_inputs = NULL;
          status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                       host_allocator, &head_inputs);
          if (iree_status_is_ok(status)) {
            iree_vm_ref_t ref = iree_hal_buffer_view_retain_ref(hidden_view);
            status = iree_vm_list_push_ref_retain(head_inputs, &ref);
            iree_vm_ref_release(&ref);
          }
        }
        double t2 = now_s();
        iree_vm_list_t* head_outputs = NULL;
        if (iree_status_is_ok(status)) {
          status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                       host_allocator, &head_outputs);
        }
        if (iree_status_is_ok(status)) {
          status = iree_vm_invoke(two_context ? head_context : context, head_fn,
                                  IREE_VM_INVOCATION_FLAG_NONE, NULL, head_inputs,
                                  head_outputs, host_allocator);
        }
        g_t_lmhead_npu += now_s() - t2;

        double t3 = now_s();
        iree_hal_buffer_view_t* logits_view = NULL;
        if (iree_status_is_ok(status)) {
          status = iree_decode_get_view(head_outputs, 0, &logits_view);
        }
        iree_host_size_t n_logits = 0;
        if (iree_status_is_ok(status)) {
          n_logits = iree_hal_buffer_view_element_count(logits_view);
          logits_raw.resize(n_logits);
          status = iree_hal_device_transfer_d2h(
              two_context ? head_device : device,
              iree_hal_buffer_view_buffer(logits_view), 0, logits_raw.data(),
              (iree_device_size_t)n_logits * sizeof(uint16_t),
              IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
        }
        if (iree_status_is_ok(status)) {
          int32_t best = 0;
          float best_f = -INFINITY;
          for (iree_host_size_t v = 0; v < n_logits; ++v) {
            float f = bf16_to_f32(logits_raw[v]);
            if (f > best_f) {
              best_f = f;
              best = (int32_t)v;
            }
          }
          if (g_has_lut && (int64_t)best >= (int64_t)g_lut_host.size()) {
            fprintf(stderr, "warning: logit index %d out of LUT range\n", best);
            token_id = best;
          } else {
            token_id = g_has_lut ? (int32_t)g_lut_host[best] : best;
          }
        }
        if (logits_view) iree_hal_buffer_view_release(logits_view);
        iree_vm_list_release(head_outputs);
        g_t_d2h += now_s() - t3;
      } else {
        // Host lm-head over the embedding table (or legacy weight).
        double t2 = now_s();
        status = iree_hal_device_transfer_d2h(
            device, iree_hal_buffer_view_buffer(hidden_view), 0, hidden.data(),
            (iree_device_size_t)g_cfg.hidden_dim * sizeof(uint16_t),
            IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
        g_t_d2h += now_s() - t2;
        if (g_t_enabled && first_step && iree_status_is_ok(status)) {
          uint64_t hsum = 1469598103934665603ULL;
          for (int d = 0; d < g_cfg.hidden_dim; ++d)
            hsum = (hsum ^ hidden[d]) * 1099511628211ULL;
          fprintf(stderr, "step0 hidden[0..3]=%u,%u,%u,%u fnv=%016llx\n",
                  hidden[0], hidden[1], hidden[2], hidden[3],
                  (unsigned long long)hsum);
        }
        t2 = now_s();
        if (iree_status_is_ok(status)) {
          for (int d = 0; d < g_cfg.hidden_dim; ++d) hf[d] = bf16_to_f32(hidden[d]);
          int32_t next_token = 0;
          float best = -INFINITY, second = -INFINITY;
          int32_t second_i = -1;
          if (g_has_embeddings) {
            // logits[v] = dot(hidden, emb_row[v]); optionally only over LUT.
            const uint16_t* table = (const uint16_t*)g_embeddings.data;
            if (g_has_lut) {
              int64_t n = (int64_t)g_lut_host.size();
              for (int64_t i = 0; i < n; ++i) {
                float f = emb_row_dot(table + (size_t)g_lut_host[i] * g_cfg.hidden_dim,
                                      hf.data(), g_cfg.hidden_dim);
                if (f > best) {
                  second = best; second_i = next_token;
                  best = f; next_token = (int32_t)g_lut_host[i];
                } else if (f > second) {
                  second = f; second_i = (int32_t)g_lut_host[i];
                }
              }
            } else {
              for (int v = 0; v < g_cfg.vocab_size; ++v) {
                float f = emb_row_dot(table + (size_t)v * g_cfg.hidden_dim, hf.data(),
                                      g_cfg.hidden_dim);
                if (f > best) {
                  second = best; second_i = next_token;
                  best = f; next_token = v;
                } else if (f > second) {
                  second = f; second_i = v;
                }
              }
            }
          } else {
            // Legacy [hidden, vocab] column table, tiled for cache.
            logits.assign(g_cfg.vocab_size, 0.f);
            const int VBLK = 16384;
            for (int vb = 0; vb < g_cfg.vocab_size; vb += VBLK) {
              int n = (vb + VBLK <= g_cfg.vocab_size) ? VBLK : (g_cfg.vocab_size - vb);
              for (int d = 0; d < g_cfg.hidden_dim; ++d) {
                const uint16_t* wrow = g_lmhead_w + (size_t)d * g_cfg.vocab_size + vb;
#if defined(__aarch64__)
                lm_head_row_neon(wrow, hf[d], logits.data() + vb, n);
#else
                float h = hf[d];
                for (int i = 0; i < n; ++i) logits[vb + i] += h * bf16_to_f32(wrow[i]);
#endif
              }
            }
            if (g_has_lut) {
              for (size_t i = 0; i < g_lut_host.size(); ++i) {
                int64_t v = g_lut_host[i];
                if (logits[v] > best) {
                  second = best; second_i = next_token;
                  best = logits[v]; next_token = (int32_t)v;
                } else if (logits[v] > second) {
                  second = logits[v]; second_i = (int32_t)v;
                }
              }
            } else {
              for (int v = 0; v < g_cfg.vocab_size; ++v) {
                if (logits[v] > best) {
                  second = best; second_i = next_token;
                  best = logits[v]; next_token = v;
                } else if (logits[v] > second) {
                  second = logits[v]; second_i = v;
                }
              }
            }
          }
          if (g_t_enabled && first_step) {
            fprintf(stderr, "step0 top: %d=%.5f, 2nd %d=%.5f, gap=%.6f\n",
                    next_token, best, second_i, second, best - second);
          }
          token_id = next_token;
        }
        g_t_lmhead += now_s() - t2;
      }
    }

    if (hidden_view) iree_hal_buffer_view_release(hidden_view);

    // Fallback path: feed present (outputs[1..N]) back as the next past.
    if (!use_tied && iree_status_is_ok(status)) {
      double t3 = now_s();
      iree_vm_list_t* next_past = NULL;
      status = iree_vm_list_create(iree_vm_make_undefined_type_def(), n_layer,
                                   host_allocator, &next_past);
      for (iree_host_size_t i = 1; i <= (iree_host_size_t)n_layer &&
                                  iree_status_is_ok(status);
           ++i) {
        iree_vm_ref_t ref = iree_vm_ref_null();
        status = iree_vm_list_get_ref_retain(outputs, i, &ref);
        if (iree_status_is_ok(status)) status = iree_vm_list_push_ref_retain(next_past, &ref);
        iree_vm_ref_release(&ref);
      }
      if (iree_status_is_ok(status)) {
        if (past_views) iree_vm_list_release(past_views);
        past_views = next_past;
      } else {
        iree_vm_list_release(next_past);
      }
      g_t_feedback += now_s() - t3;
    }
    iree_vm_list_release(outputs);
    first_step = false;

    double step_ms = (now_s() - t_step0) * 1e3;
    if (is_prefill) {
      fprintf(stdout, "prefill %d/%d: token %d (%.1f ms)\n", step + 1, n_prompt,
              token_id, step_ms);
      token_id = prompt[step + 1];
    } else {
      ++gen_count;
      decode_time += now_s() - t_step0;
      ++decode_steps;
      fprintf(stdout, "step %d: token %d (%.1f ms)\n", gen_count, token_id, step_ms);
      if (config_is_eos(token_id)) {
        eos_hit = true;
        fprintf(stdout, "eos token %d, stopping\n", token_id);
      }
    }
    fflush(stdout);
  }

  if (decode_steps > 0) {
    fprintf(stdout, "decode: %d tokens in %.3fs = %.2f tok/s (%.1f ms/token)\n",
            decode_steps, decode_time, decode_steps / decode_time,
            decode_time * 1e3 / decode_steps);
  }

  if (g_t_enabled) {
    double tot = now_s() - t_loop0;
    fprintf(stderr,
            "TORQ_DECODE_TIMING: total=%.3fs | embed=%.3f h2d=%.3f inputs=%.3f invoke=%.3f d2h=%.3f lmhead_npu=%.3f lmhead=%.3f feedback=%.3f (steps=%d)\n",
            tot, g_t_embed, g_t_h2d, g_t_inputs, g_t_invoke, g_t_d2h,
            g_t_lmhead_npu, g_t_lmhead, g_t_feedback, decode_steps + n_prompt);
  }

  tied_state_release(&tied);
  if (past_views) iree_vm_list_release(past_views);
  if (head_inputs) iree_vm_list_release(head_inputs);
  if (head_input_view) iree_hal_buffer_view_release(head_input_view);
  free(g_lmhead_w);
  if (g_has_embeddings) npy_unmap(&g_embeddings);
  if (g_has_lut) npy_unmap(&g_lut);
  if (head_device_allocator) iree_hal_allocator_release(head_device_allocator);
  if (head_device) iree_hal_device_release(head_device);
  if (head_context) iree_vm_context_release(head_context);
  iree_hal_allocator_release(device_allocator);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);

  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "decode failed: ");
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
