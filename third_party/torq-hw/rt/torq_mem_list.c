// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  06/28/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#include <stdint.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "torq_log.h"
#include "torq_sys.h"
#include "torq_xram_access.h"
#include "torq_lram_access.h"
#include "torq_mem_list.h"

#define MAX_PATH_LEN 1024

typedef struct lst_ctx_t {
    torq_sys_t *sys;
    char inp_path[MAX_PATH_LEN];
    char out_path[MAX_PATH_LEN];
    char *inp_fname;
    char *out_fname;
    int inp_max_len;
    int out_max_len;
} lst_ctx_t;

typedef int (*mem_lst_handler_t)(lst_ctx_t *ctx, const char *cmd, uint32_t addr, uint32_t size, uint32_t word_width, const char *fmt, const char *fname);

static int _mem_lst_handler(lst_ctx_t *ctx, const char *cmd, uint32_t addr, uint32_t size, uint32_t word_width, const char *fmt, const char *fname)
{
    int r;
    xdbg("MEM_LST: cmd=%s addr=0x%08x size=%d word_width=%d fmt=%s fname=%s\n", cmd, addr, size, word_width, fmt, fname);
    xdie_v(!(!strcmp(fmt, "hex") || !strcmp(fmt, "raw")), "unsuppported mem file format: %s\n", fmt);
    xdie_v(size%word_width, "size must be word aligned\n");
    if (!strcmp(cmd, "load")) {
        snprintf(ctx->inp_fname, ctx->inp_max_len, "%s", fname); 
        r = torq_lram_load_from_file(ctx->sys, addr, word_width, size/word_width, fmt, ctx->inp_path); xdie(r<0);
    } else if (!strcmp(cmd, "save")) {
        snprintf(ctx->out_fname, ctx->out_max_len, "%s", fname); 
        r = torq_lram_save_to_file(ctx->sys, addr, word_width, size/word_width, fmt, ctx->out_path); xdie(r<0);
    } else if (!strcmp(cmd, "xload")) {
        snprintf(ctx->inp_fname, ctx->inp_max_len, "%s", fname); 
        r = torq_xram_load_from_file(ctx->sys, addr, word_width, size/word_width, fmt, ctx->inp_path); xdie(r<0);
    } else if (!strcmp(cmd, "xsave")) {
        snprintf(ctx->out_fname, ctx->out_max_len, "%s", fname); 
        r = torq_xram_save_to_file(ctx->sys, addr, word_width, size/word_width, fmt, ctx->out_path); xdie(r<0);
    } else xdie_v(1, "invalid mem_lst cmd: %s\n", cmd);
    return 0;
}

static int _proc_mem_lst(lst_ctx_t *ctx, const char *lst_fname, mem_lst_handler_t hdlr)
{
    FILE *fp = 0L;
    char line[MAX_PATH_LEN+256], is_spc=1;
    char *p = 0L;
    char *s[6] = {0L, 0L, 0L, 0L, 0L, 0L};
    int r, si = 0, lcnt = 0;

    fp = fopen(lst_fname, "r");
    xdie_v(!fp, "unable to open file: %s\n", lst_fname);
    while (1) {
        p = fgets(line, sizeof(line), fp);
        if (!p) break;
        lcnt++;
        while (1) {
            if (!*p || *p == '#' || *p == '\n' || *p == '\r') {
                *p = 0;
                if (si == sizeof(s)/sizeof(s[0])) {
                    r = (*hdlr)(ctx, s[0], strtoul(s[1],0L,0), strtoul(s[2],0L,0), strtoul(s[3],0L,0), s[4], s[5]);
                    if (r<0) return r;
                }
                else xdie_v(si, "wrong format in line %d of '%s'\n", lcnt, lst_fname);
                si = 0;
                is_spc = 1;
                break;
            }
            if (isspace(*p)) { is_spc = 1; *p = 0; }
            else { if (is_spc) { if (si<sizeof(s)/sizeof(s[0])) s[si] = p; si++; is_spc = 0; } }
            p++;
        }
    }
    fclose(fp);
    return 0;
}

int torq_proc_mem_lst(torq_sys_t *sys, const char *lst_fname, const char *inp_dir, const char *out_dir)
{
    int r;
    lst_ctx_t ctx_buf, *ctx = &ctx_buf;
    ctx->sys = sys;
    r = snprintf(ctx->inp_path, MAX_PATH_LEN-1, "%s/", inp_dir);
    ctx->inp_fname = ctx->inp_path + r;
    ctx->inp_max_len = MAX_PATH_LEN-1 - r;
    r = snprintf(ctx->out_path, MAX_PATH_LEN-1, "%s/", out_dir);
    ctx->out_fname = ctx->out_path + r;
    ctx->out_max_len = MAX_PATH_LEN-1 - r;
    return _proc_mem_lst(ctx, lst_fname, _mem_lst_handler);
}
