// clang-format off
// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  12/01/2024, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#if defined(_WIN32)
#include <direct.h>
#include <io.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

#ifdef xerr
#undef xerr
#endif
#define xerr(...) printf(__VA_ARGS__);
#ifdef xdbg
#undef xdbg
#endif
#ifdef TORQ_CT_DEBUG
#define assert(c_) do {if (!(c_)) { printf("ERROR: %s : %d : %s: Assertion `" #c_ "' failed.\n", __FILE__, __LINE__,__func__); exit(1); } } while (0)
#define xdbg(...) printf(__VA_ARGS__);
#else
#include <assert.h>
#define xdbg(...) (0L)
#endif

#include "torq_desc_dump.h"

//NOTE: we use multichar constants intentionally
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wmultichar"
#endif

#define MAX_PATH_LEN 256
#define DUMP_PREFIX "tv."

typedef struct torq_desc_dump_t {
    char path[MAX_PATH_LEN+64];
    char *dirname;
    char *fname;
    FILE *fp_mlst;
    int  job_id;
    int  stg_id;
    int  slc_id;
    int  tsk_id;
    int  info_cnt, data_cnt;
    int  xw_cnt, xr_cnt, lw_cnt, lr_cnt;
    char new_job;
    char new_stg;
    char new_pkg;
} torq_desc_dump_t;

static char *cc2str(uint32_t cc, char *s)
{
    int i, j;
    char c;
    for (i=j=0; i<sizeof(cc); i++) {
        c = ((char *)&cc)[sizeof(cc)-1-i];
        if (c) s[j++] = c;
    }
    s[j] = 0;
    return s;
}

static char *cc2lstr(uint32_t cc, char *s)
{
    int i, j;
    char c;
    for (i=j=0; i<sizeof(cc); i++) {
        c = ((char *)&cc)[sizeof(cc)-1-i];
        if (c) s[j++] = tolower(c);
    }
    s[j] = 0;
    return s;
}

#if 0
static char *cc2ustr(uint32_t cc, char *s)
{
    int i, j;
    char c;
    for (i=j=0; i<sizeof(cc); i++) {
        c = ((char *)&cc)[sizeof(cc)-1-i];
        if (c) s[j++] = toupper(c);
    }
    s[j] = 0;
    return s;
}
#endif

static int make_dir(const char *dir)
{
    int r = 0;
    if (access(dir, 0) == -1) {
#if defined(_WIN32)
        r = _mkdir(dir);
#else
        r = mkdir(dir, S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH);
#endif
        assert(r==0);
    }
    return r;
}

static void open_mlst(torq_desc_dump_t *self, const char *name)
{
    char *p;
    if (!self->fname) return;
    assert(!self->fp_mlst);
    p = self->fname;
    p += sprintf(p, DUMP_PREFIX"%s", name);
    p += sprintf(p, ".mem.lst");
    self->fp_mlst = fopen(self->path, "w"); assert(self->fp_mlst);
}

static void close_mlst(torq_desc_dump_t *self)
{
    if (self->fp_mlst) {
        fclose(self->fp_mlst);
        self->fp_mlst = 0L;
    }
}

static void dump_u32(torq_desc_dump_t *self, const char *name, uint32_t data)
{
    char *p;
    FILE *fp;
    if (!self->fname) return;
    p = self->fname;
    p += sprintf(p, DUMP_PREFIX"%s", name);
    p += sprintf(p, ".txt");
    fp = fopen(self->path, "w"); assert(fp);
    fprintf(fp, "0x%08x\n", data);
    fclose(fp);
}

static void dump_fname(torq_desc_dump_t *self, const char *group, const char *name, int cnt)
{
    char *p;
    if (!self->fname) return;
    p = self->fname;
    p += sprintf(p, DUMP_PREFIX"%s", group);
    if (self->slc_id < 0) p += sprintf(p, ".nss");
    else p += sprintf(p, ".slc%d", self->slc_id);
    if (self->tsk_id >= 0) p += sprintf(p, ".task%d", self->tsk_id);
    p += sprintf(p, ".%s", name);
    if (cnt >= 0) p += sprintf(p, "%d", cnt);
    p += sprintf(p, ".txt");
}

static void dump_header(torq_desc_dump_t *self, uint32_t mem_tag, uint32_t op_tag, const char *name, int cnt, int int_type, uint32_t addr, uint32_t size)
{
    char *op = op_tag=='INP'?(mem_tag=='XRAM'?"xload":"load "):(mem_tag=='XRAM'?"xsave":"save ");
    if (self->fp_mlst) fprintf(self->fp_mlst, "%s  0x%08x %10d %3d  hex  %s\n", op, addr, size, int_type, self->fname);
}

static void dump_payload(torq_desc_dump_t *self, int int_type, uint32_t size, void *data)
{
    int i;
    FILE *fp;
    assert(int_type==1||int_type==2||int_type==4);
    assert((size&(int_type-1))==0);
    size /= int_type;
    if (data) {
        fp = fopen(self->path, "w"); assert(fp);
        if      (int_type == 4) for (i=0; i<size; i++) fprintf(fp, "%08x\n", ((uint32_t *)data)[i]);
        else if (int_type == 2) for (i=0; i<size; i++) fprintf(fp, "%04x\n", ((uint16_t *)data)[i]);
        else                    for (i=0; i<size; i++) fprintf(fp, "%02x\n", ((uint8_t *)data)[i]);
        fclose(fp);
   }
}

static void dump_data(torq_desc_dump_t *self, uint32_t mem_tag, uint32_t op_tag, const char *name, int cnt, int int_type, uint32_t addr, uint32_t size, void *data)
{
    char s[5];
    if (!self->fname) return;
    dump_fname(self, cc2lstr(op_tag, s), name, cnt);
    dump_header(self, mem_tag, op_tag, name, cnt, int_type, addr, size);
    dump_payload(self, int_type, size, data);
}

static void stg_begin(torq_desc_dump_t *self)
{
    if (self->dirname) {
        self->fname = self->dirname + sprintf(self->dirname, "job%d/", self->job_id);
        make_dir(self->path); 
        open_mlst(self, self->stg_id==0?"init":"exit");
    }
}

static void stg_end(torq_desc_dump_t *self)
{
    if (self->dirname) {
        close_mlst(self);
    }
}

static void update_ids(torq_desc_dump_t *self, int job_id, int stg_id, int tsk_id, int slc_id)
{
    assert(job_id>= 0);
    assert(stg_id>= 0);
    assert(tsk_id>=-1);
    assert(slc_id>=-1);
    self->new_job = self->job_id != job_id;
    self->new_stg = self->stg_id != stg_id || self->new_job;
    self->new_pkg = self->tsk_id != tsk_id || self->slc_id != slc_id || self->new_stg;
    if (self->new_pkg) {
        self->info_cnt = self->data_cnt = 0;
    }
    if (self->new_job) {
        self->xw_cnt = self->xr_cnt = self->lw_cnt = self->lr_cnt = 0;
    }
    if (self->new_stg && self->stg_id >= 0) stg_end(self);
    self->job_id = job_id;
    self->stg_id = stg_id;
    self->tsk_id = tsk_id;
    self->slc_id = slc_id;
    if (self->new_stg) stg_begin(self);
}

int torq_desc_dump__ndl_info(void *self_, int job_id, int tsk_id, int slc_id, uint32_t tag, int set_id, torq_ndl_cmd_t *cmd)
{
    torq_desc_dump_t *self = (torq_desc_dump_t *)self_;
    int i, nd;
    uint64_t n=1, n0=1;
    int32_t ns=1, nsnz=1;
    update_ids(self, job_id, 0, tsk_id, slc_id); // info is always processed at stage 0
    nd = cmd->nld + cmd->nhd;
    if (self->fname) {
        FILE *fp;
        char s[5];
        char *p = self->fname;
        p += sprintf(p, DUMP_PREFIX);
        if (slc_id < 0) p += sprintf(p, "nss");
        else p += sprintf(p, "slc%d", slc_id);
        if (tsk_id >= 0) p += sprintf(p, ".task%d", tsk_id);
        p += sprintf(p, ".info.txt");
        fp = fopen(self->path, self->info_cnt?"a":"w"); assert(fp);
        fprintf(fp, "%s[%d]=", cc2str(tag, s), set_id);
        if (cmd->nld) fprintf(fp, "{");
        for (i=0; i<nd; i++) {
            if (cmd->n[i] == ~0 || cmd->n[i] == 0) {
                fprintf(fp, " %c*", cmd->t[i]);
                n = 0;
            } else {
                fprintf(fp, " %c%d", cmd->t[i], cmd->n[i]);
                n *= cmd->n[i];
            }
            if (cmd->p[i]) {
                if (cmd->pmode[i]) fprintf(fp, "[p%c%d]", tolower(cmd->pmode[i]), cmd->p[i]);
                else fprintf(fp, "[p%d]", cmd->p[i]);
            }
            if (cmd->n[i]>1) {
                if (cmd->s[i] != ns) {
                    if (cmd->s[i] == nsnz) fprintf(fp, ":");
                    else fprintf(fp, ":%d", cmd->s[i]);
                }
                ns = cmd->n[i]*cmd->s[i]; // last n*s
                if (cmd->s[i] != 0) nsnz = ns; // last nonzero n*s
            }
            if (cmd->nld && i==cmd->nld-1) { n0 = n; n = 1; fprintf(fp, " }"); }
        }
        if (cmd->nsd) {
            fprintf(fp, ",");
            ns=1; nsnz=1;
            for (; i<nd+cmd->nsd; i++) {
                fprintf(fp, " %c%d", cmd->t[i], cmd->n[i]);
                if (cmd->p[i]) {
                    if (cmd->pmode[i]) fprintf(fp, "[p%c%d]", tolower(cmd->pmode[i]), cmd->p[i]);
                    else fprintf(fp, "[p%d]", cmd->p[i]);
                }
                if (cmd->n[i]>1) {
                    if (cmd->s[i] != ns) {
                        if (cmd->s[i] == nsnz) fprintf(fp, ":");
                        else fprintf(fp, ":%d", cmd->s[i]);
                    }
                    ns = cmd->n[i]*cmd->s[i]; // last n*s
                    if (cmd->s[i] != 0) nsnz = ns; // last nonzero n*s
                }
            }
        }
        if (cmd->base_addr) fprintf(fp, " @0x%08x", cmd->base_addr);
        fprintf(fp, ", total={%llu}*%llu\n", n0, n);
        fclose(fp);
    }
    self->info_cnt++;
    return n ? 1 : 0;
}

int torq_desc_dump__data(void *self_, int job_id, int stg_id, int tsk_id, int slc_id, uint32_t mem_tag, uint32_t op_tag, uint32_t data_tag, uint32_t addr_tag, uint32_t addr, uint32_t size, void *data)
{
    torq_desc_dump_t *self = (torq_desc_dump_t *)self_;
    char s[5];
    xdbg("torq_desc_dump__data(job=%d, stg=%d, tsk=%d, slc=%d, ", job_id, stg_id, tsk_id, slc_id);
    xdbg("mem_tag=%s, ", cc2str(mem_tag, s));
    xdbg("op_tag=%s, ", cc2str(op_tag, s));
    xdbg("data_tag=%s, ", cc2str(data_tag, s));
    xdbg("addr_tag=%s, ", cc2str(addr_tag, s));
    xdbg("addr=0x%08x, size=%d\n", addr, size);
    update_ids(self, job_id, stg_id, tsk_id, slc_id);
    if (addr_tag == 'STRT') dump_u32(self, "cdesc_addr", addr);
    switch (mem_tag) {
    case 'XRAM':
        switch (op_tag) {
        case 'INP':
            switch (data_tag) {
            case 'CDS':  dump_data(self, mem_tag, op_tag, "cdesc",      -1,              4, addr, size, data); break;
            case 'DDS':  dump_data(self, mem_tag, op_tag, "ddesc",      -1,              4, addr, size, data); break;
            case 'D0DS': dump_data(self, mem_tag, op_tag, "ddesc0",     -1,              4, addr, size, data); break;
            case 'D1DS': dump_data(self, mem_tag, op_tag, "ddesc1",     -1,              4, addr, size, data); break;
            case 'WDS':  dump_data(self, mem_tag, op_tag, "wdesc",      -1,              4, addr, size, data); break;
            case 'BDS':  dump_data(self, mem_tag, op_tag, "bdesc",      -1,              4, addr, size, data); break;
            case 'B0DS': dump_data(self, mem_tag, op_tag, "bdesc0",     -1,              4, addr, size, data); break;
            case 'B1DS': dump_data(self, mem_tag, op_tag, "bdesc1",     -1,              4, addr, size, data); break;
            case 'QDS':  dump_data(self, mem_tag, op_tag, "qdesc",      -1,              4, addr, size, data); break;
            case 'D':    dump_data(self, mem_tag, op_tag, "xram_dmem_",  self->xw_cnt++, 1, addr, size, data); break;
            case 'D0':   dump_data(self, mem_tag, op_tag, "xram_dmem0_", self->xw_cnt++, 1, addr, size, data); break;
            case 'D1':   dump_data(self, mem_tag, op_tag, "xram_dmem1_", self->xw_cnt++, 1, addr, size, data); break;
            case 'W':    dump_data(self, mem_tag, op_tag, "xram_wmem_",  self->xw_cnt++, 1, addr, size, data); break;
            case 'B':    dump_data(self, mem_tag, op_tag, "xram_bmem_",  self->xw_cnt++, 1, addr, size, data); break;
            case 'B0':   dump_data(self, mem_tag, op_tag, "xram_bmem0_", self->xw_cnt++, 1, addr, size, data); break;
            case 'B1':   dump_data(self, mem_tag, op_tag, "xram_bmem1_", self->xw_cnt++, 1, addr, size, data); break;
            default:     dump_data(self, mem_tag, op_tag, "xram_",       self->xw_cnt++, 1, addr, size, data); break;
            }
            break;
        case 'OUT':
            switch (data_tag) {
            case 'Q':    dump_data(self, mem_tag, op_tag, "xram_qmem_",  self->xr_cnt++, 1, addr, size, data); break;
            default:     dump_data(self, mem_tag, op_tag, "xram_",       self->xr_cnt++, 1, addr, size, data); break;
            }
            break;
        default:
            xerr("WARNING: dump op_tag ignored: '%s'\n", cc2str(op_tag, s));
        }
        break;
    case 'LRAM':
        switch (op_tag) {
        case 'INP':
            switch (data_tag) {
            case 'CDS':  dump_data(self, mem_tag, op_tag, "cdesc",      -1,              4, addr, size, data); break;
            case 'DDS':  dump_data(self, mem_tag, op_tag, "ddesc",      -1,              4, addr, size, data); break;
            case 'D0DS': dump_data(self, mem_tag, op_tag, "ddesc0",     -1,              4, addr, size, data); break;
            case 'D1DS': dump_data(self, mem_tag, op_tag, "ddesc1",     -1,              4, addr, size, data); break;
            case 'WDS':  dump_data(self, mem_tag, op_tag, "wdesc",      -1,              4, addr, size, data); break;
            case 'BDS':  dump_data(self, mem_tag, op_tag, "bdesc",      -1,              4, addr, size, data); break;
            case 'B0DS': dump_data(self, mem_tag, op_tag, "bdesc0",     -1,              4, addr, size, data); break;
            case 'B1DS': dump_data(self, mem_tag, op_tag, "bdesc1",     -1,              4, addr, size, data); break;
            case 'QDS':  dump_data(self, mem_tag, op_tag, "qdesc",      -1,              4, addr, size, data); break;
            case 'D':    dump_data(self, mem_tag, op_tag, "lram_dmem_",  self->lw_cnt++, 1, addr, size, data); break;
            case 'D0':   dump_data(self, mem_tag, op_tag, "lram_dmem0_", self->lw_cnt++, 1, addr, size, data); break;
            case 'D1':   dump_data(self, mem_tag, op_tag, "lram_dmem1_", self->lw_cnt++, 1, addr, size, data); break;
            case 'W':    dump_data(self, mem_tag, op_tag, "lram_wmem_",  self->lw_cnt++, 1, addr, size, data); break;
            case 'B':    dump_data(self, mem_tag, op_tag, "lram_bmem_",  self->lw_cnt++, 1, addr, size, data); break;
            case 'B0':   dump_data(self, mem_tag, op_tag, "lram_bmem0_", self->lw_cnt++, 1, addr, size, data); break;
            case 'B1':   dump_data(self, mem_tag, op_tag, "lram_bmem1_", self->lw_cnt++, 1, addr, size, data); break;
            default:     dump_data(self, mem_tag, op_tag, "lram_",       self->lw_cnt++, 1, addr, size, data); break;
            }
            break;
        case 'OUT':
            switch (data_tag) {
            case 'Q':    dump_data(self, mem_tag, op_tag, "lram_qmem_",  self->lr_cnt++, 1, addr, size, data); break;
            default:     dump_data(self, mem_tag, op_tag, "lram_",       self->lr_cnt++, 1, addr, size, data); break;
            }
            break;
        default:
            xerr("WARNING: invalid op_tag: '%s'\n", cc2str(op_tag, s));
        }
        break;
    default:
        xerr("WARNING: invalid mem_tag: '%s'\n", cc2str(mem_tag, s));
    }
    self->data_cnt++;
    return 0;
}

void *torq_desc_dump__open(const char *path)
{
    torq_desc_dump_t *self;
    assert(strnlen(path, MAX_PATH_LEN)<MAX_PATH_LEN);
    self = (torq_desc_dump_t *)calloc(sizeof(torq_desc_dump_t), 1); assert(self);
    if (path && path[0]) {
      make_dir(path); 
      self->dirname = self->path + sprintf(self->path, "%s/", path);
    }
    self->job_id = self->stg_id = self->tsk_id = self->slc_id = INT32_MIN;
    return (void *)self;
}

int torq_desc_dump__close(void *self_)
{
    torq_desc_dump_t *self = (torq_desc_dump_t *)self_;
    assert(self);
    if (self->stg_id >= 0) stg_end(self);
    free(self);
    return 0;
}

