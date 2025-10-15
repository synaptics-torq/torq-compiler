// clang-format off
//---------------------------------------------------------------------------
//  Copyright 2023-2024 Synaptics Inc
//---------------------------------------------------------------------------
//! \file
//! \brief      High Level Torq C Model API
//! \author     Hongjie Guan
//! \date       01/09/2024 - 01/09/2024
//---------------------------------------------------------------------------

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#if defined(_WIN32)
#include <direct.h>
#include <io.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

#ifdef xdbg
#undef xdbg
#endif
#ifdef TORQ_CT_DEBUG
#define assert(c_) do {if (!(c_)) { printf("ERROR: %s : %d : %s: Assertion `" #c_ "' failed.\n", __FILE__, __LINE__,__func__); exit(1); } } while (0)
#define xdbg(...) printf(__VA_ARGS__);
#define xdbg_func_begin() printf("+++ %s()\n", __func__);
#define xdbg_func_end() printf("--- %s()\n", __func__);
#else
#include <assert.h>
#define xdbg(...) (0L)
#define xdbg_func_begin() (0L)
#define xdbg_func_end() (0L)
#endif

#ifdef TORQ_API_VER
#undef TORQ_API_VER
#endif
#define TORQ_API_VER 0x88888888 //always the newest version

#include "torq_desc_dump.h"
#include "torq_api.h"
#include "torq_nss_regs_struct.h"

#ifdef torq_cfg_begin
#undef torq_cfg_begin
#endif
#ifdef torq_task_cfg_begin
#undef torq_task_cfg_begin
#endif
#ifdef torq_ndl_desc_write
#undef torq_ndl_desc_write
#endif
#ifdef torq_data_write
#undef torq_data_write
#endif
#ifdef torq_data_read
#undef torq_data_read
#endif

#ifndef MAX
#define	MAX(a, b) ((a)>(b)?(a):(b))
#endif

#ifdef DIAG_ACT_COVER
#include "dv_diag_ct_hooks.h"
#else
#define DV_DIAG_HOOK__SLC_CFG_DESC_GEN() do {} while (0)
#endif

//NOTE: we use multichar constants intentionally
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wmultichar"
#endif

#define CFG_DESC_DEFAULT_ADDR     0
#define CFG_DESC_MAX_LEN_PER_TASK 512+64
#define NDL_DESC_MAX_LEN_PER_TASK 64
#define NUM_FULL_NDL_DESC         6

// local defines, may be updated, repurposed, or removed
#define TORQ_DBUS_WIDTH 72
#define TORQ_WBUS_WIDTH 32
#define TORQ_BBUS_WIDTH 32
#define TORQ_PBUS_WIDTH 64
#define TORQ_QBUS_WIDTH 64


//Naming convention in this file:
//  id:   int, arbitrary range, arbitrary order
//  idx:  int, zero-based, fixed range, arbitrary order
//  cnt:  int, zero-based, ascending order
//  tag:  uint32_t, multi-character code (1..4 characters)

typedef struct torq_wrap_t {
    uint32_t cfg_desc_addr;
    uint32_t cfg_desc_xaddr;
    uint32_t cfg_desc_len;
    uint32_t cfg_desc_buf[CFG_DESC_MAX_LEN_PER_TASK];
    uint32_t cfg_desc_total_len;
    uint32_t *cfg_desc_last_hdr;
    uint32_t ndl_desc_addr[NUM_FULL_NDL_DESC];
    uint32_t ndl_desc_xaddr[NUM_FULL_NDL_DESC];
    uint32_t ndl_desc_len[NUM_FULL_NDL_DESC];
    uint32_t ndl_desc_buf[NUM_FULL_NDL_DESC][NDL_DESC_MAX_LEN_PER_TASK];
    uint32_t *ndl_desc_last_hdr[NUM_FULL_NDL_DESC];
    uint32_t xn, yn;
    torq_cfg_t cfg;
    torq_nss_regs_t regs;
    uint32_t acbr_ldim_sz, debr_ldim_sz; //for assertion check only
    uint32_t deqw_bn;

    uint32_t cfg_wr_a;
    int cfg_wr_n;
    int cfg_wr_i;

    int tsk_id[TORQ_SLICES+1];
    int job_id;
    int stg_id;
    int slc_id;
    char new_job;
    char entry_point;
    char cdma_in_use;
    char cdma_release;
    void *dump_ctx;
    torq_bitstream_segment_t *bitstream;
} torq_wrap_t;


static int _get_task_id(torq_wrap_t *wrap)
{
    int slc_uidx;
    assert(wrap->slc_id>=-1 && wrap->slc_id<TORQ_SLICES);
    slc_uidx = wrap->slc_id<0 ? TORQ_SLICES : wrap->slc_id;
    return wrap->tsk_id[slc_uidx];
}

static void _inc_task_id(torq_wrap_t *wrap)
{
    int slc_uidx;
    assert(wrap->slc_id>=-1 && wrap->slc_id<TORQ_SLICES);
    slc_uidx = wrap->slc_id<0 ? TORQ_SLICES : wrap->slc_id;
    wrap->tsk_id[slc_uidx]++;
}

void *torq_open(const char *path)
{
    torq_wrap_t *wrap;
    int i, r;
    xdbg_func_begin();
    wrap = (torq_wrap_t *)calloc(sizeof(torq_wrap_t), 1); assert(wrap);
    wrap->cfg_desc_addr = ~0;
    wrap->new_job = 1;
    wrap->job_id = -1;
    wrap->stg_id = -1;
    memset(wrap->tsk_id, -1, sizeof(wrap->tsk_id));
    wrap->dump_ctx = torq_desc_dump__open(path); assert(wrap->dump_ctx);
    xdbg_func_end();
    return (void *)wrap;
}

int torq_close(void *self)
{
    torq_wrap_t *wrap = (torq_wrap_t *)self;
    xdbg_func_begin();
    if (wrap->dump_ctx) torq_desc_dump__close(wrap->dump_ctx);
    free(wrap);
    xdbg_func_end();
    return 0;
}

int torq_job_cfg_begin(void *self)
{
    torq_wrap_t *wrap = (torq_wrap_t *)self;
    xdbg_func_begin();
    wrap->new_job = 0;
    wrap->job_id++;
    wrap->stg_id = 0;
    wrap->entry_point = -1; //not set
    xdbg_func_end();
    return 0;
}

int torq_job_cfg_end(void *self)
{
    torq_wrap_t *wrap = (torq_wrap_t *)self;
    xdbg_func_begin();
    wrap->stg_id = 1;
    memset(wrap->tsk_id, -1, sizeof(wrap->tsk_id)); // no task ids after cfg
    xdbg_func_end();
    return 0;
}

static int _add_bitstream_segment(torq_wrap_t *wrap, uint32_t lram_addr, uint32_t xram_addr, uint32_t size, void *data)
{
    torq_bitstream_segment_t *seg;

    seg = (torq_bitstream_segment_t *) malloc(sizeof(torq_bitstream_segment_t));

    if (!seg) {
        return 1;
    }

    seg->size = size;

    seg->lram_addr = lram_addr;
    seg->xram_addr = xram_addr;

    seg->data = (uint8_t *) malloc(size);

    if (!seg->data) {
        free(seg);
        return 1;
    }

    seg->next = wrap->bitstream;
    wrap->bitstream = seg;

    memcpy(seg->data, data, size);

    return 0;
}

static int _output_desc(torq_wrap_t *wrap, int last, uint32_t nxt_laddr)
{
    int i, j, task_id;
    uint32_t tag[6] = { 'DDS', 'WDS', 'BDS', 'QDS', 'D1DS', 'B1DS' };
    task_id = _get_task_id(wrap);
    if (wrap->cfg_desc_len) {
        if (last) {
            *(wrap->cfg_desc_last_hdr) |= (1u<<31);
        }
        else if (nxt_laddr != TORQ_LADDR_APPEND && nxt_laddr != TORQ_LADDR_NONE) { // link mode
            assert(nxt_laddr<(1<<24));
            nxt_laddr |= 0x70000000; // nxt
            wrap->cfg_desc_buf[wrap->cfg_desc_len] = nxt_laddr;
            wrap->cfg_desc_len++;
        }
        _add_bitstream_segment(wrap, wrap->cfg_desc_addr, wrap->cfg_desc_xaddr, wrap->cfg_desc_len<<2, wrap->cfg_desc_buf);

        if (wrap->cfg_desc_xaddr == TORQ_XADDR_NONE) {
            if (wrap->cfg_desc_addr != TORQ_LADDR_NONE) { // not dry-run
                if (wrap->dump_ctx)
                    torq_desc_dump__data(wrap->dump_ctx,
                        wrap->job_id, wrap->stg_id, task_id, wrap->slc_id,
                        'LRAM', 'INP', 'CDS', wrap->entry_point==1?'STRT':0,
                        wrap->cfg_desc_addr, wrap->cfg_desc_len<<2, wrap->cfg_desc_buf);
            }
            wrap->cfg_desc_addr += (wrap->cfg_desc_len<<2);
        }
        else {
            if (wrap->cfg_desc_addr != TORQ_LADDR_NONE) { // not dry-run
                if (wrap->dump_ctx)
                    torq_desc_dump__data(wrap->dump_ctx,
                        wrap->job_id, wrap->stg_id, task_id, wrap->slc_id,
                        'XRAM', 'INP', 'CDS', 0,
                        wrap->cfg_desc_xaddr, wrap->cfg_desc_len<<2, wrap->cfg_desc_buf);
            }
            wrap->cfg_desc_xaddr += (wrap->cfg_desc_len<<2);
        }
        wrap->cfg_desc_len = 0;
    }
    for (i=0; i<NUM_FULL_NDL_DESC; i++) {
        if (wrap->ndl_desc_len[i]) {
            assert(wrap->slc_id>=0);
            *(wrap->ndl_desc_last_hdr[i]) |= ((1u<<31) | (2<<28));

            // make sure we save the NDLs segments in LRAM only if the corresponding task goes to LRAM
            // otherwise we save them only in XRAM and a dma operation will copy them in LRAM
            if (wrap->cfg_desc_addr != TORQ_LADDR_NONE) {
                _add_bitstream_segment(wrap, wrap->ndl_desc_addr[i], wrap->ndl_desc_xaddr[i], wrap->ndl_desc_len[i]<<2, wrap->ndl_desc_buf[i]);
            } else {
                _add_bitstream_segment(wrap, TORQ_LADDR_NONE, wrap->ndl_desc_xaddr[i], wrap->ndl_desc_len[i]<<2, wrap->ndl_desc_buf[i]);
            }

            if (wrap->ndl_desc_xaddr[i] == TORQ_XADDR_NONE) {
                if (wrap->ndl_desc_addr[i] != TORQ_LADDR_NONE) {
                    if (wrap->dump_ctx)
                        torq_desc_dump__data(wrap->dump_ctx,
                            wrap->job_id, wrap->stg_id, task_id, wrap->slc_id,
                            'LRAM', 'INP', tag[i], 0,
                            wrap->ndl_desc_addr[i], wrap->ndl_desc_len[i]<<2, wrap->ndl_desc_buf[i]);
                }
                wrap->ndl_desc_addr[i] += wrap->ndl_desc_len[i]<<2;
            }
            else {
                if (wrap->ndl_desc_addr[i] != TORQ_LADDR_NONE) {
                    if (wrap->dump_ctx)
                        torq_desc_dump__data(wrap->dump_ctx,
                            wrap->job_id, wrap->stg_id, task_id, wrap->slc_id,
                            'XRAM', 'INP', tag[i], 0,
                            wrap->ndl_desc_xaddr[i], wrap->ndl_desc_len[i]<<2, wrap->ndl_desc_buf[i]);
                }
                wrap->ndl_desc_xaddr[i] += wrap->ndl_desc_len[i]<<2;
            }
            wrap->ndl_desc_len[i] = 0;
        }
    }
    return 0;
}


#include "torq_css_regs.h"
#include "torq_cpu_regs.h"
#include "torq_cdma_regs.h"
#include "torq_regs_nss_view.h"
#define RU_PREFIX0 NV_
#include "torq_reg_util.h"

#define CFG_WR_INIT() do { wrap->cfg_wr_a = wrap->cfg_wr_n = wrap->cfg_wr_i = 0; } while (0)
#define CFG_WR_N(n_,a_,d_) do { \
        assert(wrap->cfg_wr_i==wrap->cfg_wr_n); \
        assert(!((n_)&~0xfff)); \
        assert(!((a_)&~0xffff)); \
        wrap->cfg_desc_last_hdr = &wrap->cfg_desc_buf[wrap->cfg_desc_len]; \
        wrap->cfg_desc_buf[wrap->cfg_desc_len++] = 0x10000000 | ((n_)<<16) | (a_); \
        wrap->cfg_desc_buf[wrap->cfg_desc_len++] = (d_); \
        wrap->cfg_wr_a = (a_)+4; \
        wrap->cfg_wr_i = 1; \
        wrap->cfg_wr_n = (n_); \
        xdbg("@IO_WR32: %04x <- %08x\n", (a_), (d_)); \
        } while (0)
#define CFG_WR_C(a_,d_) do { \
        assert((a_)==wrap->cfg_wr_a); \
        assert(wrap->cfg_wr_i<wrap->cfg_wr_n); \
        wrap->cfg_desc_buf[wrap->cfg_desc_len++] = (d_); \
        wrap->cfg_wr_a += 4; \
        wrap->cfg_wr_i ++; \
        xdbg("-IO_WR32: %04x <- %08x\n", (a_), (d_)); \
        } while (0)
#define CFG_WR_S(a_,d_) CFG_WR_N(1,(a_),(d_))

int _css_set_ien(torq_wrap_t *wrap)
{
    CFG_WR_INIT();
    CFG_WR_N(2, RA_(CSS,IEN_CSS               ), RF_LSH(CSS,IEN_CSS_CDMA, !wrap->cdma_in_use) // disable CDMA IRQ going to CSS if CDMA is owned by NSS
                                               | RF_LSH(CSS,IEN_CSS_MBX0, 1)); // always enable NSS2CSS IRQ going to CSS
    CFG_WR_C(   RA_(CSS,IEN_NSS               ), RF_LSH(CSS,IEN_NSS_CDMA, 1)   // always enable CDMA IRQ going to NSS (it's harmless even when CSS owns CDMA)
                                               | RF_LSH(CSS,IEN_NSS_MBX1, 1)); // always enable CSS2NSS IRQ going to NSS (it's harmless even when CSS is not enabled)
    return 0;
}

int _css_dma_start(torq_wrap_t *wrap)
{
    xdbg("css_dma_config:\n");
    CFG_WR_INIT();
    CFG_WR_S(   RA_(CDMA,NSEC_CTRL            ), RF_LSH(CDMA,NSEC_CTRL_INTREN_ANYCHINTR, 1));
    CFG_WR_N(3, RA_(CDMA,DMACH0_CH_INTREN     ), RF_LSH(CDMA,DMACH0_CH_INTREN_INTREN_DONE, 1));
    CFG_WR_C(   RA_(CDMA,DMACH0_CH_CTRL       ), RF_LSH(CDMA,DMACH0_CH_CTRL_TRANSIZE,  0)
                                               | RF_LSH(CDMA,DMACH0_CH_CTRL_XTYPE,     1)
                                               | RF_LSH(CDMA,DMACH0_CH_CTRL_DONETYPE,  1));
    CFG_WR_C(   RA_(CDMA,DMACH0_CH_SRCADDR    ), wrap->cfg.cdma_src_addr);
    CFG_WR_S(   RA_(CDMA,DMACH0_CH_DESADDR    ), wrap->cfg.cdma_dst_addr);
    CFG_WR_S(   RA_(CDMA,DMACH0_CH_XSIZE      ), RF_BMSK_LSH(CDMA,DMACH0_CH_XSIZE_SRCXSIZE, wrap->cfg.cdma_len)
                                               | RF_BMSK_LSH(CDMA,DMACH0_CH_XSIZE_DESXSIZE, wrap->cfg.cdma_len));
    CFG_WR_N(3, RA_(CDMA,DMACH0_CH_SRCTRANSCFG), RF_LSH(CDMA,DMACH0_CH_SRCTRANSCFG_SRCMEMATTRLO,    4)
                                               | RF_LSH(CDMA,DMACH0_CH_SRCTRANSCFG_SRCMEMATTRHI,    4)
                                               | RF_LSH(CDMA,DMACH0_CH_SRCTRANSCFG_SRCMAXBURSTLEN, 15));
    CFG_WR_C(   RA_(CDMA,DMACH0_CH_DESTRANSCFG), RF_LSH(CDMA,DMACH0_CH_DESTRANSCFG_DESMEMATTRLO,    4)
                                               | RF_LSH(CDMA,DMACH0_CH_DESTRANSCFG_DESMEMATTRHI,    4)
                                               | RF_LSH(CDMA,DMACH0_CH_DESTRANSCFG_DESMAXBURSTLEN, 15));
    CFG_WR_C(   RA_(CDMA,DMACH0_CH_XADDRINC   ), RF_LSH(CDMA,DMACH0_CH_XADDRINC_SRCXADDRINC,        1)
                                               | RF_LSH(CDMA,DMACH0_CH_XADDRINC_DESXADDRINC,        1));
    //CFG_WR_S(   RA_(CDMA,DMACH0_CH_ISSUECAP   ), RF_LSH(CDMA,DMACH0_CH_ISSUECAP_ISSUECAP,  0));
    CFG_WR_S(   RA_(CDMA,DMACH0_CH_CMD        ), RF_LSH(CDMA,DMACH0_CH_CMD_ENABLECMD,      1));
    return 0;
}


int _css_dma_stop(torq_wrap_t *wrap)
{
    xdbg("css_dma_stop:\n");
    CFG_WR_INIT();
    CFG_WR_S(RA_(CDMA,DMACH0_CH_STATUS), RF_LSH(CDMA,DMACH0_CH_STATUS_STAT_DONE, 1));
    return 0;
}

int _css_cpu_start(torq_wrap_t *wrap)
{
    xdbg("css_cpu_start:\n");
    CFG_WR_INIT();
    CFG_WR_N(4, RA_(CSS,MBX_DAT_0), wrap->cfg.css_mbx[0]);
    CFG_WR_C(   RA_(CSS,MBX_DAT_1), wrap->cfg.css_mbx[1]);
    CFG_WR_C(   RA_(CSS,MBX_DAT_2), wrap->cfg.css_mbx[2]);
    CFG_WR_C(   RA_(CSS,MBX_DAT_3), wrap->cfg.css_mbx[3]);
    if (wrap->cfg.css_start_mode == 0) { // restart
        CFG_WR_S(RA_(CPU,PCSTART), wrap->cfg.css_start_addr);
        // 2 steps per Google's requirments
        CFG_WR_S(RA_(CPU,CTL), RF_LSH(CPU,CTL_CG, 0) | RF_LSH(CPU,CTL_RESET, 1)); // from cg==1 && rst==1 to cg==0 && rst==1
        CFG_WR_S(RA_(CPU,CTL), RF_LSH(CPU,CTL_CG, 0) | RF_LSH(CPU,CTL_RESET, 0)); // from cg==0 && rst==1 to cg==0 && rst==0
    }
    else { // wakeup
        CFG_WR_S(   RA_(CSS,MBX_IRQ_0), 1); // it's cpu's resonsiblity to clear this bit
    }
    return 0;
}

int _css_cpu_stop(torq_wrap_t *wrap)
{
    xdbg("css_cpu_stop:\n");
    CFG_WR_INIT();
    CFG_WR_S(RA_(CSS,MBX_IRQ_1), 0); // clear mbx irq from cpu
    CFG_WR_S(RA_(CSS,IEN_CSS), 0);  // disable all interrupts going to CSS CPU
    if (wrap->cfg.css_wait_mode == 0) { // reset and clock gated
        CFG_WR_S(RA_(CPU,CTL), RF_LSH(CPU,CTL_CG, 1) | RF_LSH(CPU,CTL_RESET, 1)); // from cg==0 && rst==0 to cg==1 && rst==1
    }
    else { // let it sleep
        // do nothing
    }
    return 0;
}

static void _nss_cfg_desc_init(torq_wrap_t *wrap)
{
    memset(&wrap->regs.DMA, 0, sizeof(wrap->regs.DMA));
    memset(&wrap->regs.NSS, 0, sizeof(wrap->regs.NSS));
}

static int _nss_cfg_desc_gen(torq_wrap_t *wrap)
{
    int i, n, a;
    uint32_t wait;

    i = 0;
    if (wrap->cfg.slc_start[i]) {
        n = 1;
        a = offsetof(torq_nss_regs_t,SLC0.DE_REGS.DE_CFG);
        assert(wrap->cfg_desc_len+n+1 < CFG_DESC_MAX_LEN_PER_TASK);
        wrap->regs.SLC0.DE_REGS.DE_CFG.desc = wrap->cfg.slc_cfg_addr[i];
        wrap->regs.SLC0.DE_REGS.DE_CFG.link_en = 1;
        wrap->cfg_desc_buf[wrap->cfg_desc_len++] = 0x10000000 | (n<<16) | a;
        memcpy(&wrap->cfg_desc_buf[wrap->cfg_desc_len], &wrap->regs.SLC0.DE_REGS.DE_CFG, n*4);
        wrap->cfg_desc_len += n;
    }
    i++;
    if (wrap->cfg.slc_start[i]) {
        n = 1;
        a = offsetof(torq_nss_regs_t,SLC1.DE_REGS.DE_CFG);
        assert(wrap->cfg_desc_len+n+1 < CFG_DESC_MAX_LEN_PER_TASK);
        wrap->regs.SLC1.DE_REGS.DE_CFG.desc = wrap->cfg.slc_cfg_addr[i];
        wrap->regs.SLC1.DE_REGS.DE_CFG.link_en = 1;
        wrap->cfg_desc_buf[wrap->cfg_desc_len++] = 0x10000000 | (n<<16) | a;
        memcpy(&wrap->cfg_desc_buf[wrap->cfg_desc_len], &wrap->regs.SLC1.DE_REGS.DE_CFG, n*4);
        wrap->cfg_desc_len += n;
    }
    if (wrap->cfg.dma_xr_start) {
        n = sizeof(wrap->regs.DMA.DMA_XR)/4;
        a = offsetof(torq_nss_regs_t,DMA.DMA_XR);
        assert(wrap->cfg_desc_len+n+1 < CFG_DESC_MAX_LEN_PER_TASK);
        wrap->cfg_desc_buf[wrap->cfg_desc_len++] = 0x10000000 | (n<<16) | a;
        memcpy(&wrap->cfg_desc_buf[wrap->cfg_desc_len], &wrap->regs.DMA.DMA_XR, n*4);
        wrap->cfg_desc_len += n;
    }
    if (wrap->cfg.dma_xw_start) {
        n = sizeof(wrap->regs.DMA.DMA_XW)/4;
        a = offsetof(torq_nss_regs_t,DMA.DMA_XW);
        assert(wrap->cfg_desc_len+n+1 < CFG_DESC_MAX_LEN_PER_TASK);
        wrap->cfg_desc_buf[wrap->cfg_desc_len++] = 0x10000000 | (n<<16) | a;
        memcpy(&wrap->cfg_desc_buf[wrap->cfg_desc_len], &wrap->regs.DMA.DMA_XW, n*4);
        wrap->cfg_desc_len += n;
    }

    if (wrap->cdma_release)   wrap->cdma_in_use = wrap->cdma_release = 0;
    if (wrap->cfg.cdma_start) wrap->cdma_in_use = 1;
    if (wrap->cfg.cdma_wait)  wrap->cdma_release = 1;
    if (wrap->cfg.cdma_start || wrap->cfg.css_start || wrap->cfg.cdma_wait || wrap->cfg.css_wait) _css_set_ien(wrap);

    if (wrap->cfg.cdma_start) _css_dma_start(wrap);
    if (wrap->cfg.css_start)  _css_cpu_start(wrap);

    wrap->regs.NSS.START.nss     = 0;
    wrap->regs.NSS.START.xr      = !!wrap->cfg.dma_xr_wait;
    wrap->regs.NSS.START.xw      = !!wrap->cfg.dma_xw_wait;
    wrap->regs.NSS.START.slc0    = !!wrap->cfg.slc_wait[0];
    wrap->regs.NSS.START.slc1    = !!wrap->cfg.slc_wait[1];
    wait = (*(uint32_t *)&wrap->regs.NSS.START)&0xffff;
    wait |= ((!!wrap->cfg.cdma_wait)<<12) | ((!!wrap->cfg.css_wait)<<9); //assuming CSS2NSS IRQ is MBX1 (#9)
    wrap->regs.NSS.CTRL.ien_nss  = 1;
    wrap->regs.NSS.CTRL.ien_xr   = 0;
    wrap->regs.NSS.CTRL.ien_xw   = 0;
    wrap->regs.NSS.CTRL.ien_slc0 = 0;
    wrap->regs.NSS.CTRL.ien_slc1 = 0;
    wrap->regs.NSS.STATUS.nss    = 0;
    wrap->regs.NSS.STATUS.xr     = !!wrap->cfg.dma_xr_start;
    wrap->regs.NSS.STATUS.xw     = !!wrap->cfg.dma_xw_start;
    wrap->regs.NSS.STATUS.slc0   = !!wrap->cfg.slc_start[0];
    wrap->regs.NSS.STATUS.slc1   = !!wrap->cfg.slc_start[1];
    wrap->regs.NSS.START.nss     = 0;
    wrap->regs.NSS.START.xr      = !!wrap->cfg.dma_xr_start;
    wrap->regs.NSS.START.xw      = !!wrap->cfg.dma_xw_start;
    wrap->regs.NSS.START.slc0    = !!wrap->cfg.slc_start[0];
    wrap->regs.NSS.START.slc1    = !!wrap->cfg.slc_start[1];
    n = 3;
    a = 8;
    assert(wrap->cfg_desc_len+n+1+1 < CFG_DESC_MAX_LEN_PER_TASK);
    wrap->cfg_desc_buf[wrap->cfg_desc_len++] = 0x10000000 | (n<<16) | a;
    memcpy(&wrap->cfg_desc_buf[wrap->cfg_desc_len], &wrap->regs.NSS.CTRL, n*4);
    wrap->cfg_desc_len += n;

    wrap->cfg_desc_last_hdr = &wrap->cfg_desc_buf[wrap->cfg_desc_len];
    wrap->cfg_desc_buf[wrap->cfg_desc_len++] = 0x60000000|wait;

    if (wrap->cfg.cdma_wait) _css_dma_stop(wrap);
    if (wrap->cfg.css_wait)  _css_cpu_stop(wrap);

    return 0;
}

static void _slc_cfg_desc_init(torq_wrap_t *wrap)
{
    memset(&wrap->regs.SLC0, 0, sizeof(wrap->regs.SLC0));

    wrap->regs.SLC0.CE_REGS.COMM.cedw_dis = 1;
    wrap->regs.SLC0.DE_REGS.DE_B_R.disabled = 1;
    wrap->regs.SLC0.CE_REGS.CTRL.aldw_l_size = 1;
    wrap->regs.SLC0.CE_REGS.CTRL.aldw_s_size = 1;
    wrap->regs.SLC0.CE_REGS.CTRL_C1.aldw_l_size = 1;
    wrap->regs.SLC0.CE_REGS.CTRL_C1.aldw_s_size = 1;
#ifdef CAST_INT
    wrap->regs.SLC0.CE_REGS.AL_MODE.cast_bool = 1;
    wrap->regs.SLC0.CE_REGS.AL_MODE.cast_int = CAST_INT;
#endif
}

static int _slc_cfg_desc_gen(torq_wrap_t *wrap)
{
    int a, n;
    uint32_t u32;

    assert(wrap->cfg.w_format == 0 || wrap->cfg.de_w_unsigned == 0); //only one of them can be speficied
    assert(wrap->cfg.w_format == 0 || wrap->cfg.w_format == 'SI' || wrap->cfg.w_format == 'UI');

    switch (wrap->cfg.act_format) {
    case 0:
    case 'I' : wrap->regs.SLC0.DE_REGS.DE_ACT.func_type = 0; break;
    case 'BF': wrap->regs.SLC0.DE_REGS.DE_ACT.func_type = 1; break;
    default: assert(!"ERROR: invalid act_format");
    }
    wrap->regs.SLC0.DE_REGS.DE_ACT.clip32_dis = 1;
    wrap->regs.SLC0.DE_REGS.DE_ACT.pfpp_byp = 1; // TODO: FIXME. temporal hack for ACT FP functions
    switch (wrap->cfg.act_mode) {
    case 0:
    case 'ACT': wrap->regs.SLC0.DE_REGS.DE_ACT.act_mode = 0; 
                // TODO: FIXME. temporal hack for ACT FP functions
                if (wrap->cfg.alu_op1_mode[0]==0 ||
                    wrap->cfg.alu_op1_mode[0]=='ACC' ||
                    wrap->cfg.alu_op1_mode[0]=='SACC' ||
                    wrap->cfg.alu_op1_mode[0]=='MUL') {
                    wrap->regs.SLC0.DE_REGS.DE_ACT.pfpp_byp = 0;
                }
                break;
    case 'ABS': wrap->regs.SLC0.DE_REGS.DE_ACT.act_mode = 1; break;
    case 'NEG': wrap->regs.SLC0.DE_REGS.DE_ACT.act_mode = 2; break;
    case 'CLZ': wrap->regs.SLC0.DE_REGS.DE_ACT.act_mode = 3; break;
    case 'CEL': wrap->regs.SLC0.DE_REGS.DE_ACT.act_mode = 4; break;
    case 'FLR': wrap->regs.SLC0.DE_REGS.DE_ACT.act_mode = 5; break;
    case 'I2F': wrap->regs.SLC0.DE_REGS.DE_ACT.act_mode = 6; break;
    case 'F2I': wrap->regs.SLC0.DE_REGS.DE_ACT.act_mode = 6; 
                wrap->regs.SLC0.DE_REGS.DE_ACT.clip32_dis = 0; break;
    case 'LSL': wrap->regs.SLC0.DE_REGS.DE_ACT.act_mode = 7; break;
    case 'LSR': wrap->regs.SLC0.DE_REGS.DE_ACT.act_mode = 8; break;
    case 'ASR': wrap->regs.SLC0.DE_REGS.DE_ACT.act_mode = 9; break;
    default: assert(!"ERROR: invalid act_mode");
    }
    wrap->regs.SLC0.DE_REGS.DE_ACT.bbus_byp = !(wrap->ndl_desc_len[2]); //DEBR
    //wrap->regs.SLC0.DE_REGS.DE_ACT.clip32_dis = 1;
    assert(wrap->cfg.act_zero_point >= -(1 << 15) && wrap->cfg.act_zero_point <= (1 << 15) - 1);
    assert(wrap->cfg.act_sh<16);
    assert(wrap->cfg.act_rsh<64);
    wrap->regs.SLC0.DE_REGS.DE_ACT.act_sh = wrap->cfg.act_rsh+wrap->cfg.act_sh*4;
    switch (wrap->cfg.act_round_mode) {
    case 0:
    case 'NTP': wrap->regs.SLC0.DE_REGS.DE_ACT.rnd_type = 0; break;
    case 'DBL': wrap->regs.SLC0.DE_REGS.DE_ACT.rnd_type = 1; break;
    case 'NTE': wrap->regs.SLC0.DE_REGS.DE_ACT.rnd_type = 2; break;
    case 'OFF': wrap->regs.SLC0.DE_REGS.DE_ACT.rnd_type = 3; break;
    default: assert(!"ERROR: invalid act_round_mode");
    }
    assert(wrap->acbr_ldim_sz==wrap->debr_ldim_sz || wrap->debr_ldim_sz==0);
    wrap->regs.SLC0.DE_REGS.DE_ACT.a_size = 1;
    wrap->regs.SLC0.DE_REGS.DE_ACT.q_size = wrap->deqw_bn==1 ? 0 : wrap->deqw_bn==2 ? 1 : 2;
    wrap->regs.SLC0.DE_REGS.DE_ACT.qclp_min = wrap->cfg.act_clip_min;
    wrap->regs.SLC0.DE_REGS.DE_ACT.qclp_max = wrap->cfg.act_clip_max;
    wrap->regs.SLC0.DE_REGS.DE_ACT.qzero_pt = wrap->cfg.act_zero_point;
    if      (wrap->cfg.act_lsh[0] == 0 && wrap->cfg.act_lsh[1] == 0 && wrap->cfg.act_lsh[2] == 0  && wrap->cfg.act_lsh[3] == 0 )
        wrap->regs.SLC0.DE_REGS.DE_ACT.sum_psh = 0;
    else if (wrap->cfg.act_lsh[0] == 0 && wrap->cfg.act_lsh[1] == 8 && wrap->cfg.act_lsh[2] == 0  && wrap->cfg.act_lsh[3] == 8 )
        wrap->regs.SLC0.DE_REGS.DE_ACT.sum_psh = 1;
    else if (wrap->cfg.act_lsh[0] == 0 && wrap->cfg.act_lsh[1] == 8 && wrap->cfg.act_lsh[2] == 8  && wrap->cfg.act_lsh[3] == 16)
        wrap->regs.SLC0.DE_REGS.DE_ACT.sum_psh = 2;
    else if (wrap->cfg.act_lsh[0] == 0 && wrap->cfg.act_lsh[1] == 8 && wrap->cfg.act_lsh[2] == 16 && wrap->cfg.act_lsh[3] == 24)
        wrap->regs.SLC0.DE_REGS.DE_ACT.sum_psh = 3;
    else assert(!"ERROR: invalid act_lsh");
    assert(wrap->cfg.act_sum_bits==0 || wrap->cfg.act_sum_bits==8 || wrap->cfg.act_sum_bits==16 || wrap->cfg.act_sum_bits==32 || wrap->cfg.act_sum_bits==48);
    wrap->regs.SLC0.DE_REGS.DE_ACT.func_size = wrap->cfg.act_sum_bits==0  ? 2
                                        : wrap->cfg.act_sum_bits==8  ? 0
                                        : wrap->cfg.act_sum_bits==16 ? 1
                                        : wrap->cfg.act_sum_bits==32 ? 2
                                        : wrap->cfg.act_sum_bits==48 ? 3
                                        : 2;

    DV_DIAG_HOOK__SLC_CFG_DESC_GEN(); //for DV coverage only

    wrap->regs.SLC0.CE_REGS.COMM.lpad_en = !!wrap->cfg.pad_left;
    wrap->regs.SLC0.CE_REGS.COMM.rpad_en = !!wrap->cfg.pad_right;
    wrap->regs.SLC0.CE_REGS.COMM.knl_l = wrap->cfg.kernel_left;
    wrap->regs.SLC0.CE_REGS.COMM.knl_r = wrap->cfg.kernel_right;

    wrap->regs.SLC0.CE_REGS.AL_MODE.cepx_clr_en = !wrap->cfg.no_p_clear;
    wrap->regs.SLC0.CE_REGS.AL_MODE.cepx_out_en = !wrap->cfg.no_p_output;
    wrap->regs.SLC0.CE_REGS.AL_MODE_C1.cepx_clr_en = 1;    
    wrap->regs.SLC0.CE_REGS.AL_MODE_C1.cepx_out_en = 0; //FIXME 
    switch (wrap->cfg.alu_format) {
    case 0:
    case 'I' : wrap->regs.SLC0.CE_REGS.AL_MODE.data_format = 0; break;
    case 'BF': wrap->regs.SLC0.CE_REGS.AL_MODE.data_format = 1; break;
    default: assert(!"ERROR: invalid alu_format");
    }
    wrap->regs.SLC0.CE_REGS.AL_MODE_C1.data_format = wrap->regs.SLC0.CE_REGS.AL_MODE.data_format;
    wrap->regs.SLC0.CE_REGS.COMM.cewr_zp_val = wrap->cfg.alu_format=='BF' ? 0x3f80 : 
                                               wrap->cfg.alu_op1_mode[0] == 'SACC' ? 0xff01 : 0x0101;
    wrap->regs.SLC0.CE_REGS.COMM.ceww_dis = wrap->cfg.alu_op0_mode[0] == 'DBYP' ?
                                               ((!wrap->cfg.alu_op1_mode[0] || 
                                                  wrap->cfg.alu_op1_mode[0] == 'ACC' ||
                                                  wrap->cfg.alu_op1_mode[0] == 'SACC') ? 1 : 2) : 0;
    #ifdef NAN_MODE
      wrap->regs.SLC0.CE_REGS.AL_MODE.nan_mode = NAN_MODE;
      wrap->regs.SLC0.CE_REGS.AL_MODE.nan_mode = wrap->regs.SLC0.CE_REGS.AL_MODE.nan_mode;
    #else
      wrap->regs.SLC0.CE_REGS.AL_MODE.nan_mode = 0;
      wrap->regs.SLC0.CE_REGS.AL_MODE.nan_mode = wrap->regs.SLC0.CE_REGS.AL_MODE.nan_mode;
    #endif

    switch (wrap->cfg.alu_op1_mode[4]) {
    case 0:
    case 'ACC' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 0; //ACC
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.pre_norm_en = 1;
                 break;
    case 'SACC' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 0; //ACC
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.pre_norm_en = 1;
                 wrap->regs.SLC0.CE_REGS.COMM.cewr_tgl_msb = (wrap->regs.SLC0.CE_REGS.AL_MODE.data_format==1);
                 break;
    case 'SEL' : //INT only (float treated as int)
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 0; //ACC
                 wrap->regs.SLC0.CE_REGS.COMM.cewr_tgl_lsb = 1;
                 break;
    case 'MUL' : //BF16 only
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 1; //MUL
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.pre_norm_en = 1;
                 break;
    case 'BYP' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 2; //BYP
                 break;
    case 'MAX' : 
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 9; //GE
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.skip_first  = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.invert_comp = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.output_idx  = 0;
                 break;
    case 'MIN' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 8; //GT
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.skip_first  = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.invert_comp = 0;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.output_idx  = 0;
                 break;
    case 'AMAX':
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 9; //GE
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.skip_first  = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.invert_comp = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.output_idx  = 1;
                 break;
    case 'AMIN':
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 8; //GT
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.skip_first  = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.invert_comp = 0;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.output_idx  = 1;
                 break;
    case 'GT'  :
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 8; //GT
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.skip_first  = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.invert_comp = 0;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.output_idx  = 1;
                 break;
    case 'GE'  :
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 9; //GE
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.skip_first  = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.invert_comp = 0;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.output_idx  = 1;
                 break;
    case 'EQ'  :
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 10; //EQ
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.skip_first  = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.invert_comp = 0;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.output_idx  = 1;
                 break;
    case 'AND' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 4; //AND
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.cast_bool   = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.cast_int   = 1;
                 wrap->regs.SLC0.CE_REGS.INIT.init_val    = 1;
                 break;
    case 'OR'  :
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 5; //OR
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.cast_bool   = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.cast_int   = 1;
                 break;
    case 'XOR' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 6; //XOR
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.cast_bool   = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.cast_int   = 1;
                 break;
    case 'NOT' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 6; //XOR
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.cast_bool   = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.cast_int   = 1;
                 wrap->regs.SLC0.CE_REGS.INIT.init_val    = 1;
                 break;
    case 'BAND':
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 4; //AND
                 wrap->regs.SLC0.CE_REGS.INIT.init_val    = ~0;
                 break;
    case 'BOR' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 5; //OR
                 break;
    case 'BXOR':
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 6; //XOR
                 break;
    case 'BNOT'  :
                 wrap->regs.SLC0.CE_REGS.AL_MODE_C1.op_mode   = 6; //XOR
                 wrap->regs.SLC0.CE_REGS.INIT.init_val    = ~0;
                 break;
    default: assert(!"ERROR: invalid alu_op1_mode");
    }
    switch (wrap->cfg.alu_op1_mode[0]) {
    case 0:
    case 'ACC' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 0; //ACC
                 wrap->regs.SLC0.CE_REGS.AL_MODE.pre_norm_en = 1;
                 break;
    case 'SACC' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 0; //ACC
                 wrap->regs.SLC0.CE_REGS.AL_MODE.pre_norm_en = 1;
                 wrap->regs.SLC0.CE_REGS.COMM.cewr_tgl_msb = (wrap->regs.SLC0.CE_REGS.AL_MODE_C1.data_format==1);
                 break;
    case 'SEL' : //INT only (float treated as int)
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 0; //ACC
                 wrap->regs.SLC0.CE_REGS.COMM.cewr_tgl_lsb = 1;
                 break;
    case 'MUL' : //BF16 only
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 1; //MUL
                 wrap->regs.SLC0.CE_REGS.AL_MODE.pre_norm_en = 1;
                 break;
    case 'BYP' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 2; //BYP
                 break;
    case 'MAX' : 
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 9; //GE
                 wrap->regs.SLC0.CE_REGS.AL_MODE.skip_first  = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.invert_comp = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.output_idx  = 0;
                 break;
    case 'MIN' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 8; //GT
                 wrap->regs.SLC0.CE_REGS.AL_MODE.skip_first  = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.invert_comp = 0;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.output_idx  = 0;
                 break;
    case 'AMAX':
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 9; //GE
                 wrap->regs.SLC0.CE_REGS.AL_MODE.skip_first  = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.invert_comp = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.output_idx  = 1;
                 break;
    case 'AMIN':
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 8; //GT
                 wrap->regs.SLC0.CE_REGS.AL_MODE.skip_first  = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.invert_comp = 0;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.output_idx  = 1;
                 break;
    case 'GT'  :
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 8; //GT
                 wrap->regs.SLC0.CE_REGS.AL_MODE.skip_first  = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.invert_comp = 0;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.output_idx  = 1;
                 break;
    case 'GE'  :
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 9; //GE
                 wrap->regs.SLC0.CE_REGS.AL_MODE.skip_first  = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.invert_comp = 0;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.output_idx  = 1;
                 break;
    case 'EQ'  :
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 10; //EQ
                 wrap->regs.SLC0.CE_REGS.AL_MODE.skip_first  = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.invert_comp = 0;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.output_idx  = 1;
                 break;
    case 'AND' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 4; //AND
                 wrap->regs.SLC0.CE_REGS.AL_MODE.cast_bool   = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.cast_int   = 1;
                 wrap->regs.SLC0.CE_REGS.INIT.init_val    = 1;
                 break;
    case 'OR'  :
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 5; //OR
                 wrap->regs.SLC0.CE_REGS.AL_MODE.cast_bool   = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.cast_int   = 1;
                 break;
    case 'XOR' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 6; //XOR
                 wrap->regs.SLC0.CE_REGS.AL_MODE.cast_bool   = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.cast_int   = 1;
                 break;
    case 'NOT' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 6; //XOR
                 wrap->regs.SLC0.CE_REGS.AL_MODE.cast_bool   = 1;
                 wrap->regs.SLC0.CE_REGS.AL_MODE.cast_int   = 1;
                 wrap->regs.SLC0.CE_REGS.INIT.init_val    = 1;
                 break;
    case 'BAND':
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 4; //AND
                 wrap->regs.SLC0.CE_REGS.INIT.init_val    = ~0;
                 break;
    case 'BOR' :
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 5; //OR
                 break;
    case 'BXOR':
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 6; //XOR
                 break;
    case 'BNOT'  :
                 wrap->regs.SLC0.CE_REGS.AL_MODE.op_mode   = 6; //XOR
                 wrap->regs.SLC0.CE_REGS.INIT.init_val    = ~0;
                 break;
    default: assert(!"ERROR: invalid alu_op1_mode");
    }

    wrap->regs.SLC0.CE_REGS.AL_MODE.unsigned_w = wrap->cfg.alu_w_unsigned & 15;
    wrap->regs.SLC0.CE_REGS.AL_MODE.unsigned_d = wrap->cfg.alu_d_unsigned & 15;
    wrap->regs.SLC0.CE_REGS.AL_MODE_C1.unsigned_w = (wrap->cfg.alu_w_unsigned>>4) & 15;
    wrap->regs.SLC0.CE_REGS.AL_MODE_C1.unsigned_d = (wrap->cfg.alu_w_unsigned>>4) & 15;

    wrap->regs.SLC0.DE_REGS.CTRL.pri_dr = 1;
    wrap->regs.SLC0.DE_REGS.CTRL.pri_wr = 1;
    assert(wrap->cfg.stride>=0 && wrap->cfg.stride<=2);
    wrap->regs.SLC0.CE_REGS.COMM.stride = (wrap->cfg.stride==2);
    assert(wrap->cfg.stride_offset == 0 || wrap->cfg.stride_offset < wrap->cfg.stride);
    wrap->regs.SLC0.CE_REGS.COMM.stride_offset = wrap->cfg.stride_offset;

    //assert(wrap->cfg.pad_value <= (1<<15)-1 && wrap->cfg.pad_value >= -(1<<15));
    wrap->regs.SLC0.DE_REGS.DE_D_R.pad_value = wrap->cfg.pad_value;
    wrap->regs.SLC0.CE_REGS.COMM.pad_value = wrap->cfg.pad_value;
    wrap->regs.SLC0.CE_REGS.CG_CTRL.alu_disable = wrap->cfg.alu_disable;
    wrap->regs.SLC0.DE_REGS.CTRL.act_dis = wrap->cfg.act_disable;

    // load table
    if (wrap->cfg.table) {
        wrap->regs.SLC0.DE_REGS.DE_ACT.tbl_en = 1; 
        //assert(wrap->cfg.act_clip_min == -(1<<15) && wrap->cfg.act_clip_max == (1<<15)-1);
        n = sizeof(wrap->regs.SLC0.TBL_MEM) / 4;
        a = offsetof(slc_regs_t,TBL_MEM);
        assert(wrap->cfg_desc_len + n + 1 < CFG_DESC_MAX_LEN_PER_TASK);
        wrap->cfg_desc_buf[wrap->cfg_desc_len++] = 0x10000000 | (n << 16) | a;
        memcpy(&wrap->cfg_desc_buf[wrap->cfg_desc_len], wrap->cfg.table, n * 4);
        wrap->cfg_desc_len += n;
    }

    n = sizeof(wrap->regs.SLC0.CE_REGS)/4;
    a = offsetof(slc_regs_t,CE_REGS);
    assert(wrap->cfg_desc_len+n+1 < CFG_DESC_MAX_LEN_PER_TASK);
    wrap->cfg_desc_buf[wrap->cfg_desc_len++] = 0x10000000 | (n<<16) | a;
    memcpy(&wrap->cfg_desc_buf[wrap->cfg_desc_len], &wrap->regs.SLC0.CE_REGS, n*4);
    wrap->cfg_desc_len += n;

    n = (sizeof(wrap->regs.SLC0.DE_REGS))/ 4 -1;
    a = offsetof(slc_regs_t,DE_REGS.DE_D_R); //8
    assert(wrap->cfg_desc_len+n+1+1 < CFG_DESC_MAX_LEN_PER_TASK);
    wrap->cfg_desc_buf[wrap->cfg_desc_len++] = 0x10000000 | (n<<16) | a;
    memcpy(&wrap->cfg_desc_buf[wrap->cfg_desc_len], &wrap->regs.SLC0.DE_REGS.DE_D_R, (n-2)*4);
    wrap->cfg_desc_len += n-2;

    u32 = 0;
    if (wrap->cfg.alu_disable != 0xffff) u32 |= 0x10; //+CE
    if (wrap->cfg.act_disable != 0xf) u32 |= 0x100; //+ACT
    if (wrap->ndl_desc_len[0]) u32 |= 0x1; //+DEDR
    if (wrap->ndl_desc_len[1]) u32 |= 0x2; //+DEWR
    if (wrap->ndl_desc_len[2] || wrap->ndl_desc_len[5]) u32 |= 0x4; //+DEBR or DEBRX or both
    if (wrap->ndl_desc_len[3]) u32 |= 0x8; //+DEQW
    if (wrap->cfg.no_p_output) u32 &= ~(0x10|0x4|0x8); //-ACT -DEBR -DEQW 
    wrap->cfg_desc_buf[wrap->cfg_desc_len++] = 0x11f; // status
    wrap->cfg_desc_buf[wrap->cfg_desc_len++] = u32; // start
    wrap->cfg_desc_last_hdr = &wrap->cfg_desc_buf[wrap->cfg_desc_len];
    wrap->cfg_desc_buf[wrap->cfg_desc_len++] = u32 | 0x60000000; // wait

    return 0;
}

typedef struct _mem_bdg_ldims_t {
    uint32_t bn; // byte or bit count in data
    uint32_t dn; // data count in group
    uint32_t gn; // group count in step
    int32_t  ds; // data stride
    int32_t  gs; // group stride
    char     au_is_bit; // address unit is bit
    char     d_is_mergeable;
    char     g_is_mergeable;
} _mem_bdg_ldims_t;

static int _parse_mem_bdg_ldims(torq_wrap_t *wrap, torq_ndl_cmd_t *cmd, _mem_bdg_ldims_t *ldims)
{
    // Map mem_ndl's LDIMs to the {BDG} format:
    //
    // Definition of mergeable dimension:
    //   If (dim[i].n == 1) or (dim[i].s == dim[i-1].n * dim[i-1].s), dim[i] is mergeable.
    // Rules:
    //   If an LDIM is the first LDIM and its tag is 'B', it is mapped as the only B dimension and the address unit of the NDL descriptor is byte.
    //   If an LDIM is the first LDIM and its tag is 'b', it is mapped as the only B dimension and the address unit of the NDL descriptor is bit.
    //   Other than the first LDIM, LDIMs cannot have tag 'B' or 'b'.
    //   The remaining LDIMs are split into at most 2 dimension groups: a D group and a G group, in that order.
    //   A non-mergeable LDIM always starts a new dimension group.
    //   LDIMs with tag 'D' cannot be put in the G group.
    //   LDIMs with tag 'G' cannot be put in the D group.
    //   All LDIMs in the D group are merged and mapped as the only D dimension.
    //   All LDIMs in the G group are merged and mapped as the only G dimension.
    //   There can be at most one mapped B dimension, one mapped D dimension, and one mapped G dimension.
    //   Any remaining unmapped dimension is illegal.
    //   If the mapped B dimension does not exist, a default B dimension with B.n=1, B.s=0, address unit=byte is implied.
    //   If the mapped D dimension does not exist, a default D dimension with D.n=1, D.s=0 is implied.
    //   If the mapped G dimension does not exist, a default G dimension with G.n=1, G.s=0 is implied.

    int i, nd0, nd1, mergeable;
    int32_t s=1;
    char st='[';

    ldims->bn = ldims->dn = ldims->gn = 1;
    ldims->ds = ldims->gs = 0;
    ldims->au_is_bit = 0;
    ldims->d_is_mergeable = ldims->g_is_mergeable = 1;

    nd0 = 0;
    nd1 = cmd->nld;
    for (i=nd0; i<nd1; i++) {
        xdbg("_parse_mem_bdg_ldims: prev_st=%c, s=%d, i=%d, t[i]=%c, n[i]=%d, s[i]=%d", st, s, i, cmd->t[i], cmd->n[i], cmd->s[i]);
        assert(cmd->n[i]>0);
        if (i==nd0 && (cmd->t[i]=='B' || cmd->t[i]=='b')) { // mapped as B
            assert(cmd->n[i]==1 || cmd->s[i]==1);
            ldims->bn = cmd->n[i];
            if (cmd->t[i]=='b') {
                if ((ldims->bn&7)==0) ldims->bn /= 8;
                else ldims->au_is_bit = 1;
            }
            xdbg("  --> B.n=%d, B.s=1, au_is_bit=%d\n", ldims->bn, ldims->au_is_bit);
        }
        else {
            assert(cmd->t[i]!='B' && cmd->t[i]!='b');
            mergeable = (cmd->n[i]==1 || cmd->s[i]==s);
            if (st=='[' && cmd->t[i]!='G') { // first D
                ldims->dn = cmd->n[i];
                ldims->ds = (ldims->dn>1)?cmd->s[i]:s;
                ldims->d_is_mergeable = mergeable;
                xdbg("  --> D.n=%d, D.s=%d, D.is_mergeable=%d\n", ldims->dn, ldims->ds, ldims->d_is_mergeable);
                st = 'D'; // to merge D
            }
            else if (st=='D' && cmd->t[i]!='G' && mergeable) { // merge D
                ldims->dn *= cmd->n[i];
                xdbg("  --> D.n=%d, merged\n", ldims->dn);
            }
            else if ((st=='[' && cmd->t[i]=='G') ||
                     (st=='D' && (cmd->t[i]=='G' || (cmd->t[i]!='D' && !mergeable)))) { // first G
                ldims->gn = cmd->n[i];
                ldims->gs = (ldims->gn>1)?cmd->s[i]:s;
                ldims->g_is_mergeable = mergeable;
                xdbg("  --> G.n=%d, G.s=%d, G.is_mergeable=%d\n", ldims->gn, ldims->gs, ldims->g_is_mergeable);
                st = 'G'; // to merge G
            }
            else if (st=='G' && cmd->t[i]!='D' && mergeable) { // merge G
                ldims->gn *= cmd->n[i];
                xdbg("  --> G.n=%d, merged\n", ldims->gn);
            }
            else {
                xdbg("  --> ERROR\n");
                assert(!"ERROR: invalid dim in mem_bdg_ldims");
            }
        }
        if (cmd->n[i]>1) s = (int32_t)(cmd->n[i])*cmd->s[i];
    }
    return 0;
}

typedef struct _mem_bxy_sdims_t {
    uint32_t bn; // byte count in data
    uint32_t xn[2]; // width
    uint32_t yn[2]; // height
    int32_t  xs[2]; // x stride
    int32_t  ys[2]; // y stride
} _mem_bxy_sdims_t;

static int _parse_mem_bxy_sdims(torq_wrap_t *wrap, torq_ndl_cmd_t *cmd, _mem_bxy_sdims_t *sdims)
{
    // Map mem_ndl's SDIMs to the BXXYY format:
    //
    // Definition of mergeable dimension:
    //   If (dim[i].n == 1) or (dim[i].s == dim[i-1].n * dim[i-1].s), dim[i] is mergeable.
    // Rules:
    //   If an SDIM is the first SDIM and its tag is 'B', it is mapped as the only B dimension and the address unit of the NDL descriptor is byte.
    //   If the mapped B dimension does not exist, a default B dimension with B.n=1, B.s=0, address unit=byte is implied.
    //   The remaining SDIMs must be one of the following sequences:
    //     X, Y, XX, XY, YY, XXY, XYY, XXYY
    //   Other SDIM sequences are illegal.
    //   Consecutive X dims or Y dims are merged if the second X or Y dim is mergeable.

    int i, nd0, nd1, mergeable;
    int32_t s=1;
    char st='[';

    sdims->bn = sdims->xn[0] = sdims->xn[1] = sdims->yn[0] = sdims->yn[1] = 1;
    sdims->xs[0] = sdims->xs[1] = sdims->ys[0] = sdims->ys[1] = 0;

    nd0 = cmd->nld + cmd->nhd;
    nd1 = nd0 + cmd->nsd;
    for (i=nd0; i<nd1; i++) {
        xdbg("_parse_mem_bxy_sdims: prev_st=%c, s=%d, i=%d, t[i]=%c, n[i]=%d, s[i]=%d", st, s, i, cmd->t[i], cmd->n[i], cmd->s[i]);
        assert(cmd->n[i]>0);
        if (i==nd0 && cmd->t[i]=='B') { // mapped as B
            assert(cmd->n[i]==1 || cmd->s[i]==1);
            sdims->bn = cmd->n[i];
            xdbg("  --> B.n=%d, B.s=1\n", sdims->bn);
        }
        else {
            mergeable = (cmd->n[i]==1 || cmd->s[i]==s);
            if (st=='[' && cmd->t[i]=='X') { // X0
                sdims->xn[0] = cmd->n[i];
                sdims->xs[0] = (sdims->xn[0]>1)?cmd->s[i]:s;
                xdbg("  --> X0.n=%d, X0.s=%d, X0.is_mergeable=%d\n", sdims->xn[0], sdims->xs[0], mergeable);
                st = 'X'; // to X1
            }
            else if (st=='X' && cmd->t[i]=='X') {
                if (mergeable) { // merge X
                    sdims->xn[0] *= cmd->n[i];
                    xdbg("  --> X0.n=%d, merged\n", sdims->xn[0]);
                }
                else {
                    sdims->xn[1] = cmd->n[i];
                    sdims->xs[1] = (sdims->xn[1]>1)?cmd->s[i]:s;
                    xdbg("  --> X1.n=%d, X1.s=%d, X1.is_mergeable=%d\n", sdims->xn[1], sdims->xs[1], mergeable);
                }
                st = 'y'; // to Y0
            }
            else if ((st=='[' || st=='y' || st=='X') && cmd->t[i]=='Y') {
                sdims->yn[0] = cmd->n[i];
                sdims->ys[0] = (sdims->yn[0]>1)?cmd->s[i]:s;
                xdbg("  --> Y0.n=%d, Y0.s=%d, Y0.is_mergeable=%d\n", sdims->yn[0], sdims->ys[0], mergeable);
                st = 'Y'; // to Y1
            }
            else if (st=='Y' && cmd->t[i]=='Y') {
                if (mergeable) { // merge Y
                    sdims->yn[0] *= cmd->n[i];
                    xdbg("  --> Y0.n=%d, merged\n", sdims->yn[0]);
                }
                else {
                    sdims->yn[1] = cmd->n[i];
                    sdims->ys[1] = (sdims->yn[1]>1)?cmd->s[i]:s;
                    xdbg("  --> Y1.n=%d, Y1.s=%d, Y1.is_mergeable=%d\n", sdims->yn[1], sdims->ys[1], mergeable);
                }
                st = ']'; // END
            }
            else {
                xdbg("  --> ERROR\n");
                assert(!"ERROR: invalid dim in mem_bxy_sdims");
            }
        }
        if (cmd->n[i]>1) s = (int32_t)(cmd->n[i])*cmd->s[i];
    }
    return 0;
}

static int _nss_ndl_desc_gen(torq_wrap_t *wrap, uint32_t tag, torq_ndl_cmd_t *cmd)
{
    int r, i, j;
    int32_t s;
    _mem_bdg_ldims_t ldims;
    r = _parse_mem_bdg_ldims(wrap, cmd, &ldims); assert(r>=0);
    switch (tag) {
    case 'DIXR':
        assert(ldims.au_is_bit==0); // byte address only
        assert(ldims.bn==1 || ldims.bn==2); // only B1,B2 are supported
        assert(ldims.dn==1); // D is not supported
        assert(ldims.gn==1); // G is not supported
        assert(cmd->nhd>=1 && cmd->nhd<=4);
        assert(cmd->s[cmd->nld]==ldims.bn);
        assert((cmd->base_addr&(ldims.bn-1))==0);
        wrap->regs.DMA.DMA_XR.CFG.aruser  = (wrap->cfg.dma_xr_attr>> 0)&0xf;
        wrap->regs.DMA.DMA_XR.CFG.arprot  = (wrap->cfg.dma_xr_attr>> 4)&0x7;
        wrap->regs.DMA.DMA_XR.CFG.arcache = (wrap->cfg.dma_xr_attr>> 8)&0xf;
        wrap->regs.DMA.DMA_XR.CFG.arqos   = (wrap->cfg.dma_xr_attr>>12)&0xf;
        wrap->regs.DMA.DMA_XR.CFG.xlen = cmd->n[cmd->nld]*ldims.bn;
        wrap->regs.DMA.DMA_XR.CFG.pad_val0 = wrap->cfg.pad_value&0xff;
        wrap->regs.DMA.DMA_XR.CFG.pad_val1 =(wrap->cfg.pad_value>>8)&0xff;
        if (cmd->p[cmd->nld]!=0) {
            wrap->regs.DMA.DMA_XR.CFG.pad_ops = cmd->p[cmd->nld]<0 ? 1 : 0;
            wrap->regs.DMA.DMA_XR.CFG.pad_n = abs(cmd->p[cmd->nld])*ldims.bn;
        }
#ifndef TORQ_DMA_MTU_SIZE
        wrap->regs.DMA.DMA_XR.CFG.mtu = 2; //4 beats
#else
        wrap->regs.DMA.DMA_XR.CFG.mtu = TORQ_DMA_MTU_SIZE; // (1<<TORQ_DMA_MTU_SIZE) beats
#endif
        wrap->regs.DMA.DMA_XR.CFG.pix_size = ldims.bn==1?0:1;
        wrap->regs.DMA.DMA_XR.SRC.HEAD.a = cmd->base_addr;
        wrap->regs.DMA.DMA_XR.CFG.nd = (cmd->nhd==1)?1:cmd->nhd-1;
        if (cmd->nhd == 1) { //dummy dim
            wrap->regs.DMA.DMA_XR.SRC.DIMS[0].DIM_SIZE.n = 1;
            wrap->regs.DMA.DMA_XR.SRC.DIMS[0].DIM_STRIDE.s = 0;
        }
        for (i=cmd->nld+1; i<cmd->nld+cmd->nhd; i++) {
            assert(!(cmd->n[i]>>28));
            wrap->regs.DMA.DMA_XR.SRC.DIMS[i-cmd->nld-1].DIM_SIZE.n = cmd->n[i];
            s = cmd->s[i]; //s
            assert((s&(ldims.bn-1))==0);
            for (j=cmd->nld+1; j<i; j++) s -= (int)(cmd->n[j]-1)*cmd->s[j]; //s'
            assert(s>=-(1<<23) && s<(1<<23));
            wrap->regs.DMA.DMA_XR.SRC.DIMS[i-cmd->nld-1].DIM_STRIDE.s = s;
            if (cmd->p[i]!=0) {
                wrap->regs.DMA.DMA_XR.SRC.DIMS[i-cmd->nld-1].DIM_STRIDE.pad_ops = cmd->p[i]<0 ? 1 : 0;
                wrap->regs.DMA.DMA_XR.SRC.DIMS[i-cmd->nld-1].DIM_STRIDE.pad_n = abs(cmd->p[i])*ldims.bn;
            }
        }
        break;
    case 'DILW':
        assert(ldims.au_is_bit==0); // byte address only
        assert(ldims.bn==1 || ldims.bn==2); // only B1,B2 are supported
        assert(ldims.dn==1); // D is not supported
        assert(ldims.gn==1 || ldims.gn==2); // only G1,G2 are supported
        assert((ldims.gs&(ldims.bn-1))==0);
        assert((ldims.gs&7)==0);
        assert(ldims.gs>=-(1<<23) && ldims.gs<(1<<23)); //s21+3
        assert(cmd->nhd==1);
        assert(!(cmd->n[cmd->nld]>>24));
        assert(cmd->s[cmd->nld]==ldims.bn);
        assert((cmd->base_addr&(ldims.bn-1))==0);
        assert(!(cmd->base_addr>>24));
        wrap->regs.DMA.DMA_XR.CFG.laddr = cmd->base_addr;
        wrap->regs.DMA.DMA_XR.CFG.llen = cmd->n[cmd->nld]*ldims.bn*ldims.gn;
        wrap->regs.DMA.DMA_XR.CFG.split_en = ldims.gn==1?0:1;
        wrap->regs.DMA.DMA_XR.CFG.strd_div8 = ldims.gs/8;
        break;
    case 'DOLR':
        assert(ldims.au_is_bit==0); // byte address only
        assert(ldims.bn==1 || ldims.bn==2); // only B1,B2 are supported
        assert(ldims.dn==1); // D is not supported
        assert(ldims.gn==1); // G is not supported
        assert(cmd->nhd==1);
        assert(!(cmd->n[cmd->nld]>>24));
        assert(cmd->s[cmd->nld]==ldims.bn);
        assert((cmd->base_addr&(ldims.bn-1))==0);
        assert(!(cmd->base_addr>>24));
        wrap->regs.DMA.DMA_XW.CFG.laddr = cmd->base_addr;
        wrap->regs.DMA.DMA_XW.CFG.llen  = cmd->n[cmd->nld]*ldims.bn;
        break;
    case 'DOXW':
        assert(ldims.au_is_bit==0); // byte address only
        assert(ldims.bn==1 || ldims.bn==2); // only B1,B2 are supported
        assert(ldims.dn==1); // D is not supported
        assert(ldims.gn==1); // G is not supported
        assert(cmd->nhd>=1 && cmd->nhd<=4);
        assert(cmd->s[cmd->nld]==ldims.bn);
        assert((cmd->base_addr&(ldims.bn-1))==0);
        wrap->regs.DMA.DMA_XW.CFG.awuser  = (wrap->cfg.dma_xw_attr>> 0)&0xf;
        wrap->regs.DMA.DMA_XW.CFG.awprot  = (wrap->cfg.dma_xw_attr>> 4)&0x7;
        wrap->regs.DMA.DMA_XW.CFG.awcache = (wrap->cfg.dma_xw_attr>> 8)&0xf;
        wrap->regs.DMA.DMA_XW.CFG.awqos   = (wrap->cfg.dma_xw_attr>>12)&0xf;
        wrap->regs.DMA.DMA_XW.CFG.xlen    = cmd->n[cmd->nld]*ldims.bn;
        wrap->regs.DMA.DMA_XW.CFG.pad_val0 = wrap->cfg.pad_value&0xff;
        wrap->regs.DMA.DMA_XW.CFG.pad_val1 =(wrap->cfg.pad_value>>8)&0xff;
        if (cmd->p[cmd->nld]!=0) {
            wrap->regs.DMA.DMA_XW.CFG.pad_ops = cmd->p[cmd->nld]<0 ? 1 : 0;
            wrap->regs.DMA.DMA_XW.CFG.pad_n = abs(cmd->p[cmd->nld])*ldims.bn;
        }
#ifndef TORQ_DMA_MTU_SIZE
        wrap->regs.DMA.DMA_XW.CFG.mtu = 2; //4 beats
#else
        wrap->regs.DMA.DMA_XW.CFG.mtu = TORQ_DMA_MTU_SIZE; //(1<<TORQ_DMA_MTU_SIZE) beats
#endif
        wrap->regs.DMA.DMA_XW.CFG.pix_size = ldims.bn==1?0:1;
        wrap->regs.DMA.DMA_XW.DST.HEAD.a   = cmd->base_addr;
        wrap->regs.DMA.DMA_XW.CFG.nd       = (cmd->nhd==1)?1:cmd->nhd-1;
        if (cmd->nhd == 1) { //dummy dim
            wrap->regs.DMA.DMA_XW.DST.DIMS[0].DIM_SIZE.n = 1;
            wrap->regs.DMA.DMA_XW.DST.DIMS[0].DIM_STRIDE.s = 0;
        }
        for (i=cmd->nld+1; i<cmd->nld+cmd->nhd; i++) {
            assert(!(cmd->n[i]>>28));
            wrap->regs.DMA.DMA_XW.DST.DIMS[i-cmd->nld-1].DIM_SIZE.n = cmd->n[i];
            s = cmd->s[i]; //s
            assert((s&(ldims.bn-1))==0);
            for (j=cmd->nld+1; j<i; j++) s -= (int)(cmd->n[j]-1)*cmd->s[j]; //s'
            assert(s>=-(1<<23) && s<(1<<23));
            wrap->regs.DMA.DMA_XW.DST.DIMS[i-cmd->nld-1].DIM_STRIDE.s = s;
            if (cmd->p[i]!=0) {
                wrap->regs.DMA.DMA_XW.DST.DIMS[i-cmd->nld-1].DIM_STRIDE.pad_ops = cmd->p[i]<0 ? 1 : 0;
                wrap->regs.DMA.DMA_XW.DST.DIMS[i-cmd->nld-1].DIM_STRIDE.pad_n = abs(cmd->p[i])*ldims.bn;
            }
        }
        break;
    default: return 0;
    }
    return 1;
}

static int _proc_sw_ndl_desc(torq_wrap_t *wrap, uint32_t tag, int set_id, torq_ndl_cmd_t *cmd)
{
    int i, nd;
    assert(tag=='REF' || wrap->xn); //'REF' must be the first NDL descriptor
    nd = cmd->nld + cmd->nhd;
    switch (tag) {
    case 'REF':
        assert(set_id==0);
        wrap->xn = wrap->yn = 1;
        for (i=0; i<nd; i++) {
            if (cmd->t[i] == 'X') { wrap->xn *= cmd->n[i]; }
            if (cmd->t[i] == 'Y') { wrap->yn *= cmd->n[i]; }
        }
        break;
    case 'ALU' :
    case 'DMEM':
    case 'WMEM':
    case 'BMEM':
    case 'QMEM':
        break;
    default:
        return 0;
    }
    return 1;
}

typedef struct _reg_dims_t {
    uint32_t bn, dn, gn, sn, wn, tn;
    uint32_t in, jn, kn, ln, mn, nn;
    int32_t  ds, gs, ss, sp;
} _reg_dims_t;

static int _parse_reg_dims(torq_wrap_t *wrap, torq_ndl_cmd_t *cmd, _reg_dims_t *dims)
{
    int i, nd;
    nd = cmd->nld + cmd->nhd;
    dims->bn = dims->dn = dims->gn = dims->sn = dims->wn = dims->tn = 1;
    dims->in = dims->jn = dims->kn = dims->ln = dims->mn = dims->nn = 1;
    dims->ds = dims->gs = dims->ss = dims->sp = 0;
    for (i=0; i<nd; i++) {
        if (cmd->t[i] == 'B') { assert(dims->bn==1); assert(cmd->s[i]==1); dims->bn = cmd->n[i]; continue; }
        if (cmd->t[i] == 'D') { if (dims->dn==1) dims->ds = cmd->s[i]; dims->dn *= cmd->n[i]; continue; } //TODO: for older init
        //if (cmd->t[i] == 'D') { assert(dims->dn==1); dims->dn = cmd->n[i]; dims->ds = cmd->s[i]; continue; }
        if (cmd->t[i] == 'G') { assert(dims->gn==1); dims->gn = cmd->n[i]; dims->gs = cmd->s[i]; continue; }
        if (cmd->t[i] == 'S') { assert(dims->sn==1); dims->sn = cmd->n[i]; dims->ss = cmd->s[i]; dims->sp = cmd->p[i]; continue; }
        //if (cmd->t[i] == 'W') { dims->wn = cmd->n[i]; assert(dims->wn==1); continue; } // not supported in this configuration
        if (cmd->t[i] == 'W') { assert(dims->wn==1); dims->wn = cmd->n[i]; }
        if (cmd->t[i] == 'T') { assert(dims->tn==1); dims->tn = cmd->n[i]; continue; }
        if (cmd->t[i] == 'I') { assert(dims->in==1); assert(!cmd->s[i]); dims->in = cmd->n[i]; continue; }
        if (cmd->t[i] == 'J') { assert(dims->jn==1); assert(!cmd->s[i]); dims->jn = cmd->n[i]; continue; }
        if (cmd->t[i] == 'K') { assert(dims->kn==1); assert(!cmd->s[i]); dims->kn = cmd->n[i]; continue; }
        if (cmd->t[i] == 'L') { assert(dims->ln==1); assert(!cmd->s[i]); dims->ln = cmd->n[i]; continue; }
        if (cmd->t[i] == 'M') { assert(dims->mn==1); assert(!cmd->s[i]); dims->mn = cmd->n[i]; continue; }
        if (cmd->t[i] == 'N') { assert(dims->nn==1); assert(!cmd->s[i]); dims->nn = cmd->n[i]; continue; }
    }
    return 0;
}

static int _reg_ndl_desc_gen(torq_wrap_t *wrap, uint32_t tag, int set_id, torq_ndl_cmd_t *cmd)
{
    int r;
    uint32_t d, n;
    _reg_dims_t dims;
    switch (tag) {
    case 'ACBR':
        assert(set_id==0);
        r = _parse_reg_dims(wrap, cmd, &dims); assert(r>=0);
        assert(dims.bn==2 || dims.bn==4 || dims.bn==8);
        assert(dims.dn==1 || dims.dn==4 || dims.dn==8 || dims.dn==16);
        assert(dims.gn==1);
        wrap->regs.SLC0.DE_REGS.DE_ACT.a_glb = dims.bn==8?0:1;
        wrap->regs.SLC0.DE_REGS.DE_ACT.b_size = dims.bn==2?0:1;
        wrap->acbr_ldim_sz = dims.bn*dims.dn; //for assertion check only
        n = dims.mn*dims.nn;
        assert(n>=1 && n<=(1<<24));
        wrap->regs.SLC0.DE_REGS.DE_ACT.m_size = n==(1<<24)?0:n;
        break;
    case 'CEDR':
        r = _parse_reg_dims(wrap, cmd, &dims); assert(r>=0);
        d = dims.in*dims.bn*dims.dn;
        n = dims.mn*dims.nn;
        assert(dims.in==1 || dims.in==2 || dims.in==4);
        assert(dims.bn==1 || dims.bn==2 || dims.bn==4);
        assert(d==8 || d==16 || d==32 || d==64);
        assert(dims.gn==1 || dims.gn==4 || dims.gn==8 || dims.gn==16 || dims.gn==32);
        assert(d*dims.gn<=256);
        assert(dims.ds==dims.bn || dims.ds==2 || dims.ds==4 || dims.ds==16 || dims.ds==36);
        assert(dims.ln>=1 && dims.ln<=4);
        assert(dims.sn==1 || (dims.ss==1 || dims.ss==2 || dims.ss==4));
        assert(dims.sn>=1 && dims.sn<=4);
        if (set_id == 0) {
            wrap->regs.SLC0.CE_REGS.CTRL.cedr_i_size = dims.in==4?0:dims.in;
            wrap->regs.SLC0.CE_REGS.CTRL.cedr_g_size = dims.gn>1?0:1;
            if (d==16 && dims.ds==16) {
                wrap->regs.SLC0.CE_REGS.CTRL.cedr_d_size = 0;
                wrap->regs.SLC0.CE_REGS.CTRL.cedr_d_step = 2;
                wrap->regs.SLC0.CE_REGS.CTRL.cedr_s_size = 1;
            }
            else {
                wrap->regs.SLC0.CE_REGS.CTRL.cedr_d_size = d==64?0:d==32?3:d==16?2:d==8?1:0;
                wrap->regs.SLC0.CE_REGS.CTRL.cedr_d_step = dims.ds==dims.bn?0:dims.ds==2?1:dims.ds==4?2:dims.ds==36?3:0;
                wrap->regs.SLC0.CE_REGS.CTRL.cedr_s_size = dims.sn==4?0:dims.sn; // may be overridden below
            }
            wrap->regs.SLC0.CE_REGS.CTRL.cedr_l_size = dims.ln==4?0:dims.ln;
            assert(n>=1 && n<=(1<<24));
            wrap->regs.SLC0.CE_REGS.CTRL.cedr_n_size = n==(1<<24)?0:n;
            wrap->regs.SLC0.CE_REGS.CTRL.cedr_b_size = dims.bn==4?0:dims.bn;
            if (wrap->cfg.alu_op0_mode[0]=='DBYP')
            {
                uint8_t is_argop = (wrap->cfg.alu_op1_mode[0] == 'AMAX' ||
                                    wrap->cfg.alu_op1_mode[0] == 'AMIN' ||
                                    wrap->cfg.alu_op1_mode[0] == 'GT' ||
                                    wrap->cfg.alu_op1_mode[0] == 'GE' ||
                                    wrap->cfg.alu_op1_mode[0] == 'EQ');

                if (wrap->cfg.alu_op1_mode[0]!='BYP' &&
                    wrap->cfg.alu_op1_mode[0]!='MUL' &&
                    wrap->cfg.alu_op1_mode[0]!='ACC' &&
                    wrap->cfg.alu_op1_mode[0]!='SACC' &&
                    wrap->cfg.alu_op1_mode[0]!=0) {
                    switch(dims.bn) {
                    case 1: // 1B
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_g_size = 0;
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_w_size = 1;
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_b_size = is_argop ? 0 : 1;
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_unsigned = 0;
                        break;
                    case 2: // 2B
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_g_size = 2;
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_w_size = is_argop ? 1 : 2;
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_b_size = is_argop ? 0 : 1;
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_unsigned = 0;
                        break;
                    default: // 4B
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_g_size = 1;
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_w_size = is_argop ? 1 : 0;
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_b_size = is_argop ? 0 : 1;
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_unsigned = 0;
                        break;
                    }
                }
                else if ((wrap->cfg.alu_op1_mode[0]=='ACC' ||
                          wrap->cfg.alu_op1_mode[0]=='SACC' ||
                          wrap->cfg.alu_op1_mode[0]==0) && 
                          wrap->cfg.alu_format=='BF' && dims.bn==4) { // fp32 reduce sum
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_g_size = 1;
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_w_size = 2;
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_b_size = 0;
                        wrap->regs.SLC0.CE_REGS.AC_P.rmp_unsigned = 0;
                }
            }
            {
                uint32_t stride = wrap->cfg.stride==0 ? 1 : wrap->cfg.stride;
                uint32_t kw = wrap->cfg.kernel_left+1+wrap->cfg.kernel_right;
                if (!(kw == 1)) {
                wrap->regs.SLC0.CE_REGS.CTRL.cedr_s_size = 3;
                    wrap->regs.SLC0.CE_REGS.CTRL.cedr_s_break0 = (3-((kw+stride-1)/stride)%3)%3;
                    wrap->regs.SLC0.CE_REGS.CTRL.cedr_s_break1 = (3-(kw-(kw+stride-1)/stride)%3)%3;
                    wrap->regs.SLC0.CE_REGS.CTRL.cedr_i1_size = (kw+3*stride-1)/(3*stride);
                    wrap->regs.SLC0.CE_REGS.CTRL.cedr_i1_break = (stride==2)?(((kw%(stride*3))==1)?1:0):0;
                    wrap->regs.SLC0.CE_REGS.CTRL.cedr_i2_size = stride!=2;
                }
                else {
                  wrap->regs.SLC0.CE_REGS.CTRL.cedr_i2_size = 1;
                  wrap->regs.SLC0.CE_REGS.CTRL.cedr_i1_size = 1;
                  wrap->regs.SLC0.CE_REGS.CTRL.cedr_i1_break = 0;
                }
            }
            break;
        }
        else {
            assert(set_id==1); 
            wrap->regs.SLC0.CE_REGS.CTRL_C1.cedr_i_size = dims.in==4?0:dims.in;
            wrap->regs.SLC0.CE_REGS.CTRL_C1.cedr_g_size = dims.gn>1?0:1;
            if (d==16 && dims.ds==16) {
                wrap->regs.SLC0.CE_REGS.CTRL_C1.cedr_d_size = 0;
                wrap->regs.SLC0.CE_REGS.CTRL_C1.cedr_d_step = 2;
                wrap->regs.SLC0.CE_REGS.CTRL_C1.cedr_s_size = 1;
            }
            else {
                wrap->regs.SLC0.CE_REGS.CTRL_C1.cedr_d_size = d==64?0:d==32?3:d==16?2:d==8?1:0;
                wrap->regs.SLC0.CE_REGS.CTRL_C1.cedr_d_step = dims.ds==dims.bn?0:dims.ds==2?1:dims.ds==4?2:dims.ds==36?3:0;
                wrap->regs.SLC0.CE_REGS.CTRL_C1.cedr_s_size = dims.sn==4?0:dims.sn;
            }
            wrap->regs.SLC0.CE_REGS.CTRL_C1.cedr_l_size = dims.ln==4?0:dims.ln;
            assert(n>=1 && n<=(1<<8));
            wrap->regs.SLC0.CE_REGS.CTRL_C1.cedr_n_size = n==(1<<8)?0:n;
            break;
        }
    case 'CEWR':
        assert(set_id==0);
        r = _parse_reg_dims(wrap, cmd, &dims); assert(r>=0);
        assert(dims.bn==1 || dims.bn==2 || dims.bn==4);
        wrap->regs.SLC0.CE_REGS.CTRL.cewr_b_size = dims.bn==4?0:dims.bn;
        assert(dims.jn==1 || dims.jn==2 || dims.jn==4);
        wrap->regs.SLC0.CE_REGS.CTRL.cewr_j_size = dims.jn==4?0:dims.jn;
        assert(dims.gn==1 || dims.gn==4 || dims.gn==8 || dims.gn==16);
        wrap->regs.SLC0.CE_REGS.CTRL.cewr_g_size = dims.gn==16?0:dims.gn==8?3:dims.gn==4?2:1;
        assert(dims.dn>=1 && dims.dn<=32);
        wrap->regs.SLC0.CE_REGS.CTRL.cewr_d_size = dims.dn==1;
        assert(dims.ln>=0 && dims.ln<=4);
        wrap->regs.SLC0.CE_REGS.CTRL.cewr_l_size = dims.ln==4?0:dims.ln;
        assert(dims.sn==1 || dims.ss==dims.bn*dims.dn*dims.gn);
        assert(dims.sn>=1 && dims.sn<=36);
        wrap->regs.SLC0.CE_REGS.CTRL.cewr_s_size = dims.sn;
        assert(dims.sp>=1-(int)dims.sn && dims.sp<=0);
        wrap->regs.SLC0.CE_REGS.CTRL.cewr_s_break = -dims.sp;
        assert(dims.wn==1 || (dims.mn==1 && dims.nn==1)); //reuse the same h/w dim as either W ior N
        if (dims.wn==1) {
            n = dims.mn*dims.nn;
        }
        else {
            wrap->regs.SLC0.CE_REGS.COMM.cewr_has_w = 1;
            n = dims.wn;
        }
        assert(n>=1 && n<=(1<<24));
        wrap->regs.SLC0.CE_REGS.CTRL.cewr_n_size = n==(1<<24)?0:n;
        break;
    case 'CEPR':
    case 'CEPW':
        r = _parse_reg_dims(wrap, cmd, &dims); assert(r>=0);
        if (set_id == 0) {    
            n = dims.mn*dims.nn;
            assert(n>=1 && n<=(1<<16));
            wrap->regs.SLC0.CE_REGS.CTRL.cepw_n_size = n==(1<<16)?0:n;
            assert(dims.tn>=1 && dims.tn<=(1<<24));
            wrap->regs.SLC0.CE_REGS.CTRL.cepw_t_size = dims.tn==(1<<24)?0:dims.tn;
            break;
        }
        else {
            assert(set_id==1);
            n = dims.mn*dims.nn;
            assert(n>=1 && n<=(1<<8));
            wrap->regs.SLC0.CE_REGS.CTRL_C1.cepr_n_size = n==(1<<8)?0:n;
            break;
        }
    case 'ACPR':
        assert(set_id==0);
        r = _parse_reg_dims(wrap, cmd, &dims); assert(r>=0);
        assert(dims.bn==1 || dims.bn==2 || dims.bn==4);
        wrap->regs.SLC0.CE_REGS.AC_P.cepr_b_size = dims.bn==4?0:dims.bn; //TODO: rename
        assert(dims.dn==1 || (dims.ds<=64 && (dims.ds&3)==0));
        //assert(dims.ds==dims.bn);
        wrap->regs.SLC0.CE_REGS.AC_P.cepr_d_step = dims.ds>>2; //TODO: remove
        assert(dims.dn==2 || dims.dn==8 || dims.dn==4 || dims.dn==16);
        wrap->regs.SLC0.CE_REGS.AC_P.cepr_d_size = dims.dn==16?0:dims.dn;
        assert(dims.sn>=1 && dims.sn<=16);
        wrap->regs.SLC0.CE_REGS.AC_P.cepr_s1_size = dims.sn==64?0:dims.sn; //TODO: reduce
        assert(dims.sn==1 || (dims.ss==16 || dims.ss==32 || dims.ss==64 || dims.ss==128 || dims.ss==256));
        wrap->regs.SLC0.CE_REGS.AC_P.cepr_s1_step = dims.ss==16?4:dims.ss==32?3:dims.ss==64?2:dims.ss==128?1:dims.ss==256?0:0;
        wrap->regs.SLC0.CE_REGS.AC_P.cepr_s0_size = 1; //TODO: add s0==4?0:s0;
        break;
    case 'ACPW':
        assert(set_id==0);
        r = _parse_reg_dims(wrap, cmd, &dims); assert(r>=0);
        if (1) { // act/normal mode
            assert(dims.bn==4);
            assert(dims.dn==1 || dims.dn==2 || dims.dn==4);
            assert(dims.dn*dims.gn==16);
            wrap->regs.SLC0.DE_REGS.DE_ACT.sum_pgrp = dims.dn==4?2:dims.dn==2?1:0;
        }
        else { //TODO: elementwise
            assert(dims.bn==1 || dims.bn==2 || dims.bn==4);
            assert(dims.bn*dims.dn==32);
            assert(dims.gn==2);
        }
        break;
    case 'CEDW':
        r = _parse_reg_dims(wrap, cmd, &dims); assert(r>=0);
        assert(dims.sn>=1 && dims.sn<=4);
        if (set_id == 0) {
            wrap->regs.SLC0.CE_REGS.CTRL.cedw_s_size = dims.sn==4?0:dims.sn;
            wrap->regs.SLC0.CE_REGS.COMM.cedw_dis = !dims.tn;
            break;
        }
        else {
            assert(set_id==1); 
            wrap->regs.SLC0.CE_REGS.CTRL_C1.cedw_s_size = dims.sn==4?0:dims.sn;
            break;
        }
    case 'CEWW':
        assert(set_id==0);
        r = _parse_reg_dims(wrap, cmd, &dims); assert(r>=0);
        assert(dims.sn==1 || dims.sn==2 || dims.sn==4);
        wrap->regs.SLC0.CE_REGS.CTRL.ceww_s_size = dims.sn==4?0:dims.sn;
        break;
    case 'ALDW': // if not present, default is {D256}L1S1
        r = _parse_reg_dims(wrap, cmd, &dims); assert(r>=0);
        assert(dims.dn==64 || dims.dn==256);
        assert(dims.ln==1 || dims.ln==2 || dims.ln==4);
        assert(dims.sn==1 || dims.sn==4);
        if (set_id == 0) {
            wrap->regs.SLC0.CE_REGS.CTRL.aldw_d_size = dims.dn==64?1:0;
            wrap->regs.SLC0.CE_REGS.CTRL.aldw_l_size = dims.ln==4?0:dims.ln;
            wrap->regs.SLC0.CE_REGS.CTRL.aldw_s_size = dims.sn==4?0:1;
        }
        else {
            assert(set_id==1);
            wrap->regs.SLC0.CE_REGS.CTRL_C1.aldw_d_size = dims.dn==64?1:0;
            wrap->regs.SLC0.CE_REGS.CTRL_C1.aldw_l_size = dims.ln==4?0:dims.ln;
            wrap->regs.SLC0.CE_REGS.CTRL_C1.aldw_s_size = dims.sn==4?0:1;
        }
    case 'ACBW':
    case 'ALPR':
    case 'ALPW':
        break;
    default: return 0;
    }
    return 1;
}

static int _mem_ndl_desc_gen(torq_wrap_t* wrap, uint32_t tag, int set_id, uint32_t data_addr, torq_ndl_cmd_t *cmd, int last_cmd, uint32_t *desc_buf)
{
    int i, j, nd;
    int32_t s, p;
    uint32_t len=0;
    int ndi=0, di=-1, dip=-1, dis=-1, ndj=0, dj=-1, djp=-1, djs=-1;
    nd = cmd->nld + cmd->nhd;
    assert(cmd->nhd>=1 && cmd->nhd<=15);
    if (tag=='DEDR' && set_id == 0) data_addr &= (1<<24)-1;
    else assert(data_addr<(1<<24));
    desc_buf[len++] = 0
        | ((!!last_cmd)<<31) //last_cmd
        | ((last_cmd ?2 : 0)<<28) //id
        | (cmd->nhd<<24) //nd
        | (data_addr); //addr
    for (i=cmd->nld; i<nd; i++) {
        if (cmd->t[i]=='I') ndi++;
        if (cmd->t[i]=='J') ndj++;
    }
    for (i=cmd->nld; i<nd; i++) {
        if (!cmd->n[i]) return 0;
        assert(cmd->n[i]<(1<<28));
        if (cmd->t[i]=='I') {
            di++;
            if (wrap->cfg.stride==2 && di==ndi-1) dip = di;
            if (wrap->cfg.stride==2 && di==0) dis = i;
        }
        if (cmd->t[i]=='J') {
            dj++;
            if (wrap->cfg.stride==2 && dj==0) djp = dj;
            if (wrap->cfg.stride==2 && dj==0) djs = i;
        }
        desc_buf[len++] = 0
            | (0<<31) //last_dim
            | (((tag=='DEDR' || tag=='DEQW') && cmd->t[i]=='A' ? 4 : 0) << 28) //a_dim (4)
            | ((tag=='DEDR' && cmd->t[i]=='J' && dj!=djp ? 2 : 0) << 28) //j_dim (2)
            | ((tag=='DEDR' && cmd->t[i]=='I' && di!=dip ? 1 : 0) << 28) //i_dim (1)
            | ((tag=='DEDR' && cmd->t[i]=='J' && dj==djp ? 6 : 0) << 28) //jp_dim (6)
            | ((tag=='DEDR' && cmd->t[i]=='I' && di==dip ? 5 : 0) << 28) //ip_dim (5)
            | cmd->n[i];
        s = cmd->s[i]; //s
        for (j=cmd->nld; j<i; j++) s -= (int)(cmd->n[j]-1)*cmd->s[j]; //s'
        if (tag == 'DEDR' && set_id == 0) {
            wrap->regs.SLC0.DE_REGS.DE_D_R.ndss_idx = 0;
            wrap->regs.SLC0.DE_REGS.DE_D_R.ndss_size = 0;
        }
        if (tag == 'DEDR' && set_id == 0 && wrap->cfg.stride == 2)
        {
            int32_t ndss_idx, ndss_size;
            ndss_idx = djs-cmd->nld;
            ndss_size = (wrap->cfg.kernel_top+1+wrap->cfg.kernel_bottom)&1;
            if (wrap->cfg.kernel_top == 0 && wrap->cfg.kernel_bottom == 0) ndss_size = 0;
            wrap->regs.SLC0.DE_REGS.DE_D_R.ndss_idx = ndss_idx;
            wrap->regs.SLC0.DE_REGS.DE_D_R.ndss_size = ndss_size;
            if (ndss_idx >= 0 && i - cmd->nld > ndss_idx+1)
            {
                s += cmd->s[cmd->nld + ndss_idx] * ndss_size;       // adjust s'
            }
            ndss_idx = dis-cmd->nld;
            ndss_size = ((wrap->cfg.kernel_left+1+wrap->cfg.kernel_right)%6)==1;
            if (wrap->cfg.kernel_left == 0 && wrap->cfg.kernel_right == 0) ndss_size = 0;
            wrap->regs.SLC0.DE_REGS.DE_D_R.ndss_idx1 = ndss_idx;
            wrap->regs.SLC0.DE_REGS.DE_D_R.ndss_size1 = ndss_size;
            if (ndss_idx >= 0 && i - cmd->nld > ndss_idx+1)
            {
                s += cmd->s[cmd->nld + ndss_idx] * ndss_size;       // adjust s'
            }
        }
        p = cmd->p[i];
        p =  (p<0)?0:p; //TODO: negative p handled in regs
        assert(p >= 0 && p < 16);
        assert(s>=-(1<<23) && s<(1<<23));
        desc_buf[len++] = (s&((1<<24)-1))|(p<<28);
    }
    assert(len >= 2);
    desc_buf[len-2] |= (1u<<31); //last_dim
    return (int)len;
}

static int _proc_mem_ndl_cmd(torq_wrap_t *wrap, uint32_t tag, int set_id, torq_ndl_cmd_t *cmd, uint32_t ndl_desc_addr, uint32_t ndl_desc_xaddr, uint32_t *desc_len)
{
    int i, idx;
    int32_t s=0, a_adj;
    uint32_t n, len, w_unsigned;
    uint32_t *desc_buf;
    _mem_bdg_ldims_t ldims;
    int32_t in_even, kernel_left_even, kernel_left_odd;
    int32_t jn_even, jn_odd, kernel_top_even, kernel_top_odd, kernel_bottom_even, kernel_bottom_odd;
 
    switch (tag) {
    case 'DEDR': idx = set_id ? 4 : 0; break;
    case 'DEWR': idx = 1; break;
    case 'DEBR': idx = set_id ? 5 : 2; break;
    case 'DEQW': idx = 3; break;
    default: return 0;
    }
    
    // for stride=2, kernel size for even/odd planes
    if (wrap->cfg.stride == 2)
    {
        in_even = (wrap->cfg.kernel_left + 1 + wrap->cfg.kernel_right + 1) >> 1;
        kernel_left_even = (wrap->cfg.kernel_left - wrap->cfg.stride_offset + 1) >> 1;
        kernel_left_odd = wrap->cfg.kernel_left - wrap->cfg.stride_offset - kernel_left_even;
        // subkernel must contain center tap
        assert(kernel_left_even >= 0 && in_even - kernel_left_even - 1 >= 0);
        int32_t in_odd = (wrap->cfg.kernel_left + 1 + wrap->cfg.kernel_right) >> 1;
        if (in_odd > 0)
        {
        assert(kernel_left_odd >= 0 && in_odd - kernel_left_odd - 1 >= 0);
        }

        jn_even = (wrap->cfg.kernel_top + 1 + wrap->cfg.kernel_bottom + 1) >> 1;
        jn_odd = (wrap->cfg.kernel_top + 1 + wrap->cfg.kernel_bottom) >> 1;
        kernel_top_even = (wrap->cfg.kernel_top - wrap->cfg.stride_offset + 1) >> 1;
        kernel_bottom_even = jn_even - 1 - kernel_top_even;
        kernel_top_odd = wrap->cfg.kernel_top - wrap->cfg.stride_offset - kernel_top_even;
        kernel_bottom_odd = jn_odd - 1 - kernel_top_odd;
        assert(kernel_top_even >= 0 && kernel_bottom_even >= 0);
        if (jn_odd > 0)
        {
        assert(kernel_top_odd >= 0 && kernel_bottom_odd >= 0);
        }
    }
    assert(wrap->xn); //'REF' must be the first NDL descriptor
    assert(cmd->nhd);
    assert(wrap->ndl_desc_len[idx]+cmd->nhd*2+1 < NDL_DESC_MAX_LEN_PER_TASK);

    if (wrap->cfg.w_format == 0) w_unsigned = wrap->cfg.de_w_unsigned; //TODO: obsolete
    else                         w_unsigned = (wrap->cfg.w_format == 'UI'); //TODO: floating point formats below 16-bit  not supported yet

    if (wrap->ndl_desc_len[idx] == 0) {
        assert(ndl_desc_addr  != TORQ_LADDR_APPEND);
        assert(ndl_desc_xaddr != TORQ_XADDR_APPEND);
    }
    else {
        assert(ndl_desc_addr  == TORQ_LADDR_APPEND); //TODO: to support link mode
        assert(ndl_desc_xaddr == TORQ_XADDR_APPEND || ndl_desc_xaddr == TORQ_XADDR_NONE); //TODO: can be relaxed with DMA support
    }
    if (ndl_desc_addr != TORQ_LADDR_APPEND) wrap->ndl_desc_addr[idx] = ndl_desc_addr;
    assert(!(wrap->ndl_desc_addr[idx]>>24));
    if (ndl_desc_xaddr != TORQ_XADDR_APPEND) wrap->ndl_desc_xaddr[idx] = ndl_desc_xaddr;

    desc_buf = &wrap->ndl_desc_buf[idx][wrap->ndl_desc_len[idx]];
    wrap->ndl_desc_last_hdr[idx] = desc_buf;
    a_adj = 0;
    if (tag == 'DEDR' && set_id == 0)
    {
        if (wrap->cfg.stride == 2)
        {
            a_adj = kernel_left_even + kernel_top_even * wrap->xn;
        }
        else // stride = 1
        {
            a_adj = wrap->cfg.kernel_left + wrap->cfg.kernel_top * wrap->xn;
        }
        if (cmd->t[0] == 'B' && cmd->n[0] == 2)
        {
            a_adj = a_adj * 2;
        }
    }
    len = _mem_ndl_desc_gen(wrap, tag, set_id, cmd->base_addr-a_adj, cmd, 0, desc_buf);
    wrap->ndl_desc_len[idx] += len;
    if (!len) return 0;

    _parse_mem_bdg_ldims(wrap, cmd, &ldims);
    n = ldims.bn * ldims.dn * ldims.gn;

    switch (tag) {
    case 'DEDR':
    if (set_id == 0) { //DEDR0
        assert(ldims.au_is_bit==0); // byte address only
        assert(ldims.bn==1 || ldims.bn==2 || ldims.bn==4); // only B1,B2,B4 are supported
        assert(ldims.d_is_mergeable); // D must be contiguous
        assert(ldims.gn==1 || ldims.gn==2 || ldims.gn==4); // only G1,G2,G4 are supported
        assert((ldims.gs&(ldims.bn-1))==0); // G stride is B.n-aligned
        assert(ldims.gs>=-(1<<23) && ldims.gs<(1<<23)); // s24
        assert(n>=1 && n<=TORQ_DBUS_WIDTH);

        wrap->regs.SLC0.DE_REGS.DE_D_R.simd_size = (ldims.gn==1?0:ldims.gn==2?1:ldims.gn==4?2:0);
        wrap->regs.SLC0.DE_REGS.DE_D_R.simd_strd = ldims.gs;
        wrap->regs.SLC0.DE_REGS.DE_D_R.w_size = ldims.dn*ldims.bn;
        int step_adj = wrap->cfg.kernel_left + wrap->cfg.kernel_right;
        if (wrap->cfg.stride == 2)
        {
            step_adj = ((step_adj + 1 + 1)>>1) - 1;
        }
        step_adj = step_adj > 2 ? 2 : step_adj;
	    /*
        if (ldims.dn >= 64 && ldims.dn <= 66)       assert(ldims.dn - step_adj == 64);
        else if (ldims.dn >= 32 && ldims.dn <= 34)  assert(ldims.dn - step_adj == 32);
        else if (ldims.dn >= 16 && ldims.dn <= 18)  assert(ldims.dn - step_adj == 16);
        else if (ldims.dn >= 8 && ldims.dn <= 10)   assert(ldims.dn - step_adj == 8);
        else if (ldims.dn >= 4 && ldims.dn <= 6)    assert(ldims.dn - step_adj == 4);
        else assert(0);
	    */
        wrap->regs.SLC0.DE_REGS.DE_D_R.w_step = (ldims.dn-step_adj)*ldims.bn;
        wrap->regs.SLC0.DE_REGS.DE_D_R.pix_size = (ldims.bn == 2 ? 1 : 0);

        wrap->regs.SLC0.DE_REGS.DE_D_R.desc = wrap->ndl_desc_addr[idx];
        if (cmd->sync_mode != 0) {
            wrap->regs.SLC0.DE_REGS.DE_D_R.simd_mode = (cmd->sync_mode=='P');
            wrap->regs.SLC0.DE_REGS.DE_D_R.ignt_id = 0;
        }
        wrap->regs.SLC0.DE_REGS.DE_D_R.sync_idx = cmd->sync_nhd==0?0:cmd->sync_nhd-1;
        wrap->regs.SLC0.DE_REGS.DE_D_R.xy_mode = 0; //TODO: to support XY mode
        wrap->regs.SLC0.DE_REGS.DE_D_R.x_size = wrap->xn;
        {
            uint64_t init_mask = 0;
            uint32_t xn_mod, xn = wrap->xn;
            for (i=0; i<36; i++) {
                xn_mod = (18*xn+i-18)%xn;
                init_mask |=((xn_mod==(xn-1)) ? (1LL<<i) : 0);
            }
            wrap->regs.SLC0.DE_REGS.DE_D_R.mask_lo  = (uint32_t)init_mask;
            wrap->regs.SLC0.DE_REGS.DE_D_R.mask_hi  = (uint32_t)(init_mask>>32);
        }
        wrap->regs.SLC0.DE_REGS.DE_D_R.x_init = -(wrap->cfg.kernel_left); //TODO: X/A must be the lowest dim for non-1x1 kernel
        if (wrap->cfg.stride == 2)
        {
            wrap->regs.SLC0.DE_REGS.DE_D_R.x_init = -kernel_left_even;
        }
        //assert(wrap->cfg.pad_left<=1 && wrap->cfg.pad_right<=1); //limitation in mask-based h/w x-padding
        assert(wrap->cfg.pad_left == 0 || wrap->cfg.pad_left == 1);
        assert(wrap->cfg.pad_right == 0 || wrap->cfg.pad_right == 1);
        wrap->regs.SLC0.DE_REGS.DE_D_R.xmask_en = (wrap->cfg.pad_left || wrap->cfg.pad_right);
        wrap->regs.SLC0.DE_REGS.DE_D_R.ymask_en = //(wrap->cfg.pad_left || wrap->cfg.pad_right) ||
                                                  (wrap->cfg.pad_top || wrap->cfg.pad_bottom); // TODO: FIXME

        //assert(wrap->cfg.pad_top >= 0 && wrap->cfg.pad_top <= wrap->cfg.kernel_top);
        //assert(wrap->cfg.pad_bottom >= 0 && wrap->cfg.pad_bottom <= wrap->cfg.kernel_bottom);
        // Only support full valid padding or full same padding
        assert(wrap->cfg.pad_top == 0 || wrap->cfg.pad_top == wrap->cfg.kernel_top);
        assert(wrap->cfg.pad_bottom == 0 || wrap->cfg.pad_bottom == wrap->cfg.kernel_bottom);
        {
            int32_t kernel_h = wrap->cfg.kernel_top + wrap->cfg.kernel_bottom + 1;
            int32_t yj_min = wrap->cfg.pad_top; // y-pad if y+j < yj_min
            int32_t yj_max = wrap->yn + kernel_h - 2 - wrap->cfg.pad_bottom; // y-pad if y+j > yj_max
            wrap->regs.SLC0.DE_REGS.DE_D_R.ypad_max = (yj_max+1)*wrap->xn + wrap->cfg.kernel_left - 1; //TODO: kernel or pad parameters?
            wrap->regs.SLC0.DE_REGS.DE_D_R.ypad_min = MAX((int32_t)(yj_min*wrap->xn + wrap->cfg.kernel_left), 0);
            wrap->regs.SLC0.DE_REGS.DE_D_R.stride = 0;
            wrap->regs.SLC0.DE_REGS.DE_D_R.xi_dist = 0;     
            wrap->regs.SLC0.DE_REGS.DE_D_R.yi_dist = 0; 
            wrap->regs.SLC0.DE_REGS.DE_D_R.ye_dist = 0;
            if (wrap->cfg.stride == 2)
            {
                yj_min = (wrap->cfg.pad_top > 0) ? kernel_top_even : 0;
                yj_max = wrap->yn + jn_even - 2 - (wrap->cfg.pad_bottom > 0 ? kernel_bottom_even : 0);
                wrap->regs.SLC0.DE_REGS.DE_D_R.ypad_max = (yj_max + 1) * wrap->xn + kernel_left_even - 1;
                wrap->regs.SLC0.DE_REGS.DE_D_R.ypad_min = MAX((int32_t)(yj_min * wrap->xn + kernel_left_even), 0);
                wrap->regs.SLC0.DE_REGS.DE_D_R.stride = 1;
                wrap->regs.SLC0.DE_REGS.DE_D_R.xi_dist = kernel_left_even - kernel_left_odd;
                wrap->regs.SLC0.DE_REGS.DE_D_R.yi_dist = kernel_top_odd - kernel_top_even;
                wrap->regs.SLC0.DE_REGS.DE_D_R.ye_dist = jn_odd - jn_even - (wrap->cfg.pad_bottom > 0 ?  kernel_bottom_odd - kernel_bottom_even : 0);
            }
        }
        break;
    }
    else { //DEDR1
        assert(ldims.au_is_bit==0); // byte address only
        assert(ldims.bn==1 || ldims.bn==2 || ldims.bn==4); // only B1,B2,B4 are supported
        assert(ldims.d_is_mergeable); // D must be contiguous
        assert(ldims.gn==1); // G is not supported
        assert(n>=1 && n<=TORQ_DBUS_WIDTH);
        wrap->regs.SLC0.DE_REGS.DE_D_RX.desc = wrap->ndl_desc_addr[idx];
        wrap->regs.SLC0.DE_REGS.DE_D_RX.w_size = n;
        wrap->regs.SLC0.DE_REGS.DE_D_RX.enable = 1;
        if (cmd->sync_mode != 0) {
            wrap->regs.SLC0.DE_REGS.DE_D_R.simd_mode = (cmd->sync_mode=='P');
            wrap->regs.SLC0.DE_REGS.DE_D_R.ignt_id = 1;
        }
        wrap->regs.SLC0.DE_REGS.DE_D_RX.sync_idx = cmd->sync_nhd==0?0:cmd->sync_nhd-1;
        break;
    }
    case 'DEWR':
        assert(set_id==0);
        if (ldims.au_is_bit) { // address unit is bit
            assert(ldims.bn==1 || ldims.bn==2 || ldims.bn==4 || ldims.bn==6); // only b1,b2,b4,b6 are supported
            wrap->regs.SLC0.DE_REGS.DE_W_R.d_cfmt = (ldims.bn>>1) | (w_unsigned<<2);
            wrap->regs.SLC0.DE_REGS.DE_W_R.d_comp = 1;
            n /= ldims.bn; // weights are zero- or sign-extended to 8-bit
        }
        else { // address unit is byte
            assert(ldims.bn==1 || ldims.bn==2 || ldims.bn==4); // only B1,B2,B4 are supported
        }
        assert(ldims.d_is_mergeable); // D must be contiguous
        assert(ldims.gn==1); // G is not supported
        assert(n>=1 && n<=TORQ_WBUS_WIDTH);
        wrap->regs.SLC0.DE_REGS.DE_W_R.w_size = n==32?0:n;
        wrap->regs.SLC0.DE_REGS.DE_W_R.desc = wrap->ndl_desc_addr[idx];
        break;
    case 'DEBR':
    if (set_id == 0) { //DEBR0
        assert(ldims.au_is_bit==0); // byte address only
        assert(ldims.d_is_mergeable); // D is contiguous
        assert(ldims.gn==1); // G is not supported
        assert(n>=1 && n<=TORQ_BBUS_WIDTH);
        assert(n==4 || n==8 || n==16 || n==32); //TODO: relax this?
        wrap->regs.SLC0.DE_REGS.DE_B_R.w_size = n==32?0:n;
        wrap->debr_ldim_sz = n; //for assertion check only
        wrap->regs.SLC0.DE_REGS.DE_B_R.desc = wrap->ndl_desc_addr[idx];
        wrap->regs.SLC0.DE_REGS.DE_B_R.disabled = 0;
        if (cmd->sync_mode != 0) {
            assert(cmd->sync_mode=='R');
            wrap->regs.SLC0.DE_REGS.DE_B_R.ignt_id = 0;
        }
        wrap->regs.SLC0.DE_REGS.DE_B_R.sync_idx = cmd->sync_nhd==0?0:cmd->sync_nhd-1;
        break;
    }
    else { //DEBR1
        assert(ldims.au_is_bit==0); // byte address only
        assert(ldims.d_is_mergeable); // D is contiguous
        assert(ldims.gn==1); // G is not supported
        assert(n>=1 && n<=TORQ_BBUS_WIDTH);
        wrap->regs.SLC0.DE_REGS.DE_B_RX.w_size = n==32?0:n;
        wrap->regs.SLC0.DE_REGS.DE_B_RX.desc = wrap->ndl_desc_addr[idx];
        wrap->regs.SLC0.DE_REGS.DE_B_RX.enable = 1;
        if (cmd->sync_mode != 0) {
            assert(cmd->sync_mode=='R');
            wrap->regs.SLC0.DE_REGS.DE_B_R.ignt_id = 1;
        }
        wrap->regs.SLC0.DE_REGS.DE_B_RX.sync_idx = cmd->sync_nhd==0?0:cmd->sync_nhd-1;
        wrap->regs.SLC0.DE_REGS.DE_Q_W.sct_mode = 1;
        break;
    }
    case 'DEQW':
        assert(set_id==0);
        assert(ldims.au_is_bit==0); // byte address only
        assert(ldims.bn==1 || ldims.bn==2 || ldims.bn==4); // only B1,B2,B4 are supported
        assert(n>=1 && n<=TORQ_QBUS_WIDTH);
        wrap->regs.SLC0.DE_REGS.DE_Q_W.w_size = n==64?0:n;
        wrap->deqw_bn = ldims.bn;

        wrap->regs.SLC0.DE_REGS.DE_Q_W.desc = wrap->ndl_desc_addr[idx];
        wrap->regs.SLC0.DE_REGS.DE_Q_W.secv_en = 0;
        wrap->regs.SLC0.DE_REGS.DE_Q_W.split_en = 0;
        wrap->regs.SLC0.DE_REGS.DE_Q_W.split_strd = 0;
        if (wrap->cfg.act_disable == 0xf) {
            // if ACT is disabled, but DEQW is enabled, DEQW will work in the 'fill' mode.
            wrap->regs.SLC0.DE_REGS.DE_Q_W.fill_en = 1;
            //assert(wrap->cfg.pad_value <= 127 && wrap->cfg.pad_value >= -128);
            wrap->regs.SLC0.DE_REGS.DE_Q_W.fill_val = wrap->cfg.pad_value;
            wrap->regs.SLC0.DE_REGS.DE_Q_W.pix_size = ldims.bn == 1 ? 0 : 1;
        }

        // map {BnDnGn} to 3 output modes:
        if (ldims.d_is_mergeable && ldims.g_is_mergeable) { // 1 address, no split, {BnGnGn} with natural D.s and G.s
            s = 0;
        }
        else if (ldims.d_is_mergeable && !ldims.g_is_mergeable && ldims.gn == 2) { // 2 addresses, half-half split, {BnDnG2} with natural D.s and unnatrual G.s
            assert(n==16 || n==32 || n==64);
            assert(cmd->nsd==0);
            wrap->regs.SLC0.DE_REGS.DE_Q_W.split_en = 1;
            wrap->regs.SLC0.DE_REGS.DE_Q_W.split_type = 1;
            s = ldims.gs;
        }
        else if ((ldims.dn == 2 && !ldims.d_is_mergeable) && (ldims.gn == 1 || ldims.gs == ldims.bn)) { // 2 address, even-odd split, {BnD2Gn} with unnatrual Ds and Gs=Bn
            assert(cmd->nsd==0); //TODO: may need xy view to specify pixel size (if != act output size), or always use xy view to do even-odd split
            wrap->regs.SLC0.DE_REGS.DE_Q_W.split_en = 1;
            wrap->regs.SLC0.DE_REGS.DE_Q_W.split_type = 0;
            wrap->regs.SLC0.DE_REGS.DE_Q_W.pix_size = ldims.bn == 1 ? 0 : 1;
            s = ldims.ds;
        }
        else assert(!"ERROR: invalid DEQW LDIMs");

        if (cmd->nsd) {
            _mem_bxy_sdims_t sdims;
            _parse_mem_bxy_sdims(wrap, cmd, &sdims);
            assert(sdims.bn==1 || sdims.bn==2); // only B1,B2 are supported

            wrap->regs.SLC0.DE_REGS.DE_Q_W.pix_size = sdims.bn==1?0:1;

            // map BXX to 2 output modes:
            if ((sdims.xn[0] == 1 || sdims.xs[0] == sdims.bn) && sdims.xn[1] == 1) { // 1 address, no split
                s = 0;
            }
            else if (sdims.xn[0] == 2 && sdims.xn[1] > 1 && sdims.xs[1] == sdims.bn) { // 2 addresses, even/odd split
                wrap->regs.SLC0.DE_REGS.DE_Q_W.split_en = 1;
                wrap->regs.SLC0.DE_REGS.DE_Q_W.split_type = 0;
                s = sdims.xs[0];
            }
            else assert(!"ERROR: invalid DEQW SDIM: X");
            wrap->regs.SLC0.DE_REGS.DE_Q_W.x_size = sdims.xn[0]*sdims.xn[1]; //TODO: h/w supports x0n==2 && x_size is odd (early termination)

            if (sdims.yn[1]==1) {
                assert(sdims.ys[0]>=-(1<<17) && sdims.ys[0]<(1<<17)); //s18
                wrap->regs.SLC0.DE_REGS.DE_Q_W.yp_step = sdims.ys[0]; //(sdims.ys[0]+1)>>1;
                wrap->regs.SLC0.DE_REGS.DE_Q_W.yh_step = sdims.ys[0]*2;
            }
            else if (sdims.yn[0]==2) {
                assert(sdims.ys[0]>=-(1<<23) && sdims.ys[0]<(1<<23)); //s24
                assert(sdims.ys[1]>=-(1<<17) && sdims.ys[1]<(1<<17)); //s18
                wrap->regs.SLC0.DE_REGS.DE_Q_W.yp_step = sdims.ys[0];
                wrap->regs.SLC0.DE_REGS.DE_Q_W.yh_step = sdims.ys[1];
            }
            else assert(!"ERROR: invalid DEQW SDIM: Y");
            wrap->regs.SLC0.DE_REGS.DE_Q_W.y_size = sdims.yn[0]*sdims.yn[1]; //TODO: h/w supports y0n==2 && y_size is odd (early termination)

            wrap->regs.SLC0.DE_REGS.DE_Q_W.secv_en = 1;
        }

        if (s) {
            //assert((s&7)==0);
            assert(s>=-(1<<23) && s<(1<<23)); //s21+3
            //s /= 8;
            wrap->regs.SLC0.DE_REGS.DE_Q_W.split_strd = s;
        }
        break;
    }
    *desc_len = len<<2;
    return 1;
}

int torq_cfg_begin(void *self, int slc_id, uint32_t cfg_desc_laddr, uint32_t cfg_desc_xaddr)
{
    torq_wrap_t *wrap = (torq_wrap_t *)self;
    xdbg_func_begin();
    if (wrap->new_job) torq_job_cfg_begin(wrap);
    wrap->slc_id = slc_id;
    assert(cfg_desc_laddr != TORQ_LADDR_APPEND); // first laddr
    // compiler{
    //assert((cfg_desc_laddr & 3) == 0);
    assert(cfg_desc_laddr == TORQ_LADDR_NONE || (cfg_desc_laddr & 3) == 0);
    // }compiler
    assert(cfg_desc_xaddr != TORQ_XADDR_APPEND); // first xaddr
    assert(cfg_desc_xaddr == TORQ_XADDR_NONE || (cfg_desc_xaddr & 3) == 0);
    wrap->cfg_desc_addr = cfg_desc_laddr;
    wrap->cfg_desc_xaddr = cfg_desc_xaddr;
    wrap->cfg_desc_len = wrap->cfg_desc_total_len = 0;
    xdbg_func_end();
    return 0;
}

int _torq_task_cfg_end(void *self, int last,  uint32_t nxt_laddr)
{
    torq_wrap_t *wrap = (torq_wrap_t *)self;
    xdbg_func_begin();
    _output_desc(wrap, last, nxt_laddr);
    if (wrap->entry_point == 1) wrap->entry_point = 0;
    xdbg_func_end();
    return 0;
}

int torq_cfg_end(void *self)
{
    torq_wrap_t *wrap = (torq_wrap_t *)self;
    xdbg_func_begin();
    _torq_task_cfg_end(wrap, 1, TORQ_LADDR_NONE);
    xdbg_func_end();
    return (int)(wrap->cfg_desc_total_len<<2);
}

int torq_task_cfg_begin(void *self, torq_cfg_t *cfg, uint32_t cfg_desc_laddr, uint32_t cfg_desc_xaddr)
{
    torq_wrap_t *wrap = (torq_wrap_t *)self;
    int i;
    xdbg_func_begin();
    //assert(cfg_desc_laddr == TORQ_LADDR_APPEND); // relaxed with link mode support
    //assert(cfg_desc_xaddr == TORQ_XADDR_APPEND || cfg_desc_xaddr == TORQ_XADDR_NONE); // relaxed with DMA support
    assert(!(wrap->cfg_desc_xaddr == TORQ_XADDR_NONE && cfg_desc_xaddr == TORQ_XADDR_APPEND));
    assert(!(wrap->cfg_desc_addr == TORQ_XADDR_NONE && cfg_desc_laddr == TORQ_XADDR_APPEND));
    if (_get_task_id(wrap) >= 0) _torq_task_cfg_end(wrap, 0, cfg_desc_laddr);
    _inc_task_id(wrap);
    if (wrap->slc_id < 0 && wrap->entry_point<0) wrap->entry_point = 1;
    if (cfg_desc_laddr != TORQ_LADDR_APPEND) wrap->cfg_desc_addr = cfg_desc_laddr;
    if (cfg_desc_xaddr != TORQ_XADDR_APPEND) wrap->cfg_desc_xaddr = cfg_desc_xaddr;
    //memset(wrap->ndl_desc_len, 0, sizeof(wrap->ndl_desc_len));
    memcpy(&wrap->cfg, cfg, sizeof(torq_cfg_t));
    wrap->xn = 0;
    wrap->debr_ldim_sz = 0;
    if (wrap->slc_id<0) _nss_cfg_desc_init(wrap);
    else _slc_cfg_desc_init(wrap);
    xdbg_func_end();
    return 0; 
}

int torq_task_cfg_end(void *self)
{
    torq_wrap_t *wrap = (torq_wrap_t *)self;
    xdbg_func_begin();
    if (wrap->slc_id<0) _nss_cfg_desc_gen(wrap);
    else _slc_cfg_desc_gen(wrap);
    wrap->cfg_desc_total_len += wrap->cfg_desc_len;
    // cdesc is not the final version yet, so can't be output
    // _torq_task_cfg_end() is deferred to the next torq_task_cfg_begin() or torq_cfg_end()
    xdbg_func_end();
    return (wrap->cfg_desc_len<<2);
}

int torq_ndl_desc_write(void *self, uint32_t tag, int set_id, torq_ndl_cmd_t *cmd, uint32_t ndl_desc_laddr, uint32_t ndl_desc_xaddr)
{
    torq_wrap_t *wrap = (torq_wrap_t *)self;
    int i, r;
    uint32_t desc_len=0;
    int task_id = _get_task_id(wrap);
    r = torq_desc_dump__ndl_info(wrap->dump_ctx, wrap->job_id, task_id, wrap->slc_id, tag, set_id, cmd); assert(r>=0); 
    if (wrap->slc_id < 0) {
        assert(set_id==0);
        r = _nss_ndl_desc_gen(wrap, tag, cmd);
    } else {
        do {
            r = _proc_sw_ndl_desc(wrap, tag, set_id, cmd); if (r) break;
            r = _proc_mem_ndl_cmd(wrap, tag, set_id, cmd, ndl_desc_laddr, ndl_desc_xaddr, &desc_len); if (r) break;
            r = _reg_ndl_desc_gen(wrap, tag, set_id, cmd);
        } while (0);
    }
    assert(r>=0);
    if (!r) assert(!"ERROR: invalid NDL descriptor tag");
    return (int)desc_len;
}

int torq_lram_read(void *self, int slc_id, uint32_t tag, uint32_t addr, uint32_t size, void* data) // data is ignored at compile time
{
    int r;
    torq_wrap_t *wrap = (torq_wrap_t *)self;
    int task_id = _get_task_id(wrap);
    r = torq_desc_dump__data(wrap->dump_ctx, wrap->job_id, wrap->stg_id, task_id, slc_id, 'LRAM', 'OUT', tag, 0, addr, size, 0L); assert(r>=0); 
    return 0;
}

int torq_lram_write(void *self, int slc_id, uint32_t tag, uint32_t addr, uint32_t size, void* data)
{
    int r;
    torq_wrap_t *wrap = (torq_wrap_t *)self;
    int task_id = _get_task_id(wrap);
    r = torq_desc_dump__data(wrap->dump_ctx, wrap->job_id, wrap->stg_id, task_id, slc_id, 'LRAM', 'INP', tag, 0, addr, size, data); assert(r>=0); 
    return 0;
}

int torq_xram_read(void *self, int slc_id, uint32_t tag, uint32_t addr, uint32_t size, void* data) // data is ignored at compile time
{
    int r;
    torq_wrap_t *wrap = (torq_wrap_t *)self;
    int task_id = _get_task_id(wrap);
    r = torq_desc_dump__data(wrap->dump_ctx, wrap->job_id, wrap->stg_id, task_id, slc_id, 'XRAM', 'OUT', tag, 0, addr, size, 0L); assert(r>=0); 
    return 0;
}

int torq_xram_write(void *self, int slc_id, uint32_t tag, uint32_t addr, uint32_t size, void* data)
{
    int r;
    torq_wrap_t *wrap = (torq_wrap_t *)self;
    int task_id = _get_task_id(wrap);
    r = torq_desc_dump__data(wrap->dump_ctx, wrap->job_id, wrap->stg_id, task_id, slc_id, 'XRAM', 'INP', tag, 0, addr, size, data); assert(r>=0); 
    return 0;
}

int torq_get_bitstream(void *self, torq_bitstream_segment_t** bitstream)
{
    torq_wrap_t *wrap = (torq_wrap_t *)self;
    *bitstream = wrap->bitstream;
    return 0;
}

int torq_run(void *self)
{
    torq_wrap_t *wrap = (torq_wrap_t *)self;
    int r;
    xdbg_func_begin();
    r = torq_job_cfg_end(wrap); assert(r>=0);
    // run job (no operations in compile time)
    wrap->new_job = 1;
    xdbg_func_end();
    return 0;
}
