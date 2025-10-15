// Copyright 2023-2024 Synaptics Incorporated. All rights reserved.
// Created:  11/22/2023, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#include <stdint.h>
#include <string.h>
#include "torq_os.h"
#include "torq_log.h"
#include "torq_sys.h"
#include "torq_mem_list.h"
#include "torq_hw.h"


int torq_run_job(torq_sys_t *sys, const char *inp_dir, const char *out_dir)
{
    FILE *fp;
    int r;
    uint32_t addr = 0;
    char s[1024];

    snprintf(s, sizeof(s)-1, "%s/tv.init.mem.lst", inp_dir);
    r = torq_proc_mem_lst(sys, s, inp_dir, out_dir); xdie(r<0);

    snprintf(s, sizeof(s)-1, "%s/tv.cdesc_addr.txt", inp_dir);
    if (os_exists(s)) {
        fp = fopen(s,"r"); xdie(!fp);
        fscanf(fp, "0x%08x\n", &addr);
        fclose(fp);
    }

    r = torq_hw_set_dump(sys, out_dir); xdie(r<0);
    r = torq_hw_start(sys, addr); xdie(r<0);
    r = torq_hw_wait(sys); xdie(r<0);
    r = torq_hw_end(sys); xdie(r<0);

    snprintf(s, sizeof(s)-1, "%s/tv.exit.mem.lst", inp_dir);
    if (os_exists(s)) {
        r = torq_proc_mem_lst(sys, s, inp_dir, out_dir); xdie(r<0);
    }
    return 0;
}

int torq_main(int argc, const char *argv[])
{
    int i, r;
    const char *idir = "tc";
    const char *odir = "out";
    const char *sdir = 0L;
    char inp_dir[1024], out_dir[1024], cmd[1024];
    torq_sys_t sys_buf, *sys = &sys_buf;

    for (i=1; i<argc; i++) {
        if      (argc > i+1 && !strcmp(argv[i], "-i")) idir = argv[++i];
        else if (argc > i+1 && !strcmp(argv[i], "-o")) odir = argv[++i];
        else if (argc > i+1 && !strcmp(argv[i], "-s")) sdir = argv[++i];
    }
    if (!os_exists(idir)) xdie("ERROR: input dir does not exist");

    r = torq_sys_open(sys); xdie(r<0);

    r = os_mkdir(odir); xdie(r!=0);
    for (i=0;;i++) {
        snprintf(inp_dir, sizeof(inp_dir)-1, "%s/job%d", idir, i);
        if (!os_exists(inp_dir)) break;
        xmsg("@@@BEGIN %s\n", inp_dir);
        snprintf(out_dir, sizeof(out_dir)-1, "%s/job%d", odir, i);
        r = os_mkdir(out_dir); xdie(r!=0);
        if (sdir) {
            snprintf(cmd, sizeof(cmd)-1, "%s/job_init.sh %d", sdir, i);
            r = system(cmd); xdie(r!=0);
        }
        r = torq_run_job(sys, inp_dir, out_dir); xdie(r<0);
        xflush();
        if (sdir) {
            snprintf(cmd, sizeof(cmd)-1, "%s/job_exit.sh %d", sdir, i);
            r = system(cmd); xdie(r!=0);
        }
        xmsg("@@@END %s\n", inp_dir);
    }

    r = torq_sys_close(sys); xdie(r<0);

    return 0;
}
