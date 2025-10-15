// Copyright 2025 Synaptics Incorporated. All rights reserved.
// Created:  02/13/2025, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

#if (defined(TORQ_CSS_SW) || defined(TORQ_BEH_CSS_SW))

#include <stdint.h>
#include "css_sw_reg_inc.h"
#include "css_sw_inc.h"

void css_sw_task(void *cpu);

void css_sw_main(void *cpu)
{
#ifndef TORQ_BEH_CSS_SW
    do { //argument cpu is not used
#endif
        css_sw__reg_write(cpu, RA_(CSS,MBX_IRQ_0), 0); // clear NSS2CSS IRQ
        css_sw_task(cpu);
        css_sw__reg_write(cpu, RA_(CSS,MBX_IRQ_1), 1); // raise CSS2NSS IRQ
        css_sw__end_sleep(cpu);
#ifndef TORQ_BEH_CSS_SW
    } while (1);
#endif
}

#endif
