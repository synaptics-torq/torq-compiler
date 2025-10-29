#include <stdint.h>

#define print(x)
#define printInt(x)

static inline void halt() {
    // this is a special kelvin instruction that halts the cpu
    asm volatile(".word 0x08000073");
}

#include "css_kernel.c.inc"
