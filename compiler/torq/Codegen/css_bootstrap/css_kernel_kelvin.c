#include <stdint.h>

#include "css_sw/common/css_sw_reg_inc.h"
#include "iree/hal/local/executable_library.h"

static inline void halt() {
    // this is a special kelvin instruction that halts the cpu
    asm volatile(".word 0x08000073");
}

volatile uint32_t *css_mbox_start = (volatile uint32_t *)0x401fc000;

static inline uint32_t readMailbox(uint32_t id) { return *(css_mbox_start + id); }

static inline void writeMailbox(uint32_t id, uint32_t value) { *(css_mbox_start + id) = value; }

static inline void memset(void *ptr, int value, size_t num) {
    for (size_t i = 0; i < num; i++) {
        ((char *)ptr)[i] = (char)value;
    }
}

void main(
    iree_hal_executable_environment_v0_t *environment,
    iree_hal_executable_dispatch_state_v0_t *dispatch_state,
    iree_hal_executable_workgroup_state_v0_t *workgroup_state
);

void css_sw_main(void *cpu) {
    iree_hal_executable_environment_v0_t environment;
    iree_hal_executable_dispatch_state_v0_t dispatch_state;
    iree_hal_executable_workgroup_state_v0_t workgroup_state;

    memset(&environment, 0, sizeof(environment));
    memset(&dispatch_state, 0, sizeof(dispatch_state));
    memset(&workgroup_state, 0, sizeof(workgroup_state));

    uint32_t *args = (uint32_t *)readMailbox(0);
    dispatch_state.binding_count += args[0];
    dispatch_state.binding_ptrs = (void **)&args[1];

    main(&environment, &dispatch_state, &workgroup_state);

    writeMailbox(0, 0xFFFFFFFF);

    halt();
}
