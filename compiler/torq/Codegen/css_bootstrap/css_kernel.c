#include "css_sw/common/css_sw_reg_inc.h"
#include "iree/hal/local/executable_library.h"
#include <stdint.h>

#ifdef CSS_HW_KELVIN

#define print(x)
#define printInt(x)

static inline void halt() {
    // this is a special kelvin instruction that halts the cpu
    asm volatile(".word 0x08000073");
}
#endif
#ifdef CSS_HW_QEMU

static inline void print(char *data) {
    int uart = 0x10000000;

    for (int i = 0; data[i] != '\0'; i++) {
        *(volatile char *)(uart) = data[i];
    }
}

static inline void printInt(unsigned int val) {
    int uart = 0x10000000;
    char hexChars[] = "0123456789ABCDEF";

    *(volatile char *)(uart) = '0';
    *(volatile char *)(uart) = 'x';

    for (int pos = 7; pos >= 0; pos--) {
        *(volatile char *)(uart) = hexChars[(val >> (pos * 4)) & 0xF];
    }
}

static inline void halt() {

    // ensure the whole memory is written before halting
    __asm__ volatile("" ::: "memory");

    // shutdown emulation using the sifive_test device
    asm volatile("li a0, 0x5555\n"   // Exit code (0x3333 fail 0x5555 success)"
                 "li a1, 0x100000\n" // Address of the exit device"
                 "sw a0, 0(a1)\n"    // Write the exit code to the device"
    );

    // Wait for the emulation to exit
    while (1) {
    };
}

#endif

#ifdef ENABLE_VEC
static inline void riscv_enable_vec(void) {
    unsigned long mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));

    // Set VS=Initial (01)
    mstatus |= (1UL << 9); // Set VS=01 (Initial)

    asm volatile("csrw mstatus, %0" ::"r"(mstatus));
}
#endif

#ifdef ENABLE_FP
static inline void riscv_enable_fp(void) {
    unsigned long mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    // Set FS=Initial (bit13=1, bit14=0).
    mstatus |= (1UL << 13); // FS=01
    asm volatile("csrw mstatus, %0" ::"r"(mstatus));
    // Clear FCSR: fflags=0, frm=round-to-nearest-even (0), fcsr=0
    asm volatile("csrw fcsr, zero");
}
#endif

extern volatile uint32_t css_regs[REG_SIZE__TORQ_CSS_REGS];

static inline uint32_t readCssReg(uint32_t addr) { return css_regs[addr / 4]; }

static inline void writeCssReg(uint32_t addr, uint32_t data) { css_regs[addr / 4] = data; }

static inline void clearIrq0() { writeCssReg(REG_ADDR__TORQ_CSS_REGS_MBX_IRQ_0, 0); }

int mailboxAddress[4] = {
    REG_ADDR__TORQ_CSS_REGS_MBX_DAT_0, REG_ADDR__TORQ_CSS_REGS_MBX_DAT_1,
    REG_ADDR__TORQ_CSS_REGS_MBX_DAT_2, REG_ADDR__TORQ_CSS_REGS_MBX_DAT_3
};

static inline int readMailbox(int id) { return readCssReg(mailboxAddress[id]); }

static inline void writeMailbox(int id, uint32_t value) { writeCssReg(mailboxAddress[id], value); }

static inline void raiseIrq1() { writeCssReg(REG_ADDR__TORQ_CSS_REGS_MBX_IRQ_1, 1); }

void main(
    iree_hal_executable_environment_v0_t *environment,
    iree_hal_executable_dispatch_state_v0_t *dispatch_state,
    iree_hal_executable_workgroup_state_v0_t *workgroup_state
);

static inline void memset(void *ptr, int value, size_t num) {
    for (size_t i = 0; i < num; i++) {
        ((char *)ptr)[i] = (char)value;
    }
}

void trap_handler(void) {
    unsigned int mcause;
    asm volatile("csrr %0, mcause" : "=r"(mcause));

    unsigned int exception_code = mcause & 0xFFF;

    unsigned int mtval;
    asm volatile("csrr %0, mtval" : "=r"(mtval));

    print("Writing failure to mailbox...\n");
    writeMailbox(0, mcause);
    writeMailbox(1, mtval);

    // reset the trap handler
    asm volatile("csrw mtvec, %0" ::"r"(0));

    // halt cpu
    halt();
}

static inline void setup_trap_handler() {
    unsigned int handler_addr = (unsigned int)trap_handler;
    asm volatile("csrw mtvec, %0" ::"r"(handler_addr));
}

extern uint32_t __itcm_start;
extern uint32_t __stack_start;

void css_sw_main(void *cpu) {

    setup_trap_handler();

#ifdef ENABLE_FP
    print("Enabling FP support...\n");
    riscv_enable_fp();
    print("FP support enabled.\n");
#endif

#ifdef ENABLE_VEC
    print("Enabling Vector support...\n");
    riscv_enable_vec();
    print("Vector support enabled.\n");
#endif

    print("Clearing interrupt...\n");

    clearIrq0(); // clear NSS2CSS IRQ

    int32_t disabledValue = (uint32_t)&__itcm_start;

    print("Preparing parameters...\n");

    iree_hal_executable_environment_v0_t environment;
    iree_hal_executable_dispatch_state_v0_t dispatch_state;
    iree_hal_executable_workgroup_state_v0_t workgroup_state;

    memset(&environment, 0, sizeof(environment));
    memset(&dispatch_state, 0, sizeof(dispatch_state));
    memset(&workgroup_state, 0, sizeof(workgroup_state));

    uint32_t *args = (uint32_t *)readMailbox(0);

    int bindingCount = args[0];

    print("Bindings count: ");
    printInt(args[0]);
    print("\n");

    dispatch_state.binding_count += args[0];
    dispatch_state.binding_ptrs = (void **)&args[1];

    print("Number of arguments: ");
    printInt(dispatch_state.binding_count);
    print("\n");

    for (int i = 0; i < dispatch_state.binding_count; i++) {
        print("Argument ");
        printInt(i);
        print(": ");
        printInt((int)dispatch_state.binding_ptrs[i]);
        print("\n");
    }

    print("Stack starts at ");
    printInt((int)&__stack_start);
    print("\n");

    __stack_start = 0xDEADBEEF; // canary value to detect stack overflows

    print("Executing main...\n");
    main(&environment, &dispatch_state, &workgroup_state);

    if (__stack_start != 0xDEADBEEF) {
        print("Stack overflow detected!\n");
        writeMailbox(0, 0xFFFFFFFE); // special value to indicate stack overflow
    }
    else {
        print("main executed successfully\n");

        print("Writing success to mailbox...\n");
        writeMailbox(0, 0xFFFFFFFF);
    }

    print("Raising interrupt...\n");
    raiseIrq1();

    print("Halting...\n");
    halt();
}