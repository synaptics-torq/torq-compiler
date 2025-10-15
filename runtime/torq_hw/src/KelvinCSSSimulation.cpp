#include "KelvinCSSSimulation.h"

#define TORQ_BEH_CSS_SW

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "css_sw/common/css_sw_reg_inc.h"
#include "css_sw_inc.h"

#ifndef RT_BUILD

#include "iree/base/internal/flags.h"

IREE_FLAG(bool, torq_kelvin_verbose, 0, "Enable verbose logs for Kelvin CSS simulation")
IREE_FLAG(int32_t, torq_kelvin_debug, 0, "Enable gdb when invoking Kelvin for CSS emulation")

#else

static bool FLAG_torq_kelvin_verbose = 0;
static int32_t FLAG_torq_kelvin_debug = 0;

#endif

#define LOG(...)                                                                                   \
    do {                                                                                           \
        if (FLAG_torq_kelvin_verbose) {                                                           \
            printf(__VA_ARGS__);                                                                   \
        }                                                                                          \
    } while (0)

// these values are defined by the kelvin simulator definition
static const uint32_t kelvin_itcm_start = 0x00000000;
static const uint32_t kelvin_itcm_size = 0x2000; // 8KB
static const uint32_t kelvin_dtcm_start = 0x00010000;
static const uint32_t kelvin_dtcm_size = 0x8000; // 32KB

static const uint32_t mbox_reg_offset[4] = {
    REG_ADDR__TORQ_CSS_REGS_MBX_DAT_0, REG_ADDR__TORQ_CSS_REGS_MBX_DAT_1,
    REG_ADDR__TORQ_CSS_REGS_MBX_DAT_2, REG_ADDR__TORQ_CSS_REGS_MBX_DAT_3
};

static const uint32_t irqRegsOffets[4] = {
    REG_ADDR__TORQ_CSS_REGS_MBX_IRQ_0, REG_ADDR__TORQ_CSS_REGS_MBX_IRQ_1,
    REG_ADDR__TORQ_CSS_REGS_MBX_IRQ_2, REG_ADDR__TORQ_CSS_REGS_MBX_IRQ_3
};

static void dump_riscv_fault(uint32_t mcode, uint32_t mtval) {
    fprintf(stderr, "RISC-V Exception: mcode=0x%08x, mtval=0x%08x\n", mcode, mtval);

    // Decode common RISC-V exception codes (see RISC-V Privileged Spec Table 3.6)
    switch (mcode) {
    case 0x0:
        fprintf(stderr, "Instruction address misaligned. Faulting address: 0x%08x\n", mtval);
        break;
    case 0x1:
        fprintf(stderr, "Instruction access fault. Faulting address: 0x%08x\n", mtval);
        break;
    case 0x2:
        fprintf(stderr, "Illegal instruction. Faulting instruction: 0x%08x\n", mtval);
        break;
    case 0x3:
        fprintf(stderr, "Breakpoint at address: 0x%08x\n", mtval);
        break;
    case 0x4:
        fprintf(stderr, "Load address misaligned. Faulting address: 0x%08x\n", mtval);
        break;
    case 0x5:
        fprintf(stderr, "Load access fault. Faulting address: 0x%08x\n", mtval);
        break;
    case 0x6:
        fprintf(stderr, "Store/AMO address misaligned. Faulting address: 0x%08x\n", mtval);
        break;
    case 0x7:
        fprintf(stderr, "Store/AMO access fault. Faulting address: 0x%08x\n", mtval);
        break;
    case 0x8:
        fprintf(stderr, "Environment call from U-mode.\n");
        break;
    case 0x9:
        fprintf(stderr, "Environment call from S-mode.\n");
        break;
    case 0xb:
        fprintf(stderr, "Environment call from M-mode.\n");
        break;
    default:
        fprintf(stderr, "Unknown or unhandled RISC-V exception code: 0x%08x\n", mcode);
        break;
    }
}

static uint32_t prepare_code(void *cpu, KelvinSimulator *kelvin_sim) {
    uint32_t code_reg_addr = css_sw__reg_read(cpu, RA_(CPU, PCSTART));
    void *code_host_addr = css_sw__tcm_a2p(cpu, code_reg_addr);

    LOG("code entry point: received reg 0x%08x, host address 0x%08lx, Kelvin "
        "address 0x%08x\n",
        code_reg_addr, (uintptr_t)code_host_addr, kelvin_itcm_start);

    kelvin_sim->WriteTCM(
        kelvin_itcm_start + code_reg_addr, REG_SIZE__TORQ_CV_ITCM - code_reg_addr,
        static_cast<const char *>(code_host_addr)
    );

    LOG("Load code to kelvin simulator\n");

    return code_reg_addr;
}

static void prepare_data_and_registers(void *cpu, KelvinSimulator *kelvin_sim) {
    // Write out the MBOX values to the Kelvin memory
    KelvinMailbox mbox;
    for (int i = 0; i < 4; i++) {
        mbox.message[i] = css_sw__reg_read(cpu, REG_ADDR__TORQ_CV_CSS_REGS + mbox_reg_offset[i]);
        LOG("mbox_reg[%d] = received reg 0x%08x \n", i, mbox.message[i]);
    }
    kelvin_sim->WriteMailbox(mbox);

    // write out the state of the DTCM
    void *dtcm_host_addr = css_sw__tcm_a2p(cpu, REG_ADDR__TORQ_CV_DTCM);

    kelvin_sim->WriteTCM(
        kelvin_dtcm_start, kelvin_dtcm_size, static_cast<const char *>(dtcm_host_addr)
    );
    LOG("Load data and registers to kelvin simulator\n");
}

static void run_cpu_kelvin_binary_ex(void *cpu, KelvinSimulator *kelvin_sim, bool re_run);

static void process_status_code(void *cpu, KelvinSimulator *kelvin_sim, bool re_run) {
    KelvinMailbox mbox = kelvin_sim->ReadMailbox();
    uint32_t mcode = mbox.message[0];

    if (mcode != 0xFFFFFFFF) {
        uint32_t mtval = mbox.message[1];

        dump_riscv_fault(mcode, mtval);

        // Re-run here
        if (!re_run) {
            fprintf(stderr, "Re-run task \n");
            run_cpu_kelvin_binary_ex(cpu, kelvin_sim, true);
        }

        fprintf(stderr, "Aborting execution");
        abort();
    }
}

static void read_back_memory_and_registers(void *cpu, KelvinSimulator *kelvin_sim) {
    void *dtcm_host_addr = css_sw__tcm_a2p(cpu, REG_ADDR__TORQ_CV_DTCM);

    // copy back the state of the DTCM after the simulation
    kelvin_sim->ReadTCM(kelvin_dtcm_start, kelvin_dtcm_size, static_cast<char *>(dtcm_host_addr));

    // Manually read back the state of the IRQ registers for now
    uint32_t irqAfterValue[4] = {0, 1, 0, 0};
    for (int i = 0; i < 4; i++) {
        uint32_t irqBeforeValue =
            css_sw__reg_read(cpu, REG_ADDR__TORQ_CV_CSS_REGS + irqRegsOffets[i]);
        if (irqAfterValue[i] != irqBeforeValue) {
            LOG("updating irqReg[%d] = 0x%08x (was 0x%08x)\n", i, irqAfterValue[i], irqBeforeValue);
            css_sw__reg_write(
                cpu, REG_ADDR__TORQ_CV_CSS_REGS + irqRegsOffets[i], irqAfterValue[i]
            );
        }
    }
}

static void execute_program(KelvinSimulator *kelvin_sim, uint32_t start_pc) {
    LOG("------ kelvin output start ------\n");
    kelvin_sim->Run(start_pc);
    if (kelvin_sim->WaitForTermination(500000)) {
        LOG("Halted \n");
    }
    else {
        LOG("Didn't halt \n");
    }
    LOG("------ kelvin output end------\n");
}

static void run_cpu_kelvin_binary_ex(void *cpu, KelvinSimulator *kelvin_sim, bool re_run) {
    uint32_t start_pc = prepare_code(cpu, kelvin_sim);

    prepare_data_and_registers(cpu, kelvin_sim);

    // Execute program
    execute_program(kelvin_sim, start_pc);

    process_status_code(cpu, kelvin_sim, re_run);

    read_back_memory_and_registers(cpu, kelvin_sim);
}

void run_cpu_kelvin_binary(void *cpu) {
    KelvinSimulator *kelvin_sim = KelvinSimulator::Create();

    LOG("Starting CPU\n");

    run_cpu_kelvin_binary_ex(cpu, kelvin_sim, false);

    css_sw__end_sleep(cpu);
}
