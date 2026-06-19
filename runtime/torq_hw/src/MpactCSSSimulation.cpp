#include "MpactCSSSimulation.h"
#include <coralnpu_simulator.h>

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

IREE_FLAG(bool, torq_mpact_verbose, 0, "Enable verbose logs for Coral CSS simulation")
IREE_FLAG(int32_t, torq_mpact_debug, 0, "Enable gdb when invoking Coral for CSS emulation")
IREE_FLAG(string, torq_mpact_trace_file, "", "Path to output trace file for Coral CSS simulation")

#else

static bool FLAG_torq_mpact_verbose = 0;
static int32_t FLAG_torq_mpact_debug = 0;
static std::string FLAG_torq_mpact_trace_file = "";

#endif

#define LOG(...)                                                                                   \
    do {                                                                                           \
        if (FLAG_torq_mpact_verbose) {                                                           \
            printf(__VA_ARGS__);                                                                   \
        }                                                                                          \
        fflush(stdout);                                                                               \
    } while (0)

// these values are defined by the coral simulator definition
static const uint32_t coral_itcm_start = 0x00000000;
static const uint32_t coral_itcm_size = 0x2000; // 8KB
static const uint32_t coral_dtcm_start = 0x00010000;
static const uint32_t coral_dtcm_size = 0x8000; // 32KB

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


static void process_status_code(void *cpu, CoralNPUSimulator *coral_sim, bool re_run) {

    uint32_t mcode = css_sw__reg_read(cpu, RA_(CSS, MBX_DAT_0));

    if (mcode != 0xFFFFFFFF) {
        uint32_t mtval = css_sw__reg_read(cpu, RA_(CSS, MBX_DAT_1));

        dump_riscv_fault(mcode, mtval);
        fflush(stderr);

        fprintf(stderr, "Aborting execution");
        fflush(stderr);
        abort();
    }
}

static void execute_program(CoralNPUSimulator *coral_sim, uint32_t start_pc) {
    LOG("------ coral output start ------\n");
   
    coral_sim->Run(start_pc);
    if (coral_sim->WaitForTermination(500000)) {
        LOG("Halted \n");
    }
    else {
        LOG("Didn't halt \n");
    }
    
    LOG("------ coral output end------\n");
}

static void run_cpu_mpact_binary_ex(void *cpu, CoralNPUSimulator *coral_sim, bool re_run) {
    uint32_t start_pc = css_sw__reg_read(cpu, RA_(CPU,PCSTART));

    LOG("Starting CPU with PC = 0x%08x\n", start_pc);

    // Execute program
    execute_program(coral_sim, start_pc);

    // for debug purposes
    process_status_code(cpu, coral_sim, re_run);
    
}

namespace {

class TCMMemoryTarget : public CoralMemoryTarget {
public:
    TCMMemoryTarget(void *cpu) : cpu_(cpu) {}

    void Load(uint64_t address, uint8_t* data, size_t size) override {
        void *code_host_addr = css_sw__tcm_a2p(cpu_, address);
        memcpy(data, code_host_addr, size);
    }

    void Store(uint64_t address, const uint8_t* data, size_t size) override {
        void *code_host_addr = css_sw__tcm_a2p(cpu_, address);
        memcpy(code_host_addr, data, size);
    }
private:
    void *cpu_;
};

class UartTarget : public CoralMemoryTarget {
public:
    void Load(uint64_t address, uint8_t* data, size_t size) override {
        // No-op for reads
    }

    void Store(uint64_t address, const uint8_t* data, size_t size) override {
        // For simplicity, we assume all writes to this target are null-terminated strings
        std::string output(reinterpret_cast<const char*>(data), size);
        printf("%s", output.c_str());
        fflush(stdout);
    }
};

class CssRegsTarget : public CoralMemoryTarget {
public:
    CssRegsTarget(void *cpu) : cpu_(cpu) {}
    void Load(uint64_t address, uint8_t* data, size_t size) override {

        assert(size == 4); // We only support 32-bit register reads
        assert((address & 0x3) == 0); // Address should be aligned to 4 bytes

        *(uint32_t*)data = css_sw__reg_read(cpu_, address);
        
    }

    void Store(uint64_t address, const uint8_t* data, size_t size) override {

        assert(size == 4); // We only support 32-bit register reads
        assert((address & 0x3) == 0); // Address should be aligned to 4 bytes
        
        css_sw__reg_write(cpu_, address, *(uint32_t*)data);
    }

private:
    void *cpu_;
};

typedef int (*fake_cb_read_mem_t) (void *ctx, uint32_t tag, size_t addr, size_t size, uint8_t *data);
typedef int (*fake_cb_write_mem_t) (void *ctx, uint32_t tag, size_t addr, size_t size, uint8_t *data);

typedef struct fake_css_cpu_t {
    uint32_t state;
    char is_waiting;
    uint8_t* itcm;
    uint32_t itcm_size;
    uint8_t* dtcm;
    uint32_t dtcm_size;
    uint32_t cpu_regs [REG_SIZE__TORQ_CPU_REGS/4];
    void* ext_mem_ctx;
    fake_cb_read_mem_t cb_read_ext_mem;
    fake_cb_write_mem_t cb_write_ext_mem;
    void* ext_irq_ctx;
    void* cb_get_ext_irq;
    void (*css_sw)(void *);
} fake_css_cpu_t;


class LramTarget: public CoralMemoryTarget {
public:
    LramTarget(void *cpu) : cpu_((fake_css_cpu_t *)cpu) {

    }

    void Load(uint64_t address, uint8_t* data, size_t size) override {
                
        int r = (*(cpu_->cb_read_ext_mem))(cpu_->ext_mem_ctx, 'CCPU', address, size, data);
        
        if (r < 0) {
            fprintf(stderr, "Invalid CSS read at address 0x%08x, size %zu\n", (uint32_t)address, size);
            abort();
        }
    
    }

    void Store(uint64_t address, const uint8_t* data, size_t size) override {
        int r = (*(cpu_->cb_write_ext_mem))(cpu_->ext_mem_ctx, 'CCPU', address, size, (uint8_t *)data);
        
        if (r < 0) {
            fprintf(stderr, "Invalid CSS write at address 0x%08x, size %zu\n", (uint32_t)address, size);
            abort();
        }
    }
private:
    fake_css_cpu_t *cpu_;
};

class OOBTarget : public CoralMemoryTarget {
public:
    void Load(uint64_t address, uint8_t* data, size_t size) override {
        fprintf(stderr, "Out-of-bounds read at address 0x%08x, size %zu\n", (uint32_t)address, size);
        abort();
    }

    void Store(uint64_t address, const uint8_t* data, size_t size) override {
        fprintf(stderr, "Out-of-bounds write at address 0x%08x, size %zu\n", (uint32_t)address, size);
        abort();
    }
};

// tracing function
void trace_callback(uint32_t pc, uint32_t instruction, std::string &disassembly) {
    static FILE *trace_file = nullptr;
    if (!trace_file) {
        trace_file = fopen(FLAG_torq_mpact_trace_file, "w");
        if (!trace_file) {
            fprintf(stderr, "Failed to open trace file: %s\n", FLAG_torq_mpact_trace_file);
            return;
        }
    }

    fprintf(trace_file, "0x%08x: 0x%08x %s\n", pc, instruction, disassembly.c_str());
    
    // we flush all the time since we may crash the simulation
    fflush(trace_file);
}

} // namespace



void run_cpu_mpact_binary(void *cpu) {
    LOG("Creating CPU\n");

    CoralNPUSimulator *coral_sim = CoralNPUSimulator::Create();

    TCMMemoryTarget tcm_memory_target(cpu);
    UartTarget uart_target;
    CssRegsTarget css_regs_target(cpu);
    LramTarget lram_target(cpu);
    OOBTarget oob_target;

    coral_sim->RegisterMemoryTarget(REG_ADDR__TORQ_CV_ITCM, REG_SIZE__TORQ_CV_ITCM, &tcm_memory_target);
    coral_sim->RegisterMemoryTarget(REG_ADDR__TORQ_CV_DTCM, REG_SIZE__TORQ_CV_DTCM, &tcm_memory_target);
    coral_sim->RegisterMemoryTarget(REG_ADDR__TORQ_CV_LRAM, REG_SIZE__TORQ_CV_LRAM, &lram_target);
    coral_sim->RegisterMemoryTarget(0x10000000, 4, &uart_target);
    coral_sim->RegisterMemoryTarget(REG_ADDR__TORQ_CV_CSS_REGS, REG_SIZE__TORQ_CV_CSS_REGS, &css_regs_target);    
    coral_sim->RegisterMemoryTarget(REG_ADDR__TORQ_CV_CDMA_REGS, REG_SIZE__TORQ_CV_CDMA_REGS, &css_regs_target);
    coral_sim->RegisterOOBMemoryTarget(&oob_target);

    if (FLAG_torq_mpact_trace_file[0]) {
        coral_sim->SetTraceCallback(trace_callback, true);
    }

    LOG("Starting CPU\n");

    run_cpu_mpact_binary_ex(cpu, coral_sim, false);

    css_sw__end_sleep(cpu);
}
