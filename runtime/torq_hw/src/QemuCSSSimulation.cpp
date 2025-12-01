#include "QemuCSSSimulation.h"

#define TORQ_BEH_CSS_SW

#include <cstdint>

#include "css_sw/common/css_sw_reg_inc.h"
#include "css_sw_inc.h"

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>


#ifndef RT_BUILD

#include "iree/base/internal/flags.h"

IREE_FLAG(bool, torq_qemu_verbose, 0, "Enable verbose logs for QEMU CSS simulation")
IREE_FLAG(int32_t, torq_qemu_debug, 0, "Enable gdb when invoking qemu for CSS emulation (listens on the specified TCP port)")
IREE_FLAG(bool, torq_qemu_trace_instructions, false, "Enable qemu instruction tracing (output in /tmp/qemu_trace_XXXX)")

#else

static int32_t FLAG_torq_qemu_debug = 0;
static bool FLAG_torq_qemu_trace_instructions = false;
static bool FLAG_torq_qemu_verbose = false;

#endif

#define LOG(...) \
    do { \
        if (FLAG_torq_qemu_verbose) { \
            printf(__VA_ARGS__); \
        } \
    } while (0)


static void save_memory_to_temporary_file(void *addr, size_t size, char *file_template) {    

    int fd = mkstemp(file_template);

    if (fd == -1) {
        perror("Failed to create temporary file");
        abort();
    }

    write(fd, addr, size);    
    close(fd);
}

static void create_empty_file(int size, char *file_template) {

    int memFd = mkstemp(file_template);

    if (memFd == -1) {
        perror("Failed to create temporary file for memory");
        abort();
    }

    if (ftruncate(memFd, size) == -1) {
        perror("Failed to truncate memory file");
        abort();
    }

    close(memFd);

}

// this value is defined by the qemu "virtual" machine definition
static const uint32_t qemu_ram_start = 0x80000000;

// this is where the code compiled for qemu expects to find the CSS registers
static const uint32_t qemu_css_regs_base_addr = 0x80020000;

// these values are defined by the CSS hardware
static const uint32_t dtcm_size = 32 * 1024; // 32KB

// these values must match what is used in the compiler to generate the binary
static const uint32_t qemu_dtcm_base_addr = 0x80010000;

static const uint32_t mbox_reg_offset[4] = {
    REG_ADDR__TORQ_CSS_REGS_MBX_DAT_0,
    REG_ADDR__TORQ_CSS_REGS_MBX_DAT_1,
    REG_ADDR__TORQ_CSS_REGS_MBX_DAT_2,
    REG_ADDR__TORQ_CSS_REGS_MBX_DAT_3
};

static const uint32_t irqRegsOffets[4] = {
    REG_ADDR__TORQ_CSS_REGS_MBX_IRQ_0,
    REG_ADDR__TORQ_CSS_REGS_MBX_IRQ_1,
    REG_ADDR__TORQ_CSS_REGS_MBX_IRQ_2,
    REG_ADDR__TORQ_CSS_REGS_MBX_IRQ_3
};

static void write_to_memory_file(int mem_fd, int qemu_memory_address, void *data, size_t size) {
    if (lseek(mem_fd, qemu_memory_address - qemu_ram_start, SEEK_SET) == -1) {
        perror("Failed to seek to offset in memory file");
        abort();            
    }
    
    if (write(mem_fd, data, size) != size) {
        perror("Failed to write value to memory file");
        abort();
    }    
}

static void write_css_register_to_memory_file(int mem_fd, uint32_t css_reg_offset, uint32_t value) { 
    write_to_memory_file(mem_fd, qemu_css_regs_base_addr + css_reg_offset, &value, sizeof(value));       
}

static uint32_t read_css_register_from_memory_file(int mem_fd, uint32_t css_reg_offset) { 
    if (lseek(mem_fd, qemu_css_regs_base_addr + css_reg_offset - qemu_ram_start, SEEK_SET) == -1) {
        perror("Failed to seek to offset in memory file");
        abort();            
    }
        
    uint32_t value;

    if (read(mem_fd, &value, sizeof(value)) != sizeof(value)) {
        perror("Failed to read register value from memory file");
        abort();
    }

    return value;
}

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

static void prepare_code(void *cpu, char* code_file) {
    uint32_t reg_code_addr = css_sw__reg_read(cpu, RA_(CPU,PCSTART));    
    void *host_code_addr = css_sw__tcm_a2p(cpu, reg_code_addr);

    LOG("code entry point: received reg 0x%08x, host address 0x%08lx, qemu address 0x%08x\n", reg_code_addr, (uintptr_t) host_code_addr, qemu_ram_start);            
    
    // we cannot save the whole itcm because we need to give this file as starting point to qemu
    save_memory_to_temporary_file(host_code_addr, REG_SIZE__TORQ_CV_ITCM - reg_code_addr, code_file);    
    LOG("Saved code to %s\n", code_file);
}

static void prepare_data_and_registers(void *cpu, char *memory_file) {

    // create an empty file that will represent the memory file for qemu    
    // the size must be enough to cover various things required by qemu
    create_empty_file(10 * 1024 * 1024, memory_file);

    LOG("Memory file created at %s\n", memory_file);

    int mem_fd = open(memory_file, O_RDWR);

    if (mem_fd == -1) {
        perror("Failed to open memory file");
        abort();
    }

    // Write out the MBOX values to the QEMU memory
    uint32_t mbox_values[4];

    for (int i = 0; i < 4; i++) {
        mbox_values[i] = css_sw__reg_read(cpu, REG_ADDR__TORQ_CV_CSS_REGS + mbox_reg_offset[i]);
        uint32_t qemu_mbox_reg_value = mbox_values[i] + qemu_ram_start;
        LOG("mbox_reg[%d] = received reg 0x%08x qemu 0x%08x\n", i, mbox_values[i], qemu_mbox_reg_value);

        write_css_register_to_memory_file(mem_fd, mbox_reg_offset[i], qemu_mbox_reg_value); 
    }
    
    // write out state of IRQ registers            
    for (int i = 0; i < 4; i++) {
        uint32_t irqRegValue = css_sw__reg_read(cpu, REG_ADDR__TORQ_CV_CSS_REGS + irqRegsOffets[i]);
        LOG("irqReg[%d] = 0x%08x\n", i, irqRegValue);

        write_css_register_to_memory_file(mem_fd, irqRegsOffets[i], irqRegValue); 
    }

    // write out the state of the DTCM
    void *dtcm_host_address = css_sw__tcm_a2p(cpu, 0x00010000);
    uint8_t dtcm_data[dtcm_size];
    memcpy(dtcm_data, dtcm_host_address, dtcm_size);

    // update the address of all the extra args stored in DTCM
    
    LOG("argument address 0x%08x\n", mbox_values[0]);

    int args_offset_in_dtcm = (mbox_values[0] + qemu_ram_start) - qemu_dtcm_base_addr;

    uint32_t* args = (uint32_t *) &(dtcm_data[args_offset_in_dtcm]);

    uint32_t args_count = args[0];

    LOG("extra args count is %d\n", args_count);

    // patch all the extra addresses with the memory offset used by qemu
    for (int i = 0; i < args_count; i++) {
        LOG("received args[%d] = 0x%08x\n", i, args[i + 1]);
        args[i + 1] += qemu_ram_start;
        LOG("qemu args[%d] = 0x%08x\n", i, args[i + 1]);
    }

    // save the dtcm data to the memory file
    write_to_memory_file(mem_fd, qemu_dtcm_base_addr, dtcm_data, dtcm_size);

    close(mem_fd);
}

static void run_cpu_qemu_binary_ex(void *cpu, int debugger_port, bool trace_instructions, bool re_run);

static void process_status_code(void *cpu, int mem_fd, bool re_run) {
    uint32_t mcode = read_css_register_from_memory_file(mem_fd, mbox_reg_offset[0]);    

    if (mcode != 0xFFFFFFFF) {

        uint32_t mtval = read_css_register_from_memory_file(mem_fd, mbox_reg_offset[1]);

        dump_riscv_fault(mcode, mtval);
        
        if (!re_run) {

            fprintf(stderr, "Re-run task in debug mode (listening on port 1090)\n");
            fprintf(stderr, "Connect using: gdb-multiarch -ex \"target remote :1090\"\n");
            fprintf(stderr, "Instruction traces will be generated in /tmp/qemu_trace_XXXX\n");
            fprintf(stderr, "\n");

            fprintf(stderr, "================================================================\n");            

            run_cpu_qemu_binary_ex(cpu, 1090, true, true);
        }

        fprintf(stderr, "Aborting execution");
        abort();
    }
}

static void read_back_memory_and_registers(void *cpu, int mem_fd) {
    
    void *dtcm_host_address = css_sw__tcm_a2p(cpu, 0x00010000);

    // copy back the state of the DTCM after the simulation
    
    if (lseek(mem_fd, qemu_dtcm_base_addr - qemu_ram_start, SEEK_SET) == -1) {
        perror("Failed to seek to offset 0x00010000 in memory file");
        abort();
    }

    if (read(mem_fd, dtcm_host_address, dtcm_size) != dtcm_size) {
        perror("Failed to read dtcm_host_address from memory file");
        abort();
    }

    // read back the state of the IRQ registers
    for (int i = 0; i < 4; i++) {

        uint32_t beforeValue = css_sw__reg_read(cpu, REG_ADDR__TORQ_CV_CSS_REGS + irqRegsOffets[i]);
        uint32_t afterValue = read_css_register_from_memory_file(mem_fd, irqRegsOffets[i]);
        
        if (afterValue != beforeValue) {
            LOG("updating irqReg[%d] = 0x%08x (was 0x%08x)\n", i, afterValue, beforeValue);
            css_sw__reg_write(cpu, REG_ADDR__TORQ_CV_CSS_REGS + irqRegsOffets[i], afterValue);
        }

    }
}
 
static void run_cpu_qemu_binary_ex(void *cpu, int debugger_port, bool trace_instructions, bool re_run) {

    char code_file[] = "/tmp/torq_codeXXXXXX";
    char memory_file[] = "/tmp/torq_memoryXXXXXX";

    prepare_code(cpu, code_file);

    prepare_data_and_registers(cpu, memory_file);

    // take the basename of memory_file to use as memory ID in qemu
    auto memory_id = std::string(strrchr(memory_file, '/') + 1);    

    std::string qemu_cmd;

    qemu_cmd += "qemu-system-riscv32 -cpu rv32,v=true,vlen=128,elen=32 -nographic -M virt,memory-backend=" + memory_id + " -m 10M";
    qemu_cmd += " -bios " + std::string(code_file);
    qemu_cmd += " -object memory-backend-file,size=10M,id=" + memory_id + ",mem-path=" + std::string(memory_file) + ",share=on,prealloc=on";
    
    if (debugger_port != 0) {
        qemu_cmd += std::string() + " -gdb tcp::" + std::to_string(debugger_port) + " -S ";
    }

    if (trace_instructions) {        
        char trace_file[] = "/tmp/qemu_trace_XXXXXX";
        mkstemp(trace_file);
        fprintf(stderr, "Instruction trace will be created at %s\n", trace_file);
        qemu_cmd += " -d in_asm,exec,cpu_reset -D " + std::string(trace_file);
    }    

    LOG("Running command: %s\n", qemu_cmd.c_str());

    LOG("------ qemu output start ------\n");

    int ret = system(qemu_cmd.c_str());

    LOG("------ qemu output end------\n");

    if (ret != 0) {
        fprintf(stderr, "Failed to run command: %s\n", qemu_cmd.c_str());
        abort();
    } else {
        LOG("Command executed successfully\n");
    }
                
    unlink(code_file);    
    
    int mem_fd = open(memory_file, O_RDWR);

    process_status_code(cpu, mem_fd, re_run);

    read_back_memory_and_registers(cpu, mem_fd);

    close(mem_fd);

    unlink(memory_file);
    
}


void run_cpu_qemu_binary(void *cpu) {

    LOG("Starting CPU\n");

    run_cpu_qemu_binary_ex(cpu, FLAG_torq_qemu_debug, FLAG_torq_qemu_trace_instructions, false);

    css_sw__end_sleep(cpu);
}