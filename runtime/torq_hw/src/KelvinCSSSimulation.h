#pragma once

#include <cstdint>
#include <cstdlib>

struct KelvinMailbox {
    uint32_t message[4] = {0, 0, 0, 0};
};

class KelvinSimulator {
  public:
    static KelvinSimulator *Create();

    virtual ~KelvinSimulator() = default;

    // Functions for reading/writing TCMs and Mailbox.
    virtual void ReadTCM(uint32_t addr, size_t size, char *data) = 0;
    virtual const KelvinMailbox &ReadMailbox(void) = 0;
    virtual void WriteTCM(uint32_t addr, size_t size, const char *data) = 0;
    virtual void WriteMailbox(const KelvinMailbox &mailbox) = 0;

    // Wait for interrupt
    virtual bool WaitForTermination(int timeout) = 0;

    // Begin executing starting with the PC set to the specified address. Returns
    // when the core halts.
    virtual void Run(uint32_t start_addr) = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

void run_cpu_kelvin_binary(void *cpu);

#ifdef __cplusplus
}
#endif
