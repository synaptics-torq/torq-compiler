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

#include "css_kernel.c.inc"