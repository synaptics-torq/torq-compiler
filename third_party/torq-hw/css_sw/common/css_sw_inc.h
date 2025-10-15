// Copyright 2025 Synaptics Incorporated. All rights reserved.
// Created:  02/10/2025, Hongjie Guan
// Language: C99 with -fno-strict-aliasing

//
//  CSS Software Coding Guidelines:
//
//  (1) Always split the code into 2 parts:
//      (a) css_sw_main()
//          It handles the handshaking between NSS and CSS.
//          It's subject to changes due to both hardware and C model changes.
//          Do not modify it by yourself.
//      (b) css_sw_task()
//          It does the actual compute.
//          It's called by css_sw_main to perform one single CSS software task.
//          A CSS software task, similar to a slice task in NSS, is scheduled by the NSS CFG agent.
//          This part of code must be written in the pure C language.
//          It can call some of the low level functions defined in css_sw_inc.h.
//
//  (2) Accessing DTCM:
//      There are 2 types of DTCM locations:
//      (a) Private memory only used in software, such as variables and memory blocks allocated by the C compiler/C runtime
//          in the stack, heap, .data section, and .bss section.
//          Private memory is not modeled in the behavior CSS model thus not visible to hardware.
//          To access a private memory location, software does not need to know or use its physical address.
//      (b) Shared memory for exchanging data between software and hardware, such as DMA buffers. They are set aside directly
//          by the user and not managed by the C compiler/runtime.
//          To access a shared memory location, software must first convert the base physical address of the memory block to
//          a C pointer. This must be done by calling the low level function, css_sw__tcm_a2p().
//      In any case, software should not directly use physical addresses to access DTCM.
//
//  (3) Accessing hardware registers:
//      Always use the low level functions, css_sw__reg_read() and css_sw__reg_write(), to access hardware registers.
//      Only single 32-bit accesses are supported.
//
//  (4) Accessing LRAM:
//      Always use the low level functions, css_sw__reg_read() and css_sw__reg_write(), to access LRAM.
//      Only single 32-bit accesses are supported.
//      Accessing LRAM directly from the CSS software task is not recommended for performance concerns.
//
//  (5) CSS DMA:
//      CSS DMA transfers are usually pre-scheduled in the NSS CFG decriptor and managed by the NSS CFG agent.
//      Initiating ad hoc DMA transfers directly in the CSS software task is an optional feature and not fully supported 
//      in the behavior CSS C model at this time.
//      Before the ad hoc DMA transfer is brought up and verified, do not use it and do not call css_sw__dma_wait().
//
//  (6) End of CSS task:
//      This is handled in css_sw_main().
//      Typically, the CSS software task in css_sw_task() should not call css_sw__end_sleep() and css_sw__end_halt().
//
//  In typical cases, the CSS software task only needs to call 3 of the low level functions defined in css_sw_inc.h:
//      css_sw__tcm_a2p()
//      css_sw__reg_read()
//      css_sw__reg_write()
//

#ifndef CSS_SW_INC_H_
#define CSS_SW_INC_H_

#ifdef __cplusplus
extern "C" {
#endif


#ifdef TORQ_BEH_CSS_SW

void     *css_sw__tcm_a2p   (void *cpu, uint32_t addr);                 // Convert TCM physical address to C pointer
uint32_t  css_sw__tcm_p2a   (void *cpu, void *ptr);                     // Convert C pointer to TCM physical address
uint32_t  css_sw__reg_read  (void *cpu, uint32_t addr);                 // Read 32-bit data in the address space outside TCM
void      css_sw__reg_write (void *cpu, uint32_t addr, uint32_t data);  // Write 32-bit data in the address space outside TCM
void      css_sw__dma_wait  (void *cpu);                                // Wait for ad hoc DMA copy operation to complete
void      css_sw__end_sleep (void *cpu);                                // End the current CSS task and go to sleep
void      css_sw__end_halt  (void *cpu);                                // End the current CSS task and halt

#else

#define css_sw__tcm_a2p(cpu_, addr_) ((void *)(((char *)0L)+(uint32_t)(addr_)))

//__attribute__((always_inline))
//static void *css_sw__tcm_a2p(void *cpu, uint32_t addr)
//{
//    return (void *)(((char *)0L)+addr);
//}

#define css_sw__tcm_p2a(cpu_, ptr_) ((uint32_t)(intptr_t)(void *)(ptr_))

//__attribute__((always_inline))
//static uint32_t css_sw__tcm_p2a(void *cpu, void *ptr)
//{
//    return (uint32_t)(intptr_t)ptr;
//}

#define css_sw__reg_read(cpu_, addr_) (*(volatile uint32_t *)(intptr_t)(uint32_t)(addr_))

//__attribute__((always_inline))
//static uint32_t css_sw__reg_read(void *cpu, uint32_t addr)
//{
//    return *(volatile uint32_t *)(intptr_t)addr;
//}

#define css_sw__reg_write(cpu_, addr_, data_) do { *(volatile uint32_t *)(intptr_t)(uint32_t)(addr_) = (uint32_t)(data_); } while (0)

//__attribute__((always_inline))
//static void css_sw__reg_write(void *cpu, uint32_t addr, uint32_t data)
//{
//    *(volatile uint32_t *)(intptr_t)addr = data;
//}

#define css_sw__dma_wait(cpu_) do { asm volatile("wfi"); } while (0)

//__attribute__((always_inline))
//static void css_sw__dma_wait(void *cpu)
//{
//    asm volatile("wfi");
//}

#define css_sw__end_sleep(cpu_) do { asm volatile("wfi"); } while (0)

//__attribute__((always_inline))
//static void css_sw__end_sleep(void *cpu)
//{
//    asm volatile("wfi");
//}

#define css_sw__end_halt(cpu_) do { asm volatile(".word 0x08000073"); } while (0)

//__attribute__((always_inline))
//static void css_sw__end_halt(void *cpu)
//{
//    asm volatile(".word 0x08000073");
//}

#endif


#ifdef __cplusplus
}
#endif

#endif
