// clang-format off
//---------------------------------------------------------------------------
//  Copyright 2023-2024 Synaptics Inc
//---------------------------------------------------------------------------
//! \file
//! \brief      High Level Torq C Model API
//! \author     Hongjie Guan
//! \date       01/09/2024 - 01/18/2024
//---------------------------------------------------------------------------
#ifndef __TORQ_API_H__
#define __TORQ_API_H__

#include "torq_bitstream.h"

#ifdef __cplusplus
extern "C"
{
#endif


// To use the latest API, define the following TORQ_API_VER before including this header file:
//   #define TORQ_API_VER 0x00010002 // v1.2
//
// Older API versions will be still be supported for a limited time:
//   #define TORQ_API_VER 0x00010001 // v1.1
//   #define TORQ_API_VER 0x00010000 // v1.0


#define TORQ_MAX_ND  16
#define TORQ_SLICES  2

typedef struct torq_ndl_cmd_t        // To be future proof, please always zero out this struct before populating it.
{
    uint32_t    n[TORQ_MAX_ND];      // Dimension length; Similar to Numpy ndarray.shape (with reversed dimension order)
    int32_t     s[TORQ_MAX_ND];      // Dimension stride; Similar to Numpy ndarray.stride (with reversed dimension order)
    int32_t     p[TORQ_MAX_ND];      // Reserved. May be removed in the next version.
                                      // When serializing from N-D space to byte stream, at the end of this dimension,
                                      //   append p zero bytes to (if p>0) or remove p bytes from (if p<0) the byte stream;
                                      // When deserializing from byte stream to N-D space, at the end of this dimension,
                                      //   remove p bytes from (if p>0) or append p zero bytes to (if p<0) the byte stream.
    char        pmode[TORQ_MAX_ND];  // Reserved. May be removed in the next version.
                                      //    0 : Same as 'A'
                                      //   'A': Append/remove p bytes
                                      //   'O': Overwrite p bytes with cfg.pad_value and keep the same length
                                      //   'M': Mask off the memory write operation for p bytes and keep the same length
    char        t[TORQ_MAX_ND];      // Dimension tag, see Dimension Tag List
    uint8_t     nld;                  // Number of LDIMs (aka low dimensions or bit-wise dimensions)
    uint8_t     nhd;                  // Number of HDIMs (aka high dimensions or cycle-wise dimensions)
    uint8_t     nsd;                  // Number of SDIMs (dimenions in the special/secondary ND loop in parallel with the main ND loop)
    uint8_t     sync_mode;            // Multi-agent scheduling/synchronization mode
                                      //   'R': Round-robin mode; Agents are scheduled in the round-robin fashion, starting from this agent
                                      //   'P': Producer-consumer mode; This agent is the producer
                                      //    0 : Second agent in round-robin mode, or consumer agent in producer-consumer mode
    uint8_t     sync_nhd;             // Number of HDIMs within the scheduling unit in multi-agent scheduling/synchronization
                                      //   0:  default hardware behavior
                                      //   >0: synchronize at the end of the first (sync_nhd) HDIMs.
    uint32_t    base_addr;            // Base address of the tensor in LRAM
                                      //   In the h/w accellerated stride=2 case, give the base address of the segment h/w first access.
} torq_ndl_cmd_t;

typedef union torq_cfg_t
{
    // for slice (slc_id>=0) configuration only:
    struct {
        uint32_t    w_format;                // See Weight Format List; Only for weight decompression. NOT for ALU.
        uint32_t    alu_format;              // ALU number format; See Number Format List
        uint32_t    alu_op0_mode[2*4];       // See ALU Op0 Mode List; [0] for set 0, [4]: for set 1, [1..3, 5..7]: reserved, must be 0.
        uint32_t    alu_op1_mode[2*4];       // See ALU Op1 Mode List; [0] for set 0, [4]: for set 1, [1..3, 5..7]: reserved, must be 0.
        uint32_t    alu_d_unsigned : 8;      // bit 0..3: s0a0, s0a1, s0a2, s0a3; bit 4..7: s1a0, s1a1, s1a2, s1a3; s: set index, a: alu index % 4
        uint32_t    alu_w_unsigned : 4;      // bit 0..3: a0, a1, a2, a3; a: alu index % 4
        uint32_t    de_w_unsigned : 1;       // Obsolete. May be removed in the next version. Must set to 0 when non-zero w_format is specified.
        uint32_t    alu_disable : 16;        // Each bit controls 16 ALUs; 1: disable the 16-ALU group.
        uint32_t    act_disable : 4;         // Each bit controls 4 ACTs; 1: disable the 4-ACT group.

        uint32_t    act_format;              // ACT number format; See Number Format List
        uint32_t    act_mode;                // See ACT Mode List
        uint8_t     act_lsh[4];              // Left shift amount for every group of 4 adjacent int32 data from P-bus: {0,0,0,0}, {0,8,0,8}, {0,8,8,16}, {0,8,16,24}
        uint8_t     act_sum_bits;            // Reserved. May be removed in the next version.
        uint8_t     act_rsh;
        uint8_t     act_sh;                  // Obsolete. May be removed in the next version. Must set to 0 when non-zero act_rsh is speficied.
        uint32_t    act_round_mode;          // See Roudning Mode List; For the rescaling step in ACT
        int32_t     act_clip_min;
        int32_t     act_clip_max;
        int32_t     act_zero_point;

        uint8_t     no_p_clear;
        uint8_t     no_p_output;

        uint32_t    kernel_left;
        uint32_t    kernel_right; 
        uint32_t    kernel_top; 
        uint32_t    kernel_bottom; 

        int32_t     pad_left; 
        int32_t     pad_right; 
        int32_t     pad_top; 
        int32_t     pad_bottom;
        int32_t     pad_value;

        int32_t     stride;
        int32_t     stride_offset;

        uint8_t*    table;
    };

    // for NSS top (slc_id==-1) configuration only:
    struct {
        // NSS CFG agent schedules the following hardware threads in exactly the same way:
        // * Hardware compute slices in NSS (NSS.SLC0..NSS.SLCn)
        // * Software compute slice in CSS (CSS.CPU)
        // * N-DMA channels (NSS.DMA.XR and NSS.DMA.XW)
        // * C-DMA channel
        // Software configures a start bit, a wait bit, and some start parameters for each hardware thread.
        // In each NSS task (slc_id==-1), NSS CFG agent simultaneously starts all hardware threads with start=1,
        // and then wait for all hardware threads with wait=1 to complete before ending the NSS task.

        // NSS.SLC0..n: 1 start parameter per SLC
        char        slc_start[TORQ_SLICES];
        char        slc_wait[TORQ_SLICES];
        uint32_t    slc_cfg_addr[TORQ_SLICES];

        // NSS.DMA.XR: no start parameters
        char        dma_xr_start;
        char        dma_xr_wait;
        uint32_t    dma_xr_attr; // AXI attributes/sideband signals: [15:12] arqos, [11:8] arcache, [6:4] arprot, [3:0] aruser

        // NSS.DMA.XW: no start parameters
        char        dma_xw_start;
        char        dma_xw_wait;
        uint32_t    dma_xw_attr; // AXI attributes/sideband signals: [15:12] awqos, [11:8] awcache, [6:4] awprot, [3:0] awuser

        // CSS.CPU: 5 start parameters (not supported in this version)
        char        css_start;
        char        css_wait;
        char        css_start_mode; // 0: Release CPU reset (CPU should already be in the reset state); 1: Wake up CPU by rasing the NSS2CSS MBX IRQ (CPU should already be in the WFI state)
        char        css_wait_mode;  // 0: Assert CPU reset (CPU should already be in either the WFI or HALT state); 1: Do nothing (CPU should already be in the WFI state)
        uint32_t    css_start_addr; // CPU start address in the cpu address space (only used in css_start_mode 0)
        uint32_t    css_mbx[4];     // General purpose mailbox to CPU software

        // CSS.DMA: 3 start parameters (not supported in this version)
        // WARNING: There is only 1 CDMA channel shared by NSS and CSS.
        //          CDMA and CSS can be scheduled in parallel in NSS (here) ONLY IF CDMA is not used in the CSS software task.
        char        cdma_start;
        char        cdma_wait;
        uint32_t    cdma_src_addr;  // Source address in the torq address space (host view)
        uint32_t    cdma_dst_addr;  // Destination address in the torq address space (host view)
        uint32_t    cdma_len;       // Total transfer length in bytes
    };
} torq_cfg_t;


#define TORQ_LADDR_APPEND (~0)  // Append to the previous part of the same descritor
#define TORQ_LADDR_NONE   (1)   // No LRAM address, descriptor won't be generated
#define TORQ_XADDR_APPEND (~0)  // Append to the previous part of the same descritor
#define TORQ_XADDR_NONE   (1)   // No XRAM address, directly load to LRAM (not recommended)

void *torq_open(const char *path);
int   torq_close(void *self);
int   torq_cfg_begin(void *self, int slc_id, uint32_t cfg_desc_laddr, uint32_t cfg_desc_xaddr);
int   torq_cfg_end(void *self);
int   torq_task_cfg_begin(void *self, torq_cfg_t *cfg, uint32_t cfg_desc_laddr, uint32_t cfg_desc_xaddr); // cfg_desc_*addr: reserved, must be ~0
int   torq_task_cfg_end(void *self);
int   torq_ndl_desc_write(void *self, uint32_t tag, int set_id, torq_ndl_cmd_t *cmd, uint32_t ndl_desc_laddr, uint32_t ndl_desc_xaddr); // tag: see NDL Descriptor Tag List
int   torq_lram_read(void *self, int slc_id, uint32_t tag, uint32_t addr, uint32_t size, void* data); // tag: see Data Buffer Tag List
int   torq_lram_write(void *self, int slc_id, uint32_t tag, uint32_t addr, uint32_t size, void* data); // tag: see Data Buffer Tag List
int   torq_xram_read(void *self, int slc_id, uint32_t tag, uint32_t addr, uint32_t size, void* data); // tag: see Data Buffer Tag List
int   torq_xram_write(void *self, int slc_id, uint32_t tag, uint32_t addr, uint32_t size, void* data); // tag: see Data Buffer Tag List
int   torq_get_bitstream(void *self, torq_bitstream_segment_t** bitstream);
int   torq_run(void *self);


// For backward compatibility
#if (defined(TORQ_API_VER) && TORQ_API_VER < 0x00010002) // < v1.2
    #define TORQ_DBUS_WIDTH 72
    #define TORQ_WBUS_WIDTH 32
    #define TORQ_BBUS_WIDTH 32
    #define TORQ_PBUS_WIDTH 64
    #define TORQ_QBUS_WIDTH 64
#if (! || TORQ_API_VER < 0x00010001) // < v1.1
    #define torq_cfg_begin(self_,slc_id_,cfg_desc_addr_)        torq_cfg_begin((self_),(slc_id_),(cfg_desc_addr_),TORQ_XADDR_NONE)
    #define torq_task_cfg_begin(self_,cfg_,cfg_desc_addr_)       torq_task_cfg_begin((self_),(cfg_),(cfg_desc_addr_),TORQ_XADDR_NONE)
    #define torq_ndl_desc_write(self_,tag_,cmd_,ndl_desc_addr_)  torq_ndl_desc_write((self_),(tag_),0,(cmd_),(ndl_desc_addr_),TORQ_XADDR_NONE)
    #define torq_data_read torq_lram_read
    #define torq_data_write torq_lram_write
#else
    #define torq_ndl_desc_write(self_,tag_,cmd_,ndl_desc_laddr_,ndl_desc_xaddr_)  torq_ndl_desc_write((self_),(tag_),0,(cmd_),(ndl_desc_laddr_),(ndl_desc_xaddr_))
#endif
#endif


// Dimension Tag List for HDIMs of Flexible NDL Descriptors:

//   'X'      Horizontal index in output tensor
//   'Y'      Vertical index in output tensor
//   'A'      Linear index on flattened X-Y plane of output tensor (a = y*yn+x)
//   'I'      Horizontal index in kernel
//   'J'      Vertical index in kernel
//   'U'      Channel index within a channel group of input tensor 
//   'V'      Channel index within a channel group of output tensor
//   'G'      Channel group index of both input and output tensors

// Dimension Tag List for LIDMs of Both Flexible and Fixed-format NDL Descriptors:

//   'I'      Duplicate the same byte (bit-wise)
//      'B'   Byte loop in Data (bit-wise)
//   'J'      Duplicate the same data (bit-wise)
//      'D'   Data loop in Group (bit-wise)
//   'K'      Duplicate the same group (bit-wise)
//      'G'   Group loop in Step (bit-wise)

// Dimension Tag List for HIDMs of Fixed-format NDL Descriptors:

//   'L'      Repeat the same Step (cycle-wise)
//      'S'   Step loop in Word (cycle-wise)
//   'M'      Repeat the same Word (cycle-wise)
//      'W'   Word loop in Block
//   'N'      Repeat the same Block (cycle-wise)
//      'T'   Block loop in Task


// Data Buffer Tag List:

//   'D'      Input activation tensor
//   'W'      Weight tensor
//   'B'      Bias and scale tenor
//   'Q'      Output activation tensor
//   'C'      Code for CSS CPU


// NDL Descriptor Tag List:

//   'REF'    Describes the original compute task by definition; Some dimensions affect CFG_REGS; Must be written first
//   'ALU'    For software only; Describes the actual compute order mapped in ALU

//   'DMEM'   For software only; Describes the tensor format of D buffer in LRAM
//   'WMEM'   For software only; Describes the tensor format of W buffer in LRAM
//   'BMEM'   For software only; Describes the tensor format of B buffer in LRAM
//   'QMEM'   For software only; Describes the tnesor format of Q buffer in LRAM

//   'DEDR'   Flexible NDL descriptor in LRAM; Configures DE.DEDR agent; LRAM to DBUS
//   'DEWR'   Flexible NDL descriptor in LRAM; Configures DE.DEWR agent; LRAM to WBUS
//   'DEBR'   Flexible NDL descriptor in LRAM; Configures DE.DEBR agent; LRAM to DE.BBUS
//   'DEQW'   Flexible NDL descriptor in LRAM; Configures DE.DEQW agent; DE.QBUS to LRAM

//   'ACBW'   Fixed-format NDL descriptor in CFG_REGS; Configures ACT.ACBW agent; DE.BBUS to BREG
//   'ACBR'   Fixed-format NDL descriptor in CFG_REGS; Configures ACT.ACBR agent; BREG to ACT
//   'ACPW'   Fixed-format NDL descriptor in CFG_REGS; Configures ACT.ACPW agent; PBUS to ACT (experimental, subject to changes)
//   'ACPR'   Fixed-format NDL descriptor in CFG_REGS; Configures CE.ACPR agent; PRAM to PBUS
//   'CEDW'   Fixed-format NDL descriptor in CFG_REGS; Configures CE.CEDW agent; DBUS to IRAM
//   'CEDR'   Fixed-format NDL descriptor in CFG_REGS; Configures CE.CEDR agent; IRAM to ALU (read from IRAM)
//   'ALDW'   Fixed-format NDL descriptor in CFG_REGS; Configures CE.ALDW agent; IRAM to ALU (write to ALU)
//   'CEWW'   Fixed-format NDL descriptor in CFG_REGS; Configures CE.CEWW agent; WBUS to WRAM
//   'CEWR'   Fixed-format NDL descriptor in CFG_REGS; Configures CE.CEWR agent; WRAM to ALU
//   'CEPR'   Fixed-format NDL descriptor in CFG_REGS; Configures CE.CEPR agent; PRAM to ALU
//   'CEPW'   Fixed-format NDL descriptor in CFG_REGS; Configures CE.CEPW agent; ALU to PRAM

//   'DIXR'   Flexible NDL descriptor in CFG_REGS; Configures DMA.XR agent; XRAM to DMA.XR
//   'DILW'   Flexible NDL descriptor in CFG_REGS; Configures DMA.XR agent; DMA.XR to LRAM
//   'DOLR'   Flexible NDL descriptor in CFG_REGS; Configures DMA.XW agent; LRAM to DMA.XW
//   'DOXW'   Flexible NDL descriptor in CFG_REGS; Configures DMA.XW agent; DMA.XW to XRAM

// ACT Mode List:

// 0          Same as 'ACT'
// 'ACT'      Default ACT mode
// 'ABS':     q=abs(p)                   (experimental, subject to changes)
// 'NEG':     q=-p                       (experimental, subject to changes)
// 'CLZ':     q=count_leading_zeros(p)   (experimental, subject to changes)
// 'CEL':     q=ceiling(p)               (experimental, subject to changes)
// 'FLR':     q=floor(p)                 (experimental, subject to changes)
// 'LSL':     q=p<<s                     (experimental, subject to changes)
// 'LSR':     q=(uint)p>>s               (experimental, subject to changes)
// 'ASR':     q=(int)p>>s                (experimental, subject to changes)
// 'I2F':     q=(float)p                 (experimental, subject to changes)
// 'F2I':     q=(int)p                   (experimental, subject to changes)

// ACT Rounding Mode List:

// 0          Same as 'NTP'
// 'OFF'      No rounding
// 'NTE'      Round to nearest, tie towards even
// 'NTO'      (Not supported) Round to nearest, tie towards odd
// 'NTP'      Round to nearest, tie towards positive infinity
// 'NTN'      (Not supported) Round to nearest, tie towards negative infinity
// 'NTI'      (Not supported) Round to nearest, tie towards infinity (away from zero)
// 'NTZ'      (Not supported) Round to nearest, tie towards zero
// 'DBL'      (Not recommended) Double rounding as defined in TOSA

// ALU Op0 Mode List:

// 0          Same as 'MUL'
// 'MUL'      x=D*W
// 'DBYP'     x=D

// ALU Op1 Mode List:

// 0          Same as 'ACC'
// 'ACC'      P=P+x
// 'SACC'     even ALU cycles: P=P+x; odd ALU cycles: P=P-x;
// 'MUL'      P=P*x (for bfloat16 only)
// 'AMAX'     P=argmax(x0,x1,...,xN)<<16
// 'AMIN'     P=argmin(x0,x1,...,xN)<<16
// 'MAX'      P=max(P,x)
// 'MIN'      P=min(P,x) 
// 'GT'       P=(x0>x1)<<16;
// 'GE'       P=(x0>=x1)<<16;
// 'EQ'       P=(x0==x1)<<16
// 'OR'       x=!!x; P=P|x; (P0=0)
// 'AND'      x=!!x; P=P&x; (P0=1)
// 'XOR'      x=!!x; P=P^x; (P0=0)
// 'NOT'      x=!!x; P=P^x; (P0=1)
// 'BOR'      P=P|x; (P0=0)
// 'BAND'     P=P&x; (P0=~0)
// 'BXOR'     P=P^x; (P0=0)
// 'BNOT'     P=P^x; (P0=~0)
// 'BYP'      P=x (for P_INIT and TRANSPOSE)

// Number Format List (for ALU and ACT):
// 0          Same as 'INT'
// 'I'        Integer
// 'BF'       Bfloat (experimental, subject to changes)

// Weight Format List (for weight decompression only, NOT for ALU):

// 0          Same as 'SI' for 2,4,6-bit integer
// 'SI'       Signged integer for 2,4,6-bit integer
// 'UI'       Unsigned integer for 2,4,6-bit integer
// 'FP'       Reserved (for future FP4, FP8 support)
// 'BF'       Reserved
// 'NF'       Reserved (for future NF4 support)


#ifdef __cplusplus
}
#endif

#endif

