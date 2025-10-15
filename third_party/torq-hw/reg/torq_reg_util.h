// Copyright 2024 Synaptics Incorporated. All rights reserved.
// Created:  03/21/2023, Hongjie Guan
// Language: C99 with -fno-strict-aliasing
// Macro functions for register access

#ifndef TORQ_REG_UTIL_H
#define TORQ_REG_UTIL_H

#ifdef __cplusplus
extern "C" {
#endif


//--------------------------------------------------------------------------------------------------
// Macro functions for manipulating register address and bit-field
//--------------------------------------------------------------------------------------------------

#ifndef RU_PREFIX0
#define RU_PREFIX0 HV_
#endif
#ifndef RU_PREFIX1
#define RU_PREFIX1 TORQ_
#endif

#define _RU_CONCAT4(a,b,c,d)            a ## b ## c ## d
#define _RU_CONCAT5(a,b,c,d,e)          a ## b ## c ## d ## e
#define RU_CONCAT4(a,b,c,d)            _RU_CONCAT4(a, b, c, d)
#define RU_CONCAT5(a,b,c,d,e)          _RU_CONCAT5(a, b, c, d, e)

#define _BMSK32(bits_)                 (~0u >> (32-(bits_)))
#define _FMSK32(pos_,bits_)            (_BMSK32(bits_)<<(pos_))
#define RA_(blk_,reg_)                 ((RU_CONCAT5(REG_ADDR__, RU_PREFIX1, RU_PREFIX0, blk_, _REGS))+(RU_CONCAT5(REG_ADDR__, RU_PREFIX1, blk_, _REGS_, reg_)))
#define RFP_(blk_,reg_fld_)            (RU_CONCAT5(REG_FIELD_POS__, RU_PREFIX1, blk_, _REGS_, reg_fld_))
#define RFB_(blk_,reg_fld_)            (RU_CONCAT5(REG_FIELD_BITS__, RU_PREFIX1, blk_, _REGS_, reg_fld_))
#define RBM_(blk_,reg_fld_)            (_BMSK32(RFB_(blk_,reg_fld_)))
#define RFM_(blk_,reg_fld_)            (_FMSK32(RFP_(blk_,reg_fld_),RFB_(blk_,reg_fld_)))

#define RF_BMSK(blk_,reg_fld_,x_)      ((x_)&RBM_(blk_,reg_fld_))
#define RF_LSH(blk_,reg_fld_,x_)       ((x_)<<RFP_(blk_,reg_fld_))
#define RF_BMSK_LSH(blk_,reg_fld_,x_)  RF_LSH(blk_,reg_fld_,RF_BMSK(blk_,reg_fld_,(x_)))

#define RF_FMSK(blk_,reg_fld_,x_)      ((x_)&RFM_(blk_,reg_fld_))
#define RF_RSH(blk_,reg_fld_,x_)       ((uint32_t)(x_)>>RFP_(blk_,reg_fld_))
#define RF_FMSK_RSH(blk_,reg_fld_,x_)  RF_RSH(blk_,reg_fld_,RF_FMSK(blk_,reg_fld_,(x_)))


#ifdef __cplusplus
}
#endif

#endif
