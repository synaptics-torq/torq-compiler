/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2025 Synaptics Incorporated */

#ifndef __TORQ_REG_DEFINE_H__
#define __TORQ_REG_DEFINE_H__

#define REG_ADDR__SYNPU_HV_NSS_REGS         0x001f0000
#define REG_ADDR__SYNPU_HV_CSS_REGS         0x001fc000
#define REG_ADDR__SYNPU_HV_CPU_REGS         0x001fd000
#define REG_ADDR__SYNPU_HV_CDMA_REGS        0x001fe000

#define REG_ADDR__SYNPU_NSS_REGS_CFG        0x00000004
#define REG_ADDR__SYNPU_NSS_REGS_CTRL       0x00000008
#define REG_ADDR__SYNPU_NSS_REGS_STATUS     0x0000000c
#define REG_ADDR__SYNPU_NSS_REGS_START      0x00000010
#define REG_ADDR__SYNPU_CSS_REGS_IEN_HST    0x00000004

/* Register field definitions */
#define REG_FIELD_POS__SYNPU_NSS_REGS_CFG_DESC         0
#define REG_FIELD_BITS__SYNPU_NSS_REGS_CFG_DESC        24
#define REG_FIELD_POS__SYNPU_NSS_REGS_CFG_LINK_EN      24
#define REG_FIELD_POS__SYNPU_NSS_REGS_CTRL_IEN_NSS     0
#define REG_FIELD_POS__SYNPU_CSS_REGS_IEN_HST_NSS      0
#define REG_FIELD_POS__SYNPU_NSS_REGS_START_NSS        0
#define REG_FIELD_POS__SYNPU_NSS_REGS_STATUS_NSS       0
#define REG_FIELD_POS__SYNPU_NSS_REGS_STATUS_XR        2
#define REG_FIELD_POS__SYNPU_NSS_REGS_STATUS_XW        3
#define REG_FIELD_POS__SYNPU_NSS_REGS_STATUS_SLC0      4
#define REG_FIELD_POS__SYNPU_NSS_REGS_STATUS_SLC1      5

/* Helper macros for register field operations */

#define RA_(blk,reg)  REG_ADDR__SYNPU_HV_##blk##_REGS + REG_ADDR__SYNPU_##blk##_REGS_##reg
#define RF_LSH(reg, field, val) ((val) << REG_FIELD_POS__SYNPU_##reg##_REGS_##field)
#define RF_BMSK_LSH(reg, field, val) (((val) & ((1 << REG_FIELD_BITS__SYNPU_##reg##_REGS_##field) - 1)) << REG_FIELD_POS__SYNPU_##reg##_REGS_##field)
#define RF_FMSK_RSH(reg, field, val) (((val) >> REG_FIELD_POS__SYNPU_##reg##_REGS_##field) & ((1 << REG_FIELD_BITS__SYNPU_##reg##_REGS_##field) - 1))

#endif
