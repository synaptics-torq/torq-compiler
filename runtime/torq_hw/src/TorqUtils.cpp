// Copyright 2024 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TorqUtils.h"

#include <torq_nss_regs_struct.h>

#include <cassert>
#include <iomanip>
#include <variant>
#include <bitset>

using namespace std;

static uint32_t getWordAtPosition(const std::vector<uint8_t>& src, size_t position) {
    if (position + 3 >= src.size()) {
        LOGE << "wrong position: " << position << ", src.size: " << src.size();
        assert(false);
    }
    return (src[position] | (src[position + 1] << 8) | (src[position + 2] << 16) | (src[position + 3] << 24));
}

static void appendWordLE(std::vector<uint8_t>& out, uint32_t word) {
    out.push_back(word & 0xFF);
    out.push_back((word >> 8) & 0xFF);
    out.push_back((word >> 16) & 0xFF);
    out.push_back((word >> 24) & 0xFF);
}

static uint32_t extractField(uint32_t word, uint32_t position, uint32_t bitLength) {
    uint32_t mask = (1 << bitLength) - 1;
    return (word >> position) & mask;
}

void CfgCmd::loadFromVector(const std::vector<uint8_t>& src) {

    // When updating this structure ensure you update the spec

    uint32_t word = getWordAtPosition(src, 0);
    ra = extractField(word, 0, 16);
    int rn = extractField(word, 16, 12);
    assert(id() == extractField(word, 28, 3));
    isLastCmd = extractField(word, 31, 1);

    for (size_t i = 0; i < rn; i++) {
        if (4 * i + 4 + 4 >= src.size()) {
            break;
        }
        uint32_t value = getWordAtPosition(src, 4 * i + 4);
        values.push_back(value);
    }
}

int CfgCmd::id() const {
    return 1;
}

void CfgCmd::serialize(std::vector<uint8_t>& out) const {
    uint32_t rn = values.size();
    uint32_t word = (ra & 0xFFFF)
                  | ((rn & 0xFFF) << 16)
                  | ((id() & 0x7) << 28)
                  | ((isLastCmd & 0x1) << 31);
    appendWordLE(out, word);
    for (size_t i = 0; i < values.size(); ++i) {
        appendWordLE(out, values[i]);
    }
}

int CfgCmd::serializedSize() const {
    return 4 * (values.size() + 1);
}

#define FIELD_NAME(field) \
    case offsetof(torq_nss_regs_t, field): oss << " (" #field ")"; break;

#define MULTI_BYTE_FIELD_NAME(field, c) \
    case offsetof(torq_nss_regs_t, field) + c * 4 : oss << " (" #field "_" #c ")"; break;


std::string printRegisterAddress(uint32_t address) {

    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(8) << std::setfill('0') << address;

    switch (address) {
        FIELD_NAME(NSS.VERSION)
        FIELD_NAME(NSS.CFG)
        FIELD_NAME(NSS.CTRL)
        FIELD_NAME(NSS.STATUS)
        FIELD_NAME(NSS.START)
        FIELD_NAME(NSS.CLK_CTRL)
        FIELD_NAME(NSS.DCG_CTRL)
        MULTI_BYTE_FIELD_NAME(DMA.DMA_XR.CFG, 0)
        MULTI_BYTE_FIELD_NAME(DMA.DMA_XR.CFG, 1)
        MULTI_BYTE_FIELD_NAME(DMA.DMA_XR.CFG, 2)
        MULTI_BYTE_FIELD_NAME(DMA.DMA_XR.CFG, 3)
        MULTI_BYTE_FIELD_NAME(DMA.DMA_XR.CFG, 4)
        FIELD_NAME(DMA.DMA_XR.SRC.HEAD)
        FIELD_NAME(DMA.DMA_XR.SRC.DIMS[0].DIM_SIZE)
        FIELD_NAME(DMA.DMA_XR.SRC.DIMS[0].DIM_STRIDE)
        FIELD_NAME(DMA.DMA_XR.SRC.DIMS[1].DIM_SIZE)
        FIELD_NAME(DMA.DMA_XR.SRC.DIMS[1].DIM_STRIDE)
        FIELD_NAME(DMA.DMA_XR.SRC.DIMS[2].DIM_SIZE)
        FIELD_NAME(DMA.DMA_XR.SRC.DIMS[2].DIM_STRIDE)
        MULTI_BYTE_FIELD_NAME(DMA.DMA_XW.CFG, 0)
        MULTI_BYTE_FIELD_NAME(DMA.DMA_XW.CFG, 1)
        MULTI_BYTE_FIELD_NAME(DMA.DMA_XW.CFG, 2)
        MULTI_BYTE_FIELD_NAME(DMA.DMA_XW.CFG, 3)
        FIELD_NAME(DMA.DMA_XW.DST.HEAD)
        FIELD_NAME(DMA.DMA_XW.DST.DIMS[0].DIM_SIZE)
        FIELD_NAME(DMA.DMA_XW.DST.DIMS[0].DIM_STRIDE)
        FIELD_NAME(DMA.DMA_XW.DST.DIMS[1].DIM_SIZE)
        FIELD_NAME(DMA.DMA_XW.DST.DIMS[1].DIM_STRIDE)
        FIELD_NAME(DMA.DMA_XW.DST.DIMS[2].DIM_SIZE)
        FIELD_NAME(DMA.DMA_XW.DST.DIMS[2].DIM_STRIDE)
        FIELD_NAME(SLC0.DE_REGS.DE_CFG)
        FIELD_NAME(SLC0.DE_REGS.DE_D_R)
        FIELD_NAME(SLC0.DE_REGS.DE_D_RX)
        FIELD_NAME(SLC0.DE_REGS.DE_W_R)
        FIELD_NAME(SLC0.DE_REGS.DE_B_R)
        FIELD_NAME(SLC0.DE_REGS.DE_B_RX)
        FIELD_NAME(SLC0.DE_REGS.DE_Q_W)
        FIELD_NAME(SLC0.DE_REGS.DE_ACT)
        FIELD_NAME(SLC0.DE_REGS.CTRL)
        FIELD_NAME(SLC0.DE_REGS.STATUS)
        FIELD_NAME(SLC0.DE_REGS.START)
        FIELD_NAME(SLC1.DE_REGS.DE_CFG)
        FIELD_NAME(SLC1.DE_REGS.DE_D_R)
        FIELD_NAME(SLC1.DE_REGS.DE_D_RX)
        FIELD_NAME(SLC1.DE_REGS.DE_W_R)
        FIELD_NAME(SLC1.DE_REGS.DE_B_R)
        FIELD_NAME(SLC1.DE_REGS.DE_B_RX)
        FIELD_NAME(SLC1.DE_REGS.DE_Q_W)
        FIELD_NAME(SLC1.DE_REGS.DE_ACT)
        FIELD_NAME(SLC1.DE_REGS.CTRL)
        FIELD_NAME(SLC1.DE_REGS.STATUS)
        FIELD_NAME(SLC1.DE_REGS.START)
        default:
            break;
    }

    return oss.str();
}

void CfgCmd::print() const {
    std::cout << "CfgCmd {"
                << " ra: 0x" << std::hex << ra
                << ", rn: 0x" << values.size()
                << ", id: 0x" << id()
                << ", lastCmd: " << isLastCmd
                << ", values: {" << std::endl;

    for (int i = 0 ; i < values.size(); i++) {
        
        auto address = printRegisterAddress(ra + i * 4);

        // auto maybeRegisterSet = registers.getRegisterSet(address);

        std::cout << "  " << std::hex << std::setw(8) << std::setfill('0') << address << " = 0x" << std::setw(8) << std::setfill('0') << values[i] << std::dec << ", " << std::endl;
    }
    std::cout << "  } }" << std::endl;
}

void SynCmd::loadFromVector(const std::vector<uint8_t>& src) {
    
    uint32_t word = getWordAtPosition(src, 0);
    msk = extractField(word, 0, 16);
    rsvd = extractField(word, 16, 8);
    op = extractField(word, 24, 4);
    assert(extractField(word, 28, 3) == id());
    isLastCmd = extractField(word, 31, 1);
}

int SynCmd::id() const {
    return 6;
}

int SynCmd::serializedSize() const {
    return 4;
}

std::string synOpToString(uint32_t op) {
    switch (op) {
        case 0: return "wait for all";
        case 1: return "wait for any";
        case 2: return "set";
        case 3: return "clear";        
        default: return "UNKNOWN";
    }
}

std::string synMskToString(uint32_t msk) {

    std::string result;

    if (msk & 0x1) {
        result += "nss ";
    }

    if ((msk >> 2) & 0x1) {
        result += "xr ";
    }

    if ((msk >> 3) & 0x1) {
        result += "xw ";
    }

    if ((msk >> 4) & 0x1) {
        result += "slc0 ";
    }

    if ((msk >> 5) & 0x1) {
        result += "slc1 ";
    }

    return result;

}

void SynCmd::print() const {
        
    std::cout << "SynCmd {"
                << " msk: " << msk << "( " << synMskToString(msk) << ")"
                << ", rsvd: " << rsvd
                << ", op: " << op << " (" << synOpToString(op) << ")"
                << ", id: " << id()
                << ", lastCmd: " << isLastCmd
                << " }" << std::endl;
}

int NxtCmd::id() const {
    return 7;
}

void NxtCmd::loadFromVector(const std::vector<uint8_t>& src) {    
    uint32_t word = getWordAtPosition(src, 0);
    nxt = extractField(word, 0, 24);
    rsvd = extractField(word, 24, 4);
    assert(id() == extractField(word, 28, 3));
    isLastCmd = extractField(word, 31, 1);    
}

int NxtCmd::serializedSize() const {
    return 4;
}

void NxtCmd::print() const {
    std::cout << "NxtCmd {"
                << " nxt: 0x" << std::hex << nxt
                << ", rsvd: " << rsvd
                << ", id: " << id()
                << ", lastCmd: " << isLastCmd
                << " }" << std::endl;
}


void SynCmd::serialize(std::vector<uint8_t>& out) const {
    uint32_t word = (msk & 0xFFFF)
                  | ((rsvd & 0xFF) << 16)
                  | ((op & 0xF) << 24)
                  | ((id() & 0x7) << 28)
                  | ((isLastCmd & 0x1) << 31);
    appendWordLE(out, word);
}

void NxtCmd::serialize(std::vector<uint8_t>& out) const {
    uint32_t word = (nxt & 0xFFFFFF)
                  | ((rsvd & 0xF) << 24)
                  | ((id() & 0x7) << 28)
                  | ((isLastCmd & 0x1) << 31);
    appendWordLE(out, word);
}

Cmds parseCommandsUntilUnknown(const std::vector<uint8_t>& src) {
    Cmds cmds;
    size_t position = 0;

    while (src.size() - position >= 4) {

        // figure out the type of command
        uint32_t word = getWordAtPosition(src, position);
        int id = extractField(word, 28, 3);

        std::vector<uint8_t> cmdData(src.begin() + position, src.end());        

        if (id == 1) {
            cmds.push_back(std::make_unique<CfgCmd>());
        } else if (id == 6) {
            cmds.push_back(std::make_unique<SynCmd>());
        } else if (id == 7) {
            cmds.push_back(std::make_unique<NxtCmd>());
        } else {

            LOGV << "Unknown command id: " << id << " at position: " << position;
            return cmds; // unknown command, stop parsing
        }

        cmds.back()->loadFromVector(cmdData);

        position += cmds.back()->serializedSize();
    }
    
    return cmds;
}

void printCommands(const Cmds& cmds) {
    std::cout << "Commands {" << std::endl;    
    for (const auto& cmd : cmds) {
        cmd->print();
    }
    std::cout << "}" << std::endl;
}

std::vector<uint8_t> serializeCommands(const Cmds& cmds) {
    std::vector<uint8_t> out;
    for (const auto& cmd : cmds) {
        cmd->serialize(out);
    }
    return out;
}
