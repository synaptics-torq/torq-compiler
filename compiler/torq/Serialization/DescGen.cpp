#include "DescGen.h"

#include "torq_api.h"

#include <iostream>
#include <sstream>

using namespace std;

namespace mlir::syna::torq {

inline void add_hdim(torq_ndl_cmd_t &cmd, char type, int size, int stride) {
    cmd.n[cmd.nld + cmd.nhd] = size;
    cmd.s[cmd.nld + cmd.nhd] = stride;
    cmd.t[cmd.nld + cmd.nhd++] = type;
}

inline void add_ldim(torq_ndl_cmd_t &cmd, char type, int size, int stride) {
    if (type == 'B') {
        if (stride == 0)
            stride = 1;
        assert(stride == 1);
    }
    cmd.n[cmd.nld] = size;
    cmd.s[cmd.nld] = stride;
    cmd.t[cmd.nld++] = type;
}

inline void add_sdim(torq_ndl_cmd_t &cmd, char type, int size, int stride) {
    cmd.n[cmd.nld + cmd.nhd + cmd.nsd] = size;
    cmd.s[cmd.nld + cmd.nhd + cmd.nsd] = stride;
    cmd.t[cmd.nld + cmd.nhd + cmd.nsd++] = type;
}

static std::string toStr(const torq_ndl_cmd_t &ndl) {
    ostringstream os;
    os << "\t";
    if (ndl.nld) {
        os << "l{";
        for (size_t i = 0; i < ndl.nld; ++i)
            os << ndl.t[i] << "[" << ndl.n[i] << "," << ndl.s[i] << "]";
        os << "} ";
    }
    if (ndl.nhd) {
        os << "h{";
        for (size_t i = ndl.nld; i < ndl.nld + ndl.nhd; ++i)
            os << ndl.t[i] << "[" << ndl.n[i] << "," << ndl.s[i] << "]";
        os << "} ";
    }
    if (ndl.nsd) {
        os << "s{";
        for (size_t i = ndl.nld + ndl.nhd; i < ndl.nld + ndl.nhd + ndl.nsd; ++i)
            os << ndl.t[i] << "[" << ndl.n[i] << "," << ndl.s[i] << "]";
        os << "} ";
    }
    bool padding = false;
    for (auto p : ndl.p) {
        if (p != 0) {
            padding = true;
            break;
        }
    }
    if (padding) {
        os << "p{" << ndl.p[0];
        for (size_t i = 1; i < TORQ_MAX_ND; ++i)
            os << "," << ndl.p[i];
        os << "} ";
    }
    return os.str();
}

static std::string toSliceLogStr(const torq_cfg_t &cfg) {
    ostringstream os;
    os << "ACT: ";
    if (cfg.act_clip_min == -128 && cfg.act_clip_max == 127 && cfg.act_zero_point == 0) {
        os << "BYP";
    }
    else {
        os << "min: " << cfg.act_clip_min << " max: " << cfg.act_clip_max
           << " zp: " << cfg.act_zero_point;
    }
    return os.str();
}

static std::string toNssLogStr(const torq_cfg_t &cfg) {
    ostringstream os;
    for (size_t i = 0; i < 2; ++i) {
        os << "  slc " << i << ":: start: " << (int)cfg.slc_start[i]
           << " wait: " << (int)cfg.slc_wait[i] << " cfg_addr: " << cfg.slc_cfg_addr[i] << "\n";
    }
    os << "  dma xr:: start: " << (int)cfg.dma_xr_start << " wait: " << (int)cfg.dma_xr_wait
       << "\n";
    os << "  dma xw:: start: " << (int)cfg.dma_xw_start << " wait: " << (int)cfg.dma_xw_wait
       << "\n";

    os << "css start: " << (int)cfg.css_start << " wait: " << (int)cfg.css_wait << "\n";
    os << "css mbx: " << cfg.css_mbx[0] << " " << cfg.css_mbx[1] << " " << cfg.css_mbx[2] << " "
       << cfg.css_mbx[3] << "\n";

    os << "cdma start: " << (int)cfg.cdma_start << " wait: " << (int)cfg.cdma_wait
       << " src_addr: " << cfg.cdma_src_addr << " dst_addr: " << cfg.cdma_dst_addr
       << " len: " << cfg.cdma_len << "\n";

    return os.str();
}

struct TorqCfg::Private {
    torq_cfg_t cfg{};
};

TorqCfg::~TorqCfg() {}

TorqCfg::TorqCfg() : d{new Private} {}

TorqCfg::TorqCfg(TorqCfg &&c) : d(std::move(c.d)) {}

string SliceTask::toLogStr() const {
    ostringstream os;
    os << "CFG " << toSliceLogStr(d->cfg) << "\n";
    os << ref.toLogStr() << "\n";
    const MemNdl *memNdls[6] = {&dedr, &dedr1, &dewr, &debr, &debr1, &deqw};
    for (const auto ndl : memNdls) {
        if (!ndl->empty())
            os << ndl->toLogStr() << "\n";
    }
    const RegNdl *regNdls[10] = {&acbw, &acbr, &acpw, &acpr, &cedw,
                                 &cedr, &ceww, &cewr, &cepr, &aldw};
    for (const auto &ndl : regNdls)
        os << ndl->toLogStr() << "\n";

    if (!table.empty()) {
        os << "Table size: " << table.size() << "\n";
        os << "Table values: ";
        for (auto v : table) {
            os << (int)v << " ";
        }
        os << "\n";
    }

    return os.str();
}

bool TorqCfg::begin(TorqHandle h, uint32_t desc_laddr, uint32_t desc_xaddr) const {
    // FIXME: ensure taskCfgBegin() is always followed with taskCfgEnd()
    return torq_task_cfg_begin(h, &d->cfg, desc_laddr, desc_xaddr) == 0;
}

int TorqCfg::end(TorqHandle h) const { return torq_task_cfg_end(h); }

string NssTask::toLogStr() const {
    ostringstream os;
    os << "NSS CFG\n" << toNssLogStr(d->cfg) << "\n";
    const DmaNdl *ndls[4] = {&dixr, &dilw, &doxw, &dolr};
    for (const auto ndl : ndls)
        os << ndl->toLogStr() << "\n";
    return os.str();
}

void NssTask::setDmaParams(DmaParams p) {
    d->cfg.dma_xr_start = p.xrStart;
    d->cfg.dma_xw_start = p.xwStart;
    d->cfg.dma_xr_wait = p.xrWait;
    d->cfg.dma_xw_wait = p.xwWait;
}

void NssTask::setSliceParams(size_t sliceId, SliceParams p) {
    d->cfg.slc_start[sliceId] = p.start;
    d->cfg.slc_wait[sliceId] = p.wait;
    d->cfg.slc_cfg_addr[sliceId] = p.cfgAddr;
}

void NssTask::setCdmaParams(CDmaParams p) {
    d->cfg.cdma_start = p.start;
    d->cfg.cdma_wait = p.wait;
    d->cfg.cdma_src_addr = p.srcAddr;
    d->cfg.cdma_dst_addr = p.dstAddr;
    d->cfg.cdma_len = p.len;
}

void NssTask::setCssParams(CssParams p) {
    d->cfg.css_start = p.start;
    d->cfg.css_wait = p.wait;

    d->cfg.css_start_addr = p.startAddr;

    for (size_t i = 0; i < 4; ++i)
        d->cfg.css_mbx[i] = p.mbx[i];
}

int NssTask::compile(TorqHandle h, uint32_t desc_laddr, uint32_t desc_xaddr) const {
    if (!begin(h, desc_laddr, desc_xaddr)) {
        return -1;
    }
    bool success = true;

    // FIXME: we may want to set NDLs in one task and run them later
    if (d->cfg.dma_xr_start == 1) {
        success &= dixr.write(h) && dilw.write(h);
    }

    // FIXME: we may want to set NDLs in one task and run them later
    if (d->cfg.dma_xw_start == 1) {
        success &= dolr.write(h) && doxw.write(h);
    }

    if (!success) {
        return -1;
    }

    return end(h);
}

static uint32_t toChars(torq_hw::ALUOp0Mode hwOp) {
    switch (hwOp) {
    case torq_hw::ALUOp0Mode::MUL:
        return 'MUL';
    case torq_hw::ALUOp0Mode::DBYP:
        return 'DBYP';
    default:
        return 'MUL';
    }
}

static uint32_t toChars(torq_hw::ALUOp1Mode hwOp) {
    switch (hwOp) {
    case torq_hw::ALUOp1Mode::ACC:
        return 'ACC';
    case torq_hw::ALUOp1Mode::SACC:
        return 'SACC';
    case torq_hw::ALUOp1Mode::AMAX:
        return 'AMAX';
    case torq_hw::ALUOp1Mode::AMIN:
        return 'AMIN';
    case torq_hw::ALUOp1Mode::MAX:
        return 'MAX';
    case torq_hw::ALUOp1Mode::MIN:
        return 'MIN';
    case torq_hw::ALUOp1Mode::GT:
        return 'GT';
    case torq_hw::ALUOp1Mode::GE:
        return 'GE';
    case torq_hw::ALUOp1Mode::EQ:
        return 'EQ';
    case torq_hw::ALUOp1Mode::OR:
        return 'OR';
    case torq_hw::ALUOp1Mode::AND:
        return 'AND';
    case torq_hw::ALUOp1Mode::XOR:
        return 'XOR';
    case torq_hw::ALUOp1Mode::NOT:
        return 'NOT';
    case torq_hw::ALUOp1Mode::BOR:
        return 'BOR';
    case torq_hw::ALUOp1Mode::BAND:
        return 'BAND';
    case torq_hw::ALUOp1Mode::BXOR:
        return 'BXOR';
    case torq_hw::ALUOp1Mode::BNOT:
        return 'BNOT';
    case torq_hw::ALUOp1Mode::BYP:
        return 'BYP';
    case torq_hw::ALUOp1Mode::MUL:
        return 'MUL';
    default:
        return 'ACC';
    }
}

static uint32_t toChars(torq_hw::RoundingMode roundingMode) {
    switch (roundingMode) {
    case torq_hw::RoundingMode::OFF:
        return 'OFF';
    case torq_hw::RoundingMode::NTE:
        return 'NTE';
    case torq_hw::RoundingMode::NTO:
        return 'NTO';
    case torq_hw::RoundingMode::NTP:
        return 'NTP';
    case torq_hw::RoundingMode::NTN:
        return 'NTN';
    case torq_hw::RoundingMode::NTI:
        return 'NTI';
    case torq_hw::RoundingMode::NTZ:
        return 'NTZ';
    case torq_hw::RoundingMode::DBL:
        return 'DBL';
    default:
        return 'NTP';
    }
}

static uint32_t toChars(torq_hw::WeightFormat weightFormat) {
    switch (weightFormat) {
    case torq_hw::WeightFormat::SI:
        return 'SI';
    case torq_hw::WeightFormat::UI:
        return 'UI';
    case torq_hw::WeightFormat::FP:
        return 'FP';
    case torq_hw::WeightFormat::BF:
        return 'BF';
    case torq_hw::WeightFormat::NF:
        return 'NF';
    default:
        return 'SI';
    }
}

static uint32_t toChars(torq_hw::NumberFormat numberFormat) {
    switch (numberFormat) {
    case torq_hw::NumberFormat::I:
        return 'I';
    case torq_hw::NumberFormat::BF:
        return 'BF';
    default:
        return 'I';
    }
}

static uint32_t toChars(torq_hw::ACTMode actMode) {
    switch (actMode) {
    case torq_hw::ACTMode::ABS:
        return 'ABS';
    case torq_hw::ACTMode::NEG:
        return 'NEG';
    case torq_hw::ACTMode::CLZ:
        return 'CLZ';
    case torq_hw::ACTMode::CEL:
        return 'CEL';
    case torq_hw::ACTMode::FLR:
        return 'FLR';
    case torq_hw::ACTMode::I2F:
        return 'I2F';
    case torq_hw::ACTMode::F2I:
        return 'F2I';
    case torq_hw::ACTMode::LSL:
        return 'LSL';
    case torq_hw::ACTMode::LSR:
        return 'LSR';
    case torq_hw::ACTMode::ASR:
        return 'ASR';
    default:
        return 'ACT';
    }
}

SliceTask::SliceTask(AluMode m) : aluMode(m) { d->cfg.stride = 1; }

void SliceTask::setAluOp0(torq_hw::ALUOp0Mode mode) {
    // Only the first element is used
    d->cfg.alu_op0_mode[0] = toChars(mode);
}

void SliceTask::setAluOp0(
    torq_hw::ALUOp0Mode a0c0, torq_hw::ALUOp0Mode a0c1, torq_hw::ALUOp0Mode a1c0,
    torq_hw::ALUOp0Mode a1c1
) {
    d->cfg.alu_op0_mode[0] = toChars(a0c0);
    // alu_op0_mode[1/2/3] are reserved and must be set to 0.
    // d->cfg.alu_op0_mode[1] = toChars(a0c1);
    // d->cfg.alu_op0_mode[2] = toChars(a1c0);
    // d->cfg.alu_op0_mode[3] = toChars(a1c1);
}

void SliceTask::setAluOp1(torq_hw::ALUOp1Mode mode) {
    // Only the first element is used
    d->cfg.alu_op1_mode[0] = toChars(mode);
}

void SliceTask::setAluOp1(
    torq_hw::ALUOp1Mode a0c0, torq_hw::ALUOp1Mode a0c1, torq_hw::ALUOp1Mode a1c0,
    torq_hw::ALUOp1Mode a1c1
) {
    d->cfg.alu_op1_mode[0] = toChars(a0c0);
    // alu_op1_mode[1/2/3] are reserved and must be set to 0.
    // d->cfg.alu_op1_mode[1] = toChars(a0c1);
    // d->cfg.alu_op1_mode[2] = toChars(a1c0);
    // d->cfg.alu_op1_mode[3] = toChars(a1c1);
}

void SliceTask::setAluDUnsigned(AluOpUnsigned options) {
    d->cfg.alu_d_unsigned = 0;
    if (options.a0c0)
        d->cfg.alu_d_unsigned |= 0b0001;
    if (options.a0c1)
        d->cfg.alu_d_unsigned |= 0b0010;
    if (options.a1c0)
        d->cfg.alu_d_unsigned |= 0b0100;
    if (options.a1c1)
        d->cfg.alu_d_unsigned |= 0b1000;
}

void SliceTask::setAluWUnsigned(AluOpUnsigned options) {
    d->cfg.alu_w_unsigned = 0;
    if (options.a0c0)
        d->cfg.alu_w_unsigned |= 0b0001;
    if (options.a0c1)
        d->cfg.alu_w_unsigned |= 0b0010;
    if (options.a1c0)
        d->cfg.alu_w_unsigned |= 0b0100;
    if (options.a1c1)
        d->cfg.alu_w_unsigned |= 0b1000;
}

void SliceTask::setAluDUnsigned(uint32_t options) { d->cfg.alu_d_unsigned = options; }

void SliceTask::setAluWUnsigned(uint32_t options) { d->cfg.alu_w_unsigned = options; }

void SliceTask::setActivation(const ActivationParams &p) {
    for (size_t i = 0; i < 4; ++i)
        d->cfg.act_lsh[i] = p.leftShift[i];
    d->cfg.act_rsh = p.rightShift;
    d->cfg.act_clip_min = p.clipMin;
    d->cfg.act_clip_max = p.clipMax;
    d->cfg.act_zero_point = p.zeroPoint;
    d->cfg.act_mode = toChars(p.actMode);
}

void SliceTask::setKernel(const KernelDim &p) {
    d->cfg.kernel_left = p.left;
    d->cfg.kernel_right = p.right;
    d->cfg.kernel_top = p.top;
    d->cfg.kernel_bottom = p.bottom;
}

void SliceTask::setStride(int32_t stride, int32_t strideOffset) {
    d->cfg.stride = stride;
    d->cfg.stride_offset = strideOffset;
}

void SliceTask::setTable(ArrayRef<int32_t> arraytable) {
    if (arraytable.empty()) {
        d->cfg.table = nullptr;
        table.clear();
        return;
    }

    table.assign(arraytable.begin(), arraytable.end());
    d->cfg.table = reinterpret_cast<uint8_t *>(table.data());
}

void SliceTask::setPad(const Pad &p) {
    d->cfg.pad_left = p.left;
    d->cfg.pad_right = p.right;
    d->cfg.pad_top = p.top;
    d->cfg.pad_bottom = p.bottom;
    d->cfg.pad_value = p.value;
}

void SliceTask::setActRoundingMode(torq_hw::RoundingMode mode) {
    d->cfg.act_round_mode = toChars(mode);
}

void SliceTask::setAluActDisable(uint32_t alu_disable, uint32_t act_disable) {
    d->cfg.alu_disable = alu_disable;
    d->cfg.act_disable = act_disable;
}

void SliceTask::setWeightFormat(torq_hw::WeightFormat format) { d->cfg.w_format = toChars(format); }

void SliceTask::setAluActNumberFormat(
    torq_hw::NumberFormat aluFormat, torq_hw::NumberFormat actFormat
) {
    d->cfg.act_format = toChars(actFormat);
    d->cfg.alu_format = toChars(aluFormat);
}

void SliceTask::setActSumBits(uint32_t actSumBits) { d->cfg.act_sum_bits = actSumBits; }

bool SliceTask::write(
    TorqHandle h, int slcId, uint32_t &lramAddr, uint32_t &xramAddr, uint32_t &totalCfgSize,
    uint32_t &totalNdlSize
) const {
    if (!begin(h, TORQ_LADDR_APPEND, TORQ_XADDR_APPEND)) {
        return false;
    }
    // REF NDL is a special case and must be written first
    if (ref.write(h, 0, 0) < 0) {
        return false;
    }
    const MemNdl *memNdls[6] = {&dedr, &dedr1, &dewr, &debr, &debr1, &deqw};

    totalNdlSize = 0;

    for (const auto ndl : memNdls) {
        if (ndl->empty()) {
            // Skip serialization of unused NDLs
            continue;
        }
        auto ndlSize = ndl->write(h, lramAddr, xramAddr);
        if (ndlSize < 0) {
            return ndlSize;
        }
        xramAddr += ndlSize;
        lramAddr += ndlSize;
        totalNdlSize += ndlSize;
    }
    const RegNdl *regNdls[10] = {&acbw, &acbr, &acpw, &acpr, &cedw,
                                 &cedr, &ceww, &cewr, &cepr, &aldw};
    for (const auto &ndl : regNdls) {

        if (ndl->empty()) {
            // Skip serialization of unused NDLs
            continue;
        }

        if (!ndl->write(h)) {
            return -1;
        }
    }
    if (wMem.data)
        torq_xram_write(h, slcId, 'W', wMem.addr, wMem.size, const_cast<void *>(wMem.data));
    if (bMem.data)
        torq_xram_write(h, slcId, 'B', bMem.addr, bMem.size, const_cast<void *>(bMem.data));
    if (dMem.data)
        torq_xram_write(h, slcId, 'D', dMem.addr, dMem.size, const_cast<void *>(dMem.data));

    int maybeTotalCfgSize = end(h);

    if (maybeTotalCfgSize < 0) {
        return false;
    }

    totalCfgSize = maybeTotalCfgSize;
    return true;
}

template <typename T> string ndlTypeToStr(T d) {
    char *t = (char *)(&d);
    return t[3] ? string{t[3], t[2], t[1], t[0]} : string{t[2], t[1], t[0]};
}

struct Ndl::Private {
    torq_ndl_cmd_t cmd{};
};

Ndl::Ndl() : d{new Private} {}
Ndl::~Ndl() {}
Ndl::Ndl(Ndl &&c) : d(std::move(c.d)) {}

bool Ndl::empty() const { return d->cmd.nld + d->cmd.nhd + d->cmd.nsd == 0; }

void Ndl::setBaseAddress(uint32_t addr) { d->cmd.base_addr = addr; }

void Ndl::setSyncMode(uint8_t sync_mode, uint8_t sync_nhd) {
    d->cmd.sync_mode = sync_mode;
    d->cmd.sync_nhd = sync_nhd;
}

string MemNdl::toLogStr() const {
    ostringstream os;
    os << ndlTypeToStr(t) << " " << toStr(d->cmd) << " addr: " << d->cmd.base_addr;
    return os.str();
}

static char toChar(torq_hw::MemDimTag tag) {
    switch (tag) {
    case torq_hw::MemDimTag::X:
        return 'X';
    case torq_hw::MemDimTag::Y:
        return 'Y';
    case torq_hw::MemDimTag::A:
        return 'A';
    case torq_hw::MemDimTag::I:
        return 'I';
    case torq_hw::MemDimTag::J:
        return 'J';
    case torq_hw::MemDimTag::U:
        return 'U';
    case torq_hw::MemDimTag::V:
        return 'V';
    case torq_hw::MemDimTag::G:
        return 'G';
    case torq_hw::MemDimTag::O:
        return 'O';
    case torq_hw::MemDimTag::B:
        return 'B';
    case torq_hw::MemDimTag::D:
        return 'D';
    }
}

void MemNdl::addHdim(torq_hw::MemDimTag tag, int size, int stride) {
    add_hdim(d->cmd, toChar(tag), size, stride);
}
void MemNdl::addLdim(torq_hw::MemDimTag tag, int size, int stride) {
    add_ldim(d->cmd, toChar(tag), size, stride);
}
void MemNdl::addSdim(torq_hw::MemDimTag tag, int size, int stride) {
    add_sdim(d->cmd, toChar(tag), size, stride);
}

int MemNdl::write(TorqHandle h, uint32_t lramAddr, uint32_t xramAddr) const {
    return torq_ndl_desc_write(h, static_cast<uint32_t>(t), setId, &d->cmd, lramAddr, xramAddr);
}

string RegNdl::toLogStr() const {
    ostringstream os;
    os << ndlTypeToStr(t) << " " << toStr(d->cmd);
    return os.str();
}

static char toChar(torq_hw::RegDimTag tag) {
    switch (tag) {
    case torq_hw::RegDimTag::I:
        return 'I';
    case torq_hw::RegDimTag::B:
        return 'B';
    case torq_hw::RegDimTag::J:
        return 'J';
    case torq_hw::RegDimTag::D:
        return 'D';
    case torq_hw::RegDimTag::K:
        return 'K';
    case torq_hw::RegDimTag::G:
        return 'G';
    case torq_hw::RegDimTag::L:
        return 'L';
    case torq_hw::RegDimTag::S:
        return 'S';
    case torq_hw::RegDimTag::M:
        return 'M';
    case torq_hw::RegDimTag::W:
        return 'W';
    case torq_hw::RegDimTag::N:
        return 'N';
    case torq_hw::RegDimTag::T:
        return 'T';
    }
}

void RegNdl::addHdim(torq_hw::RegDimTag tag, int size, int stride) {
    if (t == CEDR && tag == torq_hw::RegDimTag::S && stride == 0) {
        /// \note: stride is used here for an assertion in torq_api.c
        ///        It looks like this is used to make sure that stride==1 in case of convolution
        ///        \sa ConvPattern::transform() in ConvPattern.c
        ///        \sa note in_reg_ndl_desc_gen() in synapi_api.c
        stride = 1;
    }
    add_hdim(d->cmd, toChar(tag), size, stride);
}
void RegNdl::addLdim(torq_hw::RegDimTag tag, int size, int stride) {
    add_ldim(d->cmd, toChar(tag), size, stride);
}
void RegNdl::addSdim(torq_hw::RegDimTag tag, int size, int stride) {
    add_sdim(d->cmd, toChar(tag), size, stride);
}

bool RegNdl::write(TorqHandle h) const {
    // Register NDL have no size, torq_ndl_desc_write() will return 0 on success
    return torq_ndl_desc_write(h, static_cast<uint32_t>(t), 0, &d->cmd, 0, 0) == 0;
}

bool DmaNdl::write(TorqHandle h) const {
    // Register NDL have no size, torq_ndl_desc_write() will return 0 on success
    return torq_ndl_desc_write(h, static_cast<uint32_t>(t), 0, &d->cmd, 0, 0) == 0;
}

string DmaNdl::toLogStr() const {
    ostringstream os;
    os << ndlTypeToStr(t) << " " << toStr(d->cmd) << " addr: " << d->cmd.base_addr;
    return os.str();
}
void DmaNdl::addDim(int size, int stride) { add_hdim(d->cmd, dimType++, size, stride); }

bool DescGen::open(const std::string &dump_path) {
    _d = torq_open(dump_path.c_str());
    return _d;
}

bool DescGen::close() {
    bool ret = torq_close(_d) == 0;
    _d = nullptr;
    return ret;
}

bool DescGen::nssBegin(uint32_t lramAddr, uint32_t xramAddr) {
    _slcId = -1;
    // FIXME: ensure nssBegin() is always followed with nssEnd()
    return torq_cfg_begin(_d, -1, lramAddr, xramAddr) == 0;
}

int DescGen::nssEnd() {

    int ret = torq_cfg_end(_d);

    // For file dump purpose
    if (ret >= 0)
        torq_run(_d);

    return ret;
}

bool DescGen::beginCfg(int slcIdx, uint32_t lramAddr, uint32_t xramAddr) {
    _slcId = slcIdx;
    // FIXME: ensure cfgBegin() is always followed with cfgEnd()
    return torq_cfg_begin(_d, slcIdx, lramAddr, xramAddr) == 0;
}

int DescGen::endCfg() { return torq_cfg_end(_d); }

const torq_bitstream_segment_t *DescGen::getBitstream() const {
    torq_bitstream_segment_t *bitstream{};
    torq_get_bitstream(_d, &bitstream);
    return bitstream;
}

} // namespace mlir::syna::torq
