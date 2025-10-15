#pragma once

#include "torq_bitstream.h"

#include "torq/Dialect/TorqHW/TorqHWAttrs.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"

#include <cassert>
#include <memory>
#include <string>
#include <vector>

namespace mlir::syna::torq {

typedef void *TorqHandle;

enum AddressConstants : uint32_t {
    NONE = 1,             ///< No address
    APPEND = (uint32_t)~0 ///< Append to the current address
};

class Ndl {
  public:
    Ndl();
    ~Ndl();
    Ndl(Ndl &&d);
    /// Set memory NDL base address (uneeded for register NDL)
    /// \param addr: base address of the NDL
    void setBaseAddress(uint32_t addr);

    /// Set Multi-agent sync mode
    void setSyncMode(uint8_t sync_mode, uint8_t sync_nhd);

    /// \return true if the NDL is empty
    bool empty() const;

  protected:
    struct Private;
    std::unique_ptr<Private> d;
};

class RegNdl : public Ndl {
  public:
    /// Simple NDL descriptors in CFG_REGS
    enum Type : uint32_t {
        ACBW = 'ACBW', ///< Configures ACT.ACBW agent; DE.BBUS to BREG
        ACBR = 'ACBR', ///< Configures ACT.ACBR agent; BREG to ACT
        ACPW = 'ACPW', ///< Configures CE.ACPW agent;  PBUS to ACT
        ACPR = 'ACPR', ///< Configures CE.ACPR agent;  PRAM to PBUS
        CEDW = 'CEDW', ///< Configures CE.CEDW agent;  DBUS to IRAM
        CEDR = 'CEDR', ///< Configures CE.CEDR agent;  IRAM to ALU (read from IRAM)
        CEWW = 'CEWW', ///< Configures CE.CEWW agent;  WBUS to WRAM
        CEWR = 'CEWR', ///< Configures CE.CEWR agent;  WRAM to ALU
        CEPR = 'CEPR', ///< Configures CE.CEPR agent;  PRAM to ALU
        CEPW = 'CEPW', ///< Configures CE.CEPW agent; ALU to PRAM
        ALDW = 'ALDW'  ///< Configures CE.ALDW agent; IRAM to ALU (write to ALU)
    };

    const Type t;
    RegNdl(Type type) : Ndl(), t(type) {}
    std::string toLogStr() const;
    void addHdim(torq_hw::RegDimTag tag, int size = 1, int stride = 0);
    void addLdim(torq_hw::RegDimTag tag, int size = 1, int stride = 0);
    void addSdim(torq_hw::RegDimTag tag, int size = 1, int stride = 0);
    /// \return true upon success
    bool write(TorqHandle h) const;
};

class MemNdl : public Ndl {
  public:
    /// Full NDL descriptors in LRAM
    enum Type : uint32_t {
        /// Describes the original compute task by definition; Some dimensions affect CFG_REGS
        /// \warning Must be written first
        REF = 'REF',
        DEDR = 'DEDR', ///< LRAM to DBUS
        DEWR = 'DEWR', ///< LRAM to WBUS
        DEBR = 'DEBR', ///< LRAM to DE.BBUS
        DEQW = 'DEQW'  ///< DE.QBUS to LRAM
    };

    Type t;
    int setId;
    MemNdl(Type type, int sid = 0) : t(type), setId(sid) {}
    std::string toLogStr() const;
    void addHdim(torq_hw::MemDimTag tag, int size = 1, int stride = 0);
    void addLdim(torq_hw::MemDimTag tag, int size = 1, int stride = 0);
    void addSdim(torq_hw::MemDimTag tag, int size = 1, int stride = 0);
    /// \return length of generated NDL in memory
    int write(TorqHandle h, uint32_t lramAddr, uint32_t xramAddr) const;
};

class DmaNdl : public Ndl {
  public:
    /// Simple NDL descriptors in CFG_REGS
    enum Type : uint32_t {
        DIXR = 'DIXR', ///< XRAM to DMA.XR
        DILW = 'DILW', ///<
        DOXW = 'DOXW', ///<
        DOLR = 'DOLR'  ///<
    };
    const Type t;
    DmaNdl(Type type) : t(type) {}
    std::string toLogStr() const;
    void addDim(int size, int stride);
    /// \return true upon success
    bool write(TorqHandle h) const;

  private:
    char dimType = '0';
};

class TorqCfg {
  public:
    TorqCfg();
    ~TorqCfg();

    TorqCfg(TorqCfg &&d);

    bool begin(TorqHandle h, uint32_t desc_laddr, uint32_t desc_xaddr) const;

    /// \return total length of generated task descriptor
    int end(TorqHandle h) const;

  protected:
    struct Private;
    std::unique_ptr<Private> d;
};

/// Convolution padding shape and value
struct Pad {
    int32_t left;
    int32_t right;
    int32_t top;
    int32_t bottom;
    int32_t value;
};

/// Convolution kernel shape
struct KernelDim {
    uint32_t left;
    uint32_t right;
    uint32_t top;
    uint32_t bottom;
};

struct ActivationParams {
    uint32_t leftShift[4];
    int32_t rightShift;
    int32_t zeroPoint;
    int32_t clipMin;
    int32_t clipMax;
    torq_hw::ACTMode actMode;
};

struct SliceParams {
    char start;
    char wait;
    uint32_t cfgAddr;
};

struct CssParams {
    char start;
    char wait;
    uint32_t startAddr;
    uint32_t mbx[4];
};

struct CDmaParams {
    bool start;
    bool wait;
    int len;
    int srcAddr;
    int dstAddr;
};

struct DmaParams {
    bool xrStart;
    bool xwStart;
    bool xrWait;
    bool xwWait;
};

class NssTask : public TorqCfg {
  public:
    DmaNdl dixr{DmaNdl::DIXR};
    DmaNdl dilw{DmaNdl::DILW};
    DmaNdl doxw{DmaNdl::DOXW};
    DmaNdl dolr{DmaNdl::DOLR};

    NssTask() = default;

    int compile(
        TorqHandle h, uint32_t desc_laddr = AddressConstants::APPEND,
        uint32_t desc_xaddr = AddressConstants::APPEND
    ) const;
    std::string toLogStr() const;

    void setSliceParams(size_t sliceId, SliceParams p);
    void setDmaParams(DmaParams p);
    void setCdmaParams(CDmaParams p);
    void setCssParams(CssParams p);
};

enum class AluMode : size_t { UNDEFINED, G64x4 = 4, G32x8 = 8, G16x16 = 16 };

struct DumpMemInfo {
    uint32_t addr{0xDEADBEEF};
    const void *data{};
    size_t size{};
};

/// List of descriptors for a given ALU execution
class SliceTask : public TorqCfg {
  public:
    const AluMode aluMode;

    /// REF is a special NDL and must be the first one written
    MemNdl ref{MemNdl::REF};
    /// Other memory based flexible NDLs
    std::vector<MemNdl> memndls;
    std::vector<RegNdl> regndls;

    MemNdl dedr{MemNdl::DEDR};
    MemNdl dedr1{MemNdl::DEDR, 1};
    MemNdl dewr{MemNdl::DEWR};
    MemNdl debr{MemNdl::DEBR};
    MemNdl debr1{MemNdl::DEBR, 1};
    MemNdl deqw{MemNdl::DEQW};
    /// Fixed register based NDLs
    RegNdl acbw{RegNdl::ACBW};
    RegNdl acbr{RegNdl::ACBR};
    RegNdl acpw{RegNdl::ACPW};
    RegNdl acpr{RegNdl::ACPR};
    RegNdl cedw{RegNdl::CEDW};
    RegNdl cedr{RegNdl::CEDR};
    RegNdl ceww{RegNdl::CEWW};
    RegNdl cewr{RegNdl::CEWR};
    RegNdl cepr{RegNdl::CEPR};
    RegNdl aldw{RegNdl::ALDW};
    std::vector<int32_t> table; ///< Optional activation table

    DumpMemInfo wMem;
    DumpMemInfo bMem;
    DumpMemInfo dMem;

    SliceTask(AluMode m = AluMode::UNDEFINED);

    bool isValid() const { return aluMode != AluMode::UNDEFINED; }

    std::string toLogStr() const;

    /// Set the ALU Op0 mode
    void setAluOp0(torq_hw::ALUOp0Mode mode);
    /// 4 set of ALU settings for { a0c0, a0c1, a1c0, a1c1 }, a0: even ALUs, a1: odd ALUs, c0: even
    /// cycles, c1: odd cycles
    void setAluOp0(
        torq_hw::ALUOp0Mode a0c0, torq_hw::ALUOp0Mode a0c1, torq_hw::ALUOp0Mode a1c0,
        torq_hw::ALUOp0Mode a1c1
    );
    void setAluOp1(torq_hw::ALUOp1Mode mode);
    /// 4 set of ALU settings for { a0c0, a0c1, a1c0, a1c1 }, a0: even ALUs, a1: odd ALUs, c0: even
    /// cycles, c1: odd cycles
    void setAluOp1(
        torq_hw::ALUOp1Mode a0c0, torq_hw::ALUOp1Mode a0c1, torq_hw::ALUOp1Mode a1c0,
        torq_hw::ALUOp1Mode a1c1
    );

    struct AluOpUnsigned {
        bool a0c0; // even ALUs, even cycles
        bool a0c1; // even ALUs, odd cycles
        bool a1c0; // odd ALUs, even cycles
        bool a1c1; // odd ALUs, odd cycles
    };
    void setAluDUnsigned(AluOpUnsigned options);
    void setAluWUnsigned(AluOpUnsigned options);
    void setAluDUnsigned(uint32_t options);
    void setAluWUnsigned(uint32_t options);

    void setActivation(const ActivationParams &p);
    void setKernel(const KernelDim &p);
    void setStride(int32_t stride, int32_t strideOffset);
    void setTable(ArrayRef<int32_t> table);
    void setPad(const Pad &p);

    void setActRoundingMode(torq_hw::RoundingMode mode);
    void setAluActDisable(uint32_t alu_disable, uint32_t act_disable);
    void setWeightFormat(torq_hw::WeightFormat format);
    void setAluActNumberFormat(torq_hw::NumberFormat aluFormat, torq_hw::NumberFormat actFormat);
    void setActSumBits(uint32_t actSumBits);

    /// Write a Torq task
    /// \param xramAddr will be updated with the next available XRAM address
    /// \param lramAddr will be updated with the next available LRAM address
    /// \param totalCfgSize will be updated with the total length of the generated CFG task
    /// descriptor \param totalNdlSize will be updated with the total length of the generated NDLs
    /// \return true if success, otherwise false
    bool write(
        TorqHandle h, int slcId, uint32_t &lramAddr, uint32_t &xramAddr, uint32_t &totalCfgSize,
        uint32_t &totalNdlSize
    ) const;
};

/// Torq Descriptor Generator
class DescGen {
  public:
    DescGen() : _d(nullptr) {}
    ~DescGen() {
        if (_d)
            close();
    }
    bool open(const std::string &dump_path = "");

    bool close();

    TorqHandle get() const { return _d; }

    bool nssBegin(uint32_t lramAddr, uint32_t xramAddr);

    /// \return total length of generated descriptor
    int nssEnd();

    bool beginCfg(int slcIdx, uint32_t lramAddr, uint32_t xramAddr);

    int getCurrentSliceId() const { return _slcId; }

    /// \return total length of generated descriptor
    int endCfg();

    const torq_bitstream_segment_t *getBitstream() const;

  private:
    TorqHandle _d;
    int _slcId;
};

} // namespace mlir::syna::torq
