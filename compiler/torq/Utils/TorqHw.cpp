#include "TorqHw.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir::syna::torq;
using namespace std;

namespace mlir::syna::torq {

static llvm::cl::opt<TorqHw::Type> clTorqHw(
    "torq-hw", llvm::cl::desc("Specify the hardware family"),
    llvm::cl::values(
        clEnumValN(TorqHw::SL261X, "SL261X", "Synaptics SL261X (default)"),
        clEnumValN(TorqHw::SR25X, "SR25X", "Synaptics SR25X"),
        clEnumValN(TorqHw::CUSTOM, "custom", "Custom hardware (see torq-hw-custom option)")
    )
);

static llvm::cl::list<size_t> clTorqHwCustom(
    llvm::cl::CommaSeparated, "torq-hw-custom",
    llvm::cl::desc("additional parameters for custom HW selection (see torq-hw option)"),
    llvm::cl::value_desc("<lram_size>,<slice_count>,<available_memory_for_tiling>")
);

class TorqHwCustom : public TorqHw {
  public:
    TorqHwCustom(size_t lramSize, size_t sliceCount, size_t availableMemoryForTiling)
        : TorqHw(CUSTOM), _lramSize(lramSize * 1024), _sliceCount(sliceCount),
          _availableMemoryForTiling(availableMemoryForTiling * 1024) {}

    std::string getString() const override { return "Custom Synaptics hardware configuration"; }

    size_t getLramSize() const override { return _lramSize; }
    size_t getSliceCount() const override { return _sliceCount; }
    size_t getAvailableMemoryForTiling() const override { return _availableMemoryForTiling; }

  private:
    size_t _lramSize;
    size_t _sliceCount;
    size_t _availableMemoryForTiling;
};

class TorqHwSL261X : public TorqHw {
  public:
    TorqHwSL261X() : TorqHw(SL261X) {}
    std::string getString() const { return "Synaptics SL261X SoC family"; }
    size_t getLramSize() const { return 512 * 1024; } // 512 KB
    size_t getSliceCount() const { return 2; }
    size_t getAvailableMemoryForTiling() const { return 400 * 1024; }
};

class TorqHwSR25X : public TorqHw {
  public:
    TorqHwSR25X() : TorqHw(SR25X) {}
    std::string getString() const { return "Synaptics SR25X SoC family"; }
    size_t getLramSize() const { return 256 * 1024; } // 256 KB
    size_t getSliceCount() const { return 1; }
    size_t getAvailableMemoryForTiling() const { return 200 * 1024; }
};

TorqHw::~TorqHw() { delete _instance; }

const TorqHw *TorqHw::_instance = nullptr;

const TorqHw &TorqHw::get() {
    if (_instance)
        return *_instance;
    switch (clTorqHw) {
    case TorqHw::CUSTOM: {
        if (clTorqHwCustom.size() != 3) {
            llvm::errs() << "Invalid custom hardware configuration. Expected 3 values:\n"
                         << "    <lram_size,slice_count,available_memory_for_tiling>\n";
            llvm::report_fatal_error("Invalid custom hardware configuration");
        }
        size_t lramSize = clTorqHwCustom[0];
        size_t sliceCount = clTorqHwCustom[1];
        size_t availableMemoryForTiling = clTorqHwCustom[2];
        llvm::dbgs() << "Using custom hardware configuration: lram_size=" << lramSize << " KB, "
                     << "slice_count=" << sliceCount << ", "
                     << "available_memory_for_tiling=" << availableMemoryForTiling << " KB\n";
        _instance = new TorqHwCustom(lramSize, sliceCount, availableMemoryForTiling);
        break;
    }
    case TorqHw::SL261X:
        _instance = new TorqHwSL261X;
        break;
    case TorqHw::SR25X:
        _instance = new TorqHwSR25X;
        break;
    default:
        // This should never happen if the command line option is correctly defined
        // and the enum values are exhaustive.
        llvm::errs() << "Unknown Torq hardware type: " << clTorqHw << "\n";
        llvm::report_fatal_error("Invalid Torq hardware type specified");
    }
    return *_instance;
}

} // namespace mlir::syna::torq
