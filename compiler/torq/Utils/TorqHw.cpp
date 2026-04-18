#include "TorqHw.h"

#include "torq/Codegen/css_bootstrap/css_kernel_riscv.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir::syna::torq;
using namespace std;

namespace mlir::syna::torq {

llvm::SmallVector<TorqHw, 1> hwTypes = {TorqHw(
    "SL2610", "Synaptics SL2610 SoC family", 512 * 1024, 2, 498 * 1024, "coral_v1", "nss_v1", 8,
    0.65, "aarch64-unknown-linux-gnu", "generic", "+neon,+crypto,+crc,+dotprod,+rdm,+rcpc,+lse,+sve"
)};

#define TORQ_CUSTOM_FORMAT                                                                         \
    "<lram_size_kb>:<slice_count>:<available_memory_for_tiling_kb>:<css_features>:<nss_features>"  \
    "[:<dma_theoretical_throughput>[:<dma_factor>:<host_triple>:<host_cpu>:<host_cpu_features>]]"

struct TorqHwParser : public llvm::cl::parser<TorqHw> {

    using llvm::cl::parser<TorqHw>::parser;

    bool parse(
        llvm::cl::Option &O, llvm::StringRef ArgName, const llvm::StringRef &ArgValue, TorqHw &Val
    ) {

        llvm::SmallVector<llvm::StringRef, 8> parts;
        ArgValue.split(parts, ':');

        if (parts.size() == 1) {

            if (ArgValue == "list") {
                llvm::outs() << "Available Torq Hardware Types:\n";
                for (const auto &hw : hwTypes) {
                    llvm::outs() << "  " << hw.getName() << ": " << hw.getDescription() << "\n";
                }

                exit(0);
            }

            for (const auto &hw : hwTypes) {
                if (hw.getName() == ArgValue) {
                    Val = hw;
                    return false;
                }
            }

            return O.error(
                ArgName, " expects a valid hardware name (pass 'list' to see available types) or a "
                         "custom hardware specification\n"
            );
        }

        if (parts.size() == 5) {
            parts.push_back("8"); // dma_theoretical_throughput default
        }
        if (parts.size() == 6) {
            parts.push_back("1.0"); // dma_factor default
        }
        if (parts.size() == 7) {
            parts.push_back("native"); // host triple default
        }
        if (parts.size() == 8) {
            parts.push_back("host"); // host cpu default
        }
        if (parts.size() == 9) {
            parts.push_back("host"); // host cpu features default
        }
        if (parts.size() != 10) {
            return O.error(ArgName, " custom hardware specification format: " TORQ_CUSTOM_FORMAT);
        }

        std::string name = "custom";
        std::string description = "Custom Torq Hardware";

        size_t lram_size_kb = 0;

        // check that lram_size is a valid number
        if (parts[0].getAsInteger(10, lram_size_kb)) {
            return O.error(ArgName, " LRAM size must be a valid number\n");
        }

        size_t lram_size = lram_size_kb * 1024;

        size_t slice_count;

        if (parts[1].getAsInteger(10, slice_count)) {
            return O.error(ArgName, " Slice count must be a valid number\n");
        }

        size_t available_memory_for_tiling_kb;

        if (parts[2].getAsInteger(10, available_memory_for_tiling_kb)) {
            return O.error(ArgName, " Available memory for tiling must be a valid number\n");
        }

        size_t available_memory_for_tiling = available_memory_for_tiling_kb * 1024;

        std::string cpu_config = parts[3].str();
        std::string nss_config = parts[4].str();

        double dma_theoretical_throughput;
        if (parts[5].getAsDouble(dma_theoretical_throughput)) {
            return O.error(ArgName, " DMA theoretical throughput must be a valid number\n");
        }
        double dma_factor;
        if (parts[6].getAsDouble(dma_factor)) {
            return O.error(ArgName, " DMA factor must be a valid number\n");
        }
        std::string host_triple = parts[7].str();
        std::string host_cpu = parts[8].str();
        std::string host_cpu_features = parts[9].str();

        llvm::outs() << "Custom Torq Hardware Configuration:\n";
        llvm::outs() << "  LRAM Size: " << lram_size << " bytes\n";
        llvm::outs() << "  Slice Count: " << slice_count << "\n";
        llvm::outs() << "  Available Memory for Tiling: " << available_memory_for_tiling
                     << " bytes\n";
        llvm::outs() << "  CSS Config: " << cpu_config << "\n";
        llvm::outs() << "  NSS Config: " << nss_config << "\n";
        llvm::outs() << "  Host Triple: " << host_triple << "\n";
        llvm::outs() << "  Host CPU: " << host_cpu << "\n";
        llvm::outs() << "  Host CPU Features: " << host_cpu_features << "\n";
        llvm::outs() << "  DMA Theoretical Throughput: " << dma_theoretical_throughput
                     << " bytes/cycle\n";
        llvm::outs() << "  DMA Factor: " << dma_factor << "\n";

        Val = TorqHw(
            name, description, lram_size, slice_count, available_memory_for_tiling, cpu_config,
            nss_config, dma_theoretical_throughput, dma_factor, host_triple, host_cpu,
            host_cpu_features
        );

        // no error
        return false;
    }
};

static llvm::cl::opt<TorqHw, false, TorqHwParser> clTorqHw(
    "torq-hw", llvm::cl::init(hwTypes[0]),
    llvm::cl::desc("Specify the target hardware (pass 'list' to list all available targets) or a "
                   "custom target specification as " TORQ_CUSTOM_FORMAT)
);

static llvm::cl::opt<bool> clEnableCSSForQemu(
    "torq-css-qemu", llvm::cl::desc("Create CSS binaries suitable for QEMU emulation"),
    llvm::cl::init(false)
);

static std::optional<TorqHw> instance{std::nullopt};

const TorqHw &TorqHw::get() {

    if (!instance.has_value()) {

        if (clEnableCSSForQemu) {
            instance = TorqHw(
                clTorqHw.getName(), clTorqHw.getDescription(), clTorqHw.getLramSize(),
                clTorqHw.getSliceCount(), clTorqHw.getAvailableMemoryForTiling(),
                clTorqHw.getCSSConfigName() + "_qemu", clTorqHw.getNSSConfigName(),
                clTorqHw.getDmaTheoreticalBytesPerCycle(), clTorqHw.getDmaFactor()
            );
        }
        else {
            instance = clTorqHw;
        }
    }

    return *instance;
}

CssConfig TorqHw::getCssConfig() const {
    auto cssConfig = getCssConfigByName(getCSSConfigName());
    if (!cssConfig) {
        llvm::errs() << "Unable to find css config " << TorqHw::get().getCSSConfigName();
        exit(1);
    }
    return cssConfig.value();
}

} // namespace mlir::syna::torq
