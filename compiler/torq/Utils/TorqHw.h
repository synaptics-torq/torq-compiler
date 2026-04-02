#pragma once

#include "torq/Codegen/css_bootstrap/css_kernel_riscv.h"

#include <cstdint>
#include <string>

namespace mlir::syna::torq {

class TorqHw {
  public:
    static const TorqHw &get();

    // Get the hardware name
    std::string getName() const { return _name; }

    // Get the hardware information as a string
    std::string getDescription() const { return _description; }

    // Get LRAM size in bytes
    size_t getLramSize() const { return _lramSize; }

    // Get Number of Slices
    size_t getSliceCount() const { return _sliceCount; }

    // Get Available Memory for Tiling in bytes
    size_t getAvailableMemoryForTiling() const { return _availableMemoryForTiling; }

    // Get the CSS configuration name
    std::string getCSSConfigName() const { return _cssConfigName; }

    CssConfig getCssConfig() const;

    // Get the NSS configuration name
    std::string getNSSConfigName() const { return _nssConfigName; }

    // Get the target host triple for cross-compilation
    std::string getHostTriple() const { return _hostTriple; }

    // Get the target host CPU for cross-compilation
    std::string getHostCpu() const { return _hostCpu; }

    // Get the target host CPU features for cross-compilation
    std::string getHostCpuFeatures() const { return _hostCpuFeatures; }

    // Get the effective DMA throughput in bytes per cycle (theoretical * dma_factor)
    double getDmaThroughputBytesPerCycle() const {
        return _dmaTheoreticalBytesPerCycle * _dmaFactor;
    }

    // Get the theoretical peak DMA throughput in bytes per cycle
    double getDmaTheoreticalBytesPerCycle() const { return _dmaTheoreticalBytesPerCycle; }

    // Get the DMA efficiency factor (practical ratio, typically 0.60-0.70)
    double getDmaFactor() const { return _dmaFactor; }

    TorqHw(
        std::string name, std::string description, size_t lramSize, size_t sliceCount,
        size_t availableMemoryForTiling, std::string cssConfigName, std::string nssConfigName,
        double dmaTheoreticalBytesPerCycle, double dmaFactor = 1.0,
        std::string hostTriple = "native", std::string hostCpu = "host",
        std::string hostCpuFeatures = "host"
    )
        : _name(name), _description(description), _lramSize(lramSize), _sliceCount(sliceCount),
          _availableMemoryForTiling(availableMemoryForTiling), _cssConfigName(cssConfigName),
          _nssConfigName(nssConfigName), _hostTriple(hostTriple), _hostCpu(hostCpu),
          _hostCpuFeatures(hostCpuFeatures),
          _dmaTheoreticalBytesPerCycle(dmaTheoreticalBytesPerCycle), _dmaFactor(dmaFactor) {}

    TorqHw() = default;

  private:
    std::string _name{};
    std::string _description{};
    std::string _version{};
    size_t _lramSize = 0;
    size_t _sliceCount = 0;
    size_t _availableMemoryForTiling = 0;
    std::string _cssConfigName{};
    std::string _nssConfigName{};
    std::string _hostTriple{};
    std::string _hostCpu{};
    std::string _hostCpuFeatures{};
    double _dmaTheoreticalBytesPerCycle = 0.0;
    double _dmaFactor = 1.0;

    static const TorqHw *_instance;
};

} // namespace mlir::syna::torq
