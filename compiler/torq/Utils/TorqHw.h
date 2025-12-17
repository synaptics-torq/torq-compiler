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

    TorqHw(
        std::string name, std::string description, size_t lramSize, size_t sliceCount,
        size_t availableMemoryForTiling, std::string cssConfigName, std::string nssConfigName
    )
        : _name(name), _description(description), _lramSize(lramSize), _sliceCount(sliceCount),
          _availableMemoryForTiling(availableMemoryForTiling), _cssConfigName(cssConfigName),
          _nssConfigName(nssConfigName) {}

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

    static const TorqHw *_instance;
};

} // namespace mlir::syna::torq
