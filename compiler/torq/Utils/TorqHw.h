#pragma once

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

    // Get the available cpu features of CSS
    std::string getCSSFeatures() const { return _cssFeatures; }

    // Get the NSS features
    std::string getNSSFeatures() const { return _nssFeatures; }

    TorqHw(
        std::string name, std::string description, size_t lramSize, size_t sliceCount,
        size_t availableMemoryForTiling, std::string cssFeatures, std::string nssFeatures
    )
        : _name(name), _description(description), _lramSize(lramSize), _sliceCount(sliceCount),
          _availableMemoryForTiling(availableMemoryForTiling), _cssFeatures(cssFeatures),
          _nssFeatures(nssFeatures) {}

    TorqHw() = default;

  private:
    std::string _name{};
    std::string _description{};
    std::string _version{};
    size_t _lramSize = 0;
    size_t _sliceCount = 0;
    size_t _availableMemoryForTiling = 0;
    std::string _cssFeatures{};
    std::string _nssFeatures{};

    static const TorqHw *_instance;
};

} // namespace mlir::syna::torq
