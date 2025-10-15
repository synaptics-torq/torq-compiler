#pragma once

#include <cstdint>
#include <string>

namespace mlir::syna::torq {

class TorqHw {
  public:
    enum Type {
        SL2610, // Synaptics SL2610 family hardware
        SL2610_1SLICE, // Synaptics SL2610 family hardware with 1 slice
        CUSTOM  // Custom hardware configuration
    };

    virtual ~TorqHw();
    TorqHw(const TorqHw &) = delete;

    Type getType() const { return _type; }

    static const TorqHw &get();

    // Get the hardware information as a string
    virtual std::string getString() const = 0;

    // Get LRAM size in bytes
    virtual size_t getLramSize() const = 0;

    // Get Number of Slices
    virtual size_t getSliceCount() const = 0;

    // Get Available Memory for Tiling in bytes
    virtual size_t getAvailableMemoryForTiling() const = 0;

  protected:
    TorqHw(Type type) : _type(type) {}
    Type _type;
    static const TorqHw *_instance;
};

} // namespace mlir::syna::torq
