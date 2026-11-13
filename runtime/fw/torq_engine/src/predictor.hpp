// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORQ_PREDICTOR_HPP_
#define TORQ_PREDICTOR_HPP_

#include <cstddef>
#include <memory>
#include <vector>

#include "torq/metadata.hpp"

namespace torq {

/// Execution backend owned by a Network.
/// Model parsing and tensor execution are platform-specific implementation details.
class Predictor {
public:
    virtual ~Predictor() = default;
    /// Copy and load a VMFB model, then discover its input/output attributes.
    /// @param model: caller-owned VMFB data, valid for the duration of this call
    /// @param size: model size in bytes
    /// @param metadata: tensor description populated on successful loading
    /// @return true if success
    virtual bool load_model(const void *model, size_t size, NetworkMetadata *metadata) = 0;
    /// Run synchronous inference using the current input/output tensor buffers.
    /// @param inputs: prepared input tensors
    /// @param outputs: output tensors receiving the inference results
    /// @return true if success; output contents are unspecified on failure
    virtual bool predict(Tensors &inputs, Tensors &outputs) = 0;
};

/// Create the build-selected platform backend.
/// @return predictor owned by the caller
std::unique_ptr<Predictor> make_torq_predictor();

}

#endif