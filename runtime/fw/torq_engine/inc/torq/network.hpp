// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

///
/// Torq Neural Network.
///

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "torq/tensor.hpp"

namespace torq {

class NetworkPrivate;

/// Load and execute a neural network on the NPU accelerator.
class Network {
public:
    /// Create an empty network with no model or tensors.
    Network();
    /// Dispose the current model, runtime resources, and tensor buffers.
    ~Network();
    /// Transfer the model, tensors, and runtime resources from another network.
    /// The source can be unloaded or loaded again after the move.
    /// @param other: network whose resources are transferred
    Network(Network &&other) noexcept;
    /// Dispose the current model and transfer another network's resources.
    /// @param other: network whose resources are transferred
    /// @return this network
    Network &operator=(Network &&other) noexcept;
    Network(const Network &) = delete;
    Network &operator=(const Network &) = delete;

    /// Load model.
    /// In case another model was previously loaded it is disposed before loading the one specified.
    /// Tensor names, shapes, and types are discovered from VMFB ABI reflection.
    /// Failure leaves the network empty. Existing tensor references are invalidated.
    ///
    /// @param model_data      VMFB model data, as from e.g. fread()
    ///                        The caller retains ownership of the model data and can delete them
    ///                        at the end of this method.
    /// @param model_size      model size in bytes
    /// @return                true if success
    bool load_model(const void *model_data, size_t model_size);
#if TORQ_OS_LINUX
    /// Load model.
    /// In case another model was previously loaded it is disposed before loading the one specified.
    /// Failure, including a file read error, leaves the network empty.
    /// The file is read synchronously and need not remain available after this call.
    /// Tensor attributes are discovered from the same VMFB ABI as memory loading.
    ///
    /// @param model_file      path to VMFB model file
    /// @return                true if success
    bool load_model(const std::string &model_file);
#endif
    /// Dispose the current model and its input/output tensors.
    void unload_model();

    /// Run inference.
    /// Input data to be processed are read from input tensor(s).
    /// Inference results are generated in output tensor(s).
    /// Inputs must already contain data; output storage is allocated automatically.
    /// The call is synchronous. Serialize access to the network and its buffers.
    /// Outputs must not be consumed after a failed prediction.
    ///
    /// @return true if success, false if inference failed or network not correctly initialized.
    bool predict();

    /// Collection of input tensors that can be accessed by index and iterated.
    Tensors inputs;
    /// Collection of output tensors that can be accessed by index and iterated.
    Tensors outputs;

private:
    std::unique_ptr<NetworkPrivate> d;
};

}  // namespace torq