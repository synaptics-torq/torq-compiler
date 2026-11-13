// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/network.hpp"

#include <memory>
#include <utility>
#include <vector>

#if TORQ_OS_LINUX
#include <fstream>
#include <string>
#endif

#include "predictor.hpp"

namespace torq {

/// Own the execution backend independently of the public tensor collections.
class NetworkPrivate {
public:
    std::unique_ptr<Predictor> predictor;
};

Network::Network() : d(std::make_unique<NetworkPrivate>()) {}
Network::~Network() { unload_model(); }
Network::Network(Network &&other) noexcept = default;

Network &Network::operator=(Network &&other) noexcept
{
    if (this != &other) {
        unload_model();
        inputs = std::move(other.inputs);
        outputs = std::move(other.outputs);
        d = std::move(other.d);
    }
    return *this;
}

void Network::unload_model()
{
    if (d) d->predictor.reset();
    inputs.tensors_.clear();
    outputs.tensors_.clear();
}

bool Network::load_model(const void *model_data, size_t model_size)
{
    unload_model();
    if (!model_data || !model_size) return false;
    if (!d) d = std::make_unique<NetworkPrivate>();

    NetworkMetadata metadata;
    d->predictor = make_torq_predictor();
    if (!d->predictor || !d->predictor->load_model(model_data, model_size,
            &metadata) || !metadata.valid || metadata.inputs.empty() || metadata.outputs.empty()) {
        unload_model();
        return false;
    }

    inputs.tensors_.reserve(metadata.inputs.size());
    outputs.tensors_.reserve(metadata.outputs.size());
    for (auto &attributes : metadata.inputs) {
        if (!attributes.size()) { unload_model(); return false; }
        inputs.tensors_.emplace_back(std::move(attributes));
    }
    for (auto &attributes : metadata.outputs) {
        if (!attributes.size()) { unload_model(); return false; }
        outputs.tensors_.emplace_back(std::move(attributes));
    }
    return true;
}

#if TORQ_OS_LINUX
bool Network::load_model(const std::string &model_file)
{
    unload_model();
    std::ifstream model_stream(model_file, std::ios::binary | std::ios::ate);
    if (!model_stream) return false;
    const std::streamsize size = model_stream.tellg();
    if (size <= 0) return false;
    Buffer model(static_cast<size_t>(size));
    if (!model.data()) return false;
    model_stream.seekg(0);
    if (!model_stream.read(static_cast<char *>(model.data()), size)) return false;
    return load_model(model.data(), model.size());
}
#endif

bool Network::predict()
{
    if (!d || !d->predictor) return false;
    for (const auto &tensor : inputs) if (!tensor.data()) return false;
    for (auto &tensor : outputs) if (!tensor.data()) return false;
    return d->predictor->predict(inputs, outputs);
}

}