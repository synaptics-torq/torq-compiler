// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORQ_METADATA_HPP_
#define TORQ_METADATA_HPP_

#include <vector>

#include "torq/tensor.hpp"

namespace torq {

/// Input/output tensor attributes discovered while loading a model.
struct NetworkMetadata {
    bool valid = false;
    std::vector<TensorAttributes> inputs;
    std::vector<TensorAttributes> outputs;
};

}

#endif