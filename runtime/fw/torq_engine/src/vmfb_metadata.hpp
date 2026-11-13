// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORQ_VMFB_METADATA_HPP_
#define TORQ_VMFB_METADATA_HPP_

#include "torq/metadata.hpp"
#include "iree/hal/api.h"

namespace torq {

/// HAL shape and element type of a tensor described by VMFB reflection.
struct TensorSpec {
    std::vector<iree_hal_dim_t> dimensions;
    iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
};

/// Parse static input/output tensor descriptions from a VMFB ABI declaration.
/// @param declaration: value of the iree.abi.declaration reflection attribute
/// @param metadata: parsed tensor attributes; cleared on failure
/// @param input_specs: optional HAL input descriptions; cleared on failure
/// @return true if the declaration describes supported input/output tensors
bool parse_vmfb_declaration(iree_string_view_t declaration, NetworkMetadata &metadata,
                            std::vector<TensorSpec> *input_specs = nullptr);

}

#endif