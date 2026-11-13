// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "vmfb_metadata.hpp"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>

namespace torq {
namespace {

constexpr size_t max_tensor_rank = 8;
constexpr const char tensor_prefix[] = "tensor<";

/// Map a HAL element type to tensor storage, or invalid if unsupported.
DataType data_type(iree_hal_element_type_t type)
{
    switch (type) {
    case IREE_HAL_ELEMENT_TYPE_INT_8: case IREE_HAL_ELEMENT_TYPE_SINT_8: return DataType::int8;
    case IREE_HAL_ELEMENT_TYPE_UINT_8: return DataType::uint8;
    case IREE_HAL_ELEMENT_TYPE_INT_16: case IREE_HAL_ELEMENT_TYPE_SINT_16: return DataType::int16;
    case IREE_HAL_ELEMENT_TYPE_UINT_16: return DataType::uint16;
    case IREE_HAL_ELEMENT_TYPE_INT_32: case IREE_HAL_ELEMENT_TYPE_SINT_32: return DataType::int32;
    case IREE_HAL_ELEMENT_TYPE_UINT_32: return DataType::uint32;
    case IREE_HAL_ELEMENT_TYPE_INT_64: case IREE_HAL_ELEMENT_TYPE_SINT_64: return DataType::int64;
    case IREE_HAL_ELEMENT_TYPE_UINT_64: return DataType::uint64;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16: return DataType::float16;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32: return DataType::float32;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64: return DataType::float64;
    default: return DataType::invalid;
    }
}

/// Extract ordered static tensors, rejecting unsupported types or dimensions.
bool parse_tensors(const std::string &declaration, const char *prefix,
                   std::vector<TensorAttributes> &attributes, std::vector<TensorSpec> *specifications)
{
    size_t position = 0;
    while ((position = declaration.find(tensor_prefix, position)) != std::string::npos) {
        position += sizeof(tensor_prefix) - 1;
        const size_t end = declaration.find('>', position);
        if (end == std::string::npos) return false;
        const std::string body = declaration.substr(position, end - position);
        TensorSpec tensor;
        TensorAttributes attribute;
        attribute.name = std::string(prefix) + std::to_string(attributes.size());
        size_t start = 0;
        size_t separator;
        while ((separator = body.find('x', start)) != std::string::npos) {
            const std::string dimension = body.substr(start, separator - start);
            if (dimension.empty() || dimension.find_first_not_of("0123456789") != std::string::npos ||
                tensor.dimensions.size() == max_tensor_rank) return false;
            errno = 0;
            const unsigned long value = std::strtoul(dimension.c_str(), nullptr, 10);
            if (errno == ERANGE || !value || value > UINT32_MAX ||
                value > std::numeric_limits<iree_hal_dim_t>::max()) return false;
            tensor.dimensions.push_back(static_cast<iree_hal_dim_t>(value));
            attribute.shape.push_back(static_cast<uint32_t>(value));
            start = separator + 1;
        }
        const std::string type = body.substr(start);
        iree_status_t status = iree_hal_parse_element_type(iree_make_cstring_view(type.c_str()),
            &tensor.element_type);
        if (!iree_status_is_ok(status)) {
            iree_status_ignore(status);
            return false;
        }
        attribute.dtype = data_type(tensor.element_type);
        if (!attribute.size() || attribute.size() > UINT32_MAX) return false;
        attributes.push_back(std::move(attribute));
        if (specifications) specifications->push_back(std::move(tensor));
        position = end + 1;
    }
    return !attributes.empty();
}

}

bool parse_vmfb_declaration(iree_string_view_t declaration, NetworkMetadata &metadata,
                            std::vector<TensorSpec> *input_specs)
{
    metadata = {};
    if (input_specs) input_specs->clear();
    if (!declaration.data || !declaration.size) return false;
    const std::string signature(declaration.data, declaration.size);
    const size_t arrow = signature.find("->");
    if (arrow == std::string::npos) return false;
    NetworkMetadata parsed;
    std::vector<TensorSpec> specifications;
    if (!parse_tensors(signature.substr(0, arrow), "input_", parsed.inputs,
            input_specs ? &specifications : nullptr) ||
        !parse_tensors(signature.substr(arrow + 2), "output_", parsed.outputs, nullptr)) return false;
    parsed.valid = true;
    metadata = std::move(parsed);
    if (input_specs) *input_specs = std::move(specifications);
    return true;
}

}