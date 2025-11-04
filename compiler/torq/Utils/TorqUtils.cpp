// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TorqUtils.h"

namespace mlir::syna {

void printBlock(Block &block) {
    // Print the block intrinsics properties (basically: argument list)
    llvm::outs() << "Block with " << block.getNumArguments() << " arguments, "
                 << block.getNumSuccessors()
                 << " successors, and "
                 // Note, this `.size()` is traversing a linked-list and is O(n).
                 << block.getOperations().size() << " operations\n";

    // A block main role is to hold a list of Operations: let's recurse into
    // printing each operation.
    for (Operation &op : block.getOperations())
        printOperation(&op);
}

void printRegion(Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
    llvm::outs() << "Region with " << region.getBlocks().size() << " blocks:\n";
    for (Block &block : region.getBlocks())
        printBlock(block);
}

void printOperation(Operation *op) {
    // Print the operation itself and some of its properties
    llvm::outs() << "visiting op: '" << op->getName() << "' with " << op->getNumOperands()
                 << " operands and " << op->getNumResults() << " results\n";
    // Print the operation attributes
    if (!op->getAttrs().empty()) {
        llvm::outs() << op->getAttrs().size() << " attributes:\n";
        for (NamedAttribute attr : op->getAttrs())
            llvm::outs() << " - '" << attr.getName() << "' : '" << attr.getValue() << "'\n";
    }

    // Recurse into each of the regions attached to the operation.
    llvm::outs() << " " << op->getNumRegions() << " nested regions:\n";
    for (Region &region : op->getRegions())
        printRegion(region);
}

namespace torq {

std::vector<uint32_t> prepareWeightDims(
    std::vector<uint32_t> weight_shapes, size_t alu_groups_check, size_t alu_groups,
    uint32_t wram_seg_width
) {
    std::vector<uint32_t> limits{wram_seg_width, 1, UINT_MAX};
    if (alu_groups_check < alu_groups) {
        limits[0] = HwInfo::wram_width;
        limits[1] = HwInfo::wram_seg_width / HwInfo::wram_width;
    }

    size_t mul = 1;
    size_t limits_idx = 0;
    std::vector<uint32_t> weight_dims;

    while (!weight_shapes.empty()) {
        auto ws = weight_shapes.back();
        weight_shapes.pop_back();
        auto accu_mul = mul * ws;
        if (accu_mul <= limits[limits_idx]) {
            mul = accu_mul;
            continue;
        }

        auto max_range = limits[limits_idx] / mul;
        size_t multiple = 1;
        for (size_t i = max_range; i > 1; --i) {
            if (ws % i == 0 && ws % alu_groups == 0) {
                multiple = i;
                break;
            }
        }

        if (multiple > 1) {
            mul *= multiple;
            weight_shapes.push_back(ws / multiple);
        }
        else {
            weight_dims.push_back(mul);
            weight_shapes.push_back(ws);
            mul = 1;
            ++limits_idx;
        }
    }

    if (mul != 1) {
        weight_dims.push_back(mul);
    }
    while (weight_dims.size() < 2) {
        weight_dims.push_back(1);
    }

    return weight_dims;
}

} // namespace torq
} // namespace mlir::syna
