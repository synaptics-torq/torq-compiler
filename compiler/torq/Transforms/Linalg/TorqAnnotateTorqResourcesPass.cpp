// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"

namespace mlir::syna::torq {
namespace {

struct TorqAnnotateTorqResourcesPass
    : public impl::TorqAnnotateTorqResourcesBase<TorqAnnotateTorqResourcesPass> {

    void runOnOperation() override {
        ModuleOp module = getOperation();

        constexpr int64_t kMaxAllocationSize = 1ULL << 32;
        constexpr int64_t kMinBufferOffsetAlignment = 4096;
        constexpr int64_t kMaxBufferRange = 1ULL << 32;
        constexpr int64_t kMinBufferRangeAlignment = 4096;
        constexpr int64_t kIndexBits = 32;
        constexpr bool kAliasMutableBindings = true;
        constexpr iree_compiler::IREE::Stream::MemoryModel kMemoryModel =
            iree_compiler::IREE::Stream::MemoryModel::Unified;

        auto resourceConfig = iree_compiler::IREE::Stream::ResourceConfigAttr::get(
            module.getContext(), kMaxAllocationSize, kMinBufferOffsetAlignment, kMaxBufferRange,
            kMinBufferRangeAlignment, kIndexBits, kAliasMutableBindings, kMemoryModel
        );

        module->setAttr("stream.resources", resourceConfig);
    }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createTorqAnnotateTorqResourcesPass() {
    return std::make_unique<TorqAnnotateTorqResourcesPass>();
}

} // namespace mlir::syna::torq
