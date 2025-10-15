// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string>

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"

#include "iree/compiler/Utils/OptionUtils.h"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

struct TorqTargetOptions {
    void bindOptions(OptionsBinder &binder) {}
};

std::shared_ptr<IREE::HAL::TargetDevice> createTarget(const TorqTargetOptions &options);

std::shared_ptr<IREE::HAL::TargetBackend> createBackend(const TorqTargetOptions &options);

} // namespace mlir::syna::torq
