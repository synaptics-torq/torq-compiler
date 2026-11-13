// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "TorqHw.h"

namespace synaptics {

std::unique_ptr<TorqHw> getAstraMachina(uint32_t xramStartAddr, size_t xramSize);

} // namespace synaptics
