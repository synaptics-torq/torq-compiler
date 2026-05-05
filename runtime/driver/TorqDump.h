// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "native_executable.h"
#include "TorqHw.h"
#include "TestVectorWriter.h"
#include "iree/base/api.h"

#include <memory>
#include <string>

namespace synaptics {
namespace dump {

extern const char* FLAG_torq_dump_io_data_dir;
extern const char* FLAG_torq_dump_buffers_dir;
extern const char* FLAG_torq_dump_test_vectors_dir;
extern bool        FLAG_torq_dump_bus_logs;

// Dump a single input or output binding to the io-data directory.
void dumpBinding(const std::string& executableName, uint32_t bindingId,
                 const void* data, size_t size, bool isInput);

// Dump all live intermediate buffers after the given action to the buffers
// directory. Reads XRAM/LRAM/DTCM/ITCM via torq and saves numpy arrays.
void dumpBuffers(TorqHw* torq, int32_t actionId,
                 iree_hal_torq_native_executable_t* executable);

// Create the per-executable subdirectory inside FLAG_torq_dump_io_data_dir.
iree_status_t setupIODumpDirs(const std::string& executableName);

// Create the per-job subdirectory inside FLAG_torq_dump_buffers_dir.
iree_status_t createJobDirectory(const std::string& executableName, int jobId);

// Return a TestVectorWriter for executableName, or nullptr if
// FLAG_torq_dump_test_vectors_dir is not set.
std::unique_ptr<TestVectorWriter> createTestVectorWriter(
    const std::string& executableName);

}  // namespace dump
}  // namespace synaptics
