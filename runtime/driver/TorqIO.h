// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "iree/base/api.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace synaptics {
namespace io {

bool writeFile(const std::string& path, const void* data, size_t size);
bool writeTextFile(const std::string& path, const std::string& content);
bool appendTextFile(const std::string& path, const std::string& content);
iree_status_t createDirectory(const std::string& path);
bool pathExists(const std::string& path);
void removeDirectoryTree(const std::string& path);

iree_status_t loadLibraryFromMemory(const uint8_t* data, size_t size,
                                    void** out_handle);
void unloadLibrary(void* handle);
void* resolveSymbol(void* handle, const std::string& name);

class TorqMutex {
public:
  TorqMutex();
  ~TorqMutex();
  void lock();
  void unlock();

  TorqMutex(const TorqMutex&) = delete;
  TorqMutex& operator=(const TorqMutex&) = delete;

  class ScopedLock {
  public:
    explicit ScopedLock(TorqMutex& m) : mutex_(m) { mutex_.lock(); }
    ~ScopedLock() { mutex_.unlock(); }
    ScopedLock(const ScopedLock&) = delete;
    ScopedLock& operator=(const ScopedLock&) = delete;
  private:
    TorqMutex& mutex_;
  };

private:
  struct Impl;
  Impl* impl_;
};

extern const char* FLAG_torq_hw_type;
extern bool        FLAG_torq_step_by_step;
extern bool        FLAG_torq_clear_memory;
extern bool        FLAG_torq_explicit_dmabuf_sync;

}  // namespace io
}  // namespace synaptics
