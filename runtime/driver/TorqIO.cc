// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TorqIO.h"
#include "torq_hw/inc/TorqHw.h"

#if IREE_FILE_IO_ENABLE
#include "iree/base/internal/flags.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <system_error>
#include <unistd.h>
#include <dlfcn.h>
#endif  // IREE_FILE_IO_ENABLE

namespace synaptics {
namespace io {

#if IREE_FILE_IO_ENABLE

const char* FLAG_torq_hw_type      = DEF_HW_TYPE;
bool        FLAG_torq_step_by_step = false;
bool        FLAG_torq_clear_memory = false;
bool        FLAG_torq_explicit_dmabuf_sync = false;

#if IREE_FLAGS_ENABLE_CLI == 1
IREE_STATIC_INITIALIZER(iree_flag_register_torq_hw_type) {
  iree_flag_register(__FILE__, __LINE__, IREE_FLAG_TYPE_string,
                     (void*)&FLAG_torq_hw_type, NULL, NULL,
                     iree_make_cstring_view("torq_hw_type"),
                     iree_make_cstring_view("Hardware type [sim, aws_fpga, soc_fpga, astra_machina]"));
}
IREE_STATIC_INITIALIZER(iree_flag_register_torq_step_by_step) {
  iree_flag_register(__FILE__, __LINE__, IREE_FLAG_TYPE_bool,
                     (void*)&FLAG_torq_step_by_step, NULL, NULL,
                     iree_make_cstring_view("torq_step_by_step"),
                     iree_make_cstring_view("Enable step-by-step NPU debug mode"));
}
IREE_STATIC_INITIALIZER(iree_flag_register_torq_clear_memory) {
  iree_flag_register(__FILE__, __LINE__, IREE_FLAG_TYPE_bool,
                     (void*)&FLAG_torq_clear_memory, NULL, NULL,
                     iree_make_cstring_view("torq_clear_memory"),
                     iree_make_cstring_view("Zero XRAM and LRAM before execution"));
}
IREE_STATIC_INITIALIZER(iree_flag_register_torq_explicit_dmabuf_sync) {
  iree_flag_register(__FILE__, __LINE__, IREE_FLAG_TYPE_bool,
                     (void*)&FLAG_torq_explicit_dmabuf_sync, NULL, NULL,
                     iree_make_cstring_view("torq_explicit_dmabuf_sync"),
                     iree_make_cstring_view("Explicitly sync cached zero-copy DMA-BUF bindings in userspace instead of relying on kernel attach/detach"));
}
#endif  // IREE_FLAGS_ENABLE_CLI

#else  // !IREE_FILE_IO_ENABLE

const char* FLAG_torq_hw_type      = DEF_HW_TYPE;
bool        FLAG_torq_step_by_step = false;
bool        FLAG_torq_clear_memory = false;
bool        FLAG_torq_explicit_dmabuf_sync = false;

#endif  // IREE_FILE_IO_ENABLE

#if IREE_FILE_IO_ENABLE
struct TorqMutex::Impl { std::mutex m; };
TorqMutex::TorqMutex()  : impl_(new Impl{}) {}
TorqMutex::~TorqMutex() { delete impl_; }
#if defined(__clang__)
[[clang::no_thread_safety_analysis]]
#endif
void TorqMutex::lock()   { impl_->m.lock(); }
#if defined(__clang__)
[[clang::no_thread_safety_analysis]]
#endif
void TorqMutex::unlock() { impl_->m.unlock(); }
#else
struct TorqMutex::Impl {};
TorqMutex::TorqMutex()  : impl_(nullptr) {}
TorqMutex::~TorqMutex() {}
void TorqMutex::lock()   {}
void TorqMutex::unlock() {}
#endif

#if IREE_FILE_IO_ENABLE

bool writeFile(const std::string& path, const void* data, size_t size) {
  std::ofstream ofs(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!ofs.is_open()) return false;
  if (data && size > 0)
    ofs.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  ofs.close();
  return ofs.good();
}

bool writeTextFile(const std::string& path, const std::string& content) {
  std::ofstream ofs(path, std::ios::out | std::ios::trunc);
  if (!ofs.is_open()) return false;
  ofs << content;
  ofs.close();
  return ofs.good();
}

bool appendTextFile(const std::string& path, const std::string& content) {
  std::ofstream ofs(path, std::ios::out | std::ios::app);
  if (!ofs.is_open()) return false;
  ofs << content;
  ofs.close();
  return ofs.good();
}

iree_status_t createDirectory(const std::string& path) {
  std::error_code ec;
  std::filesystem::create_directory(path, ec);
  if (ec && ec != std::errc::file_exists) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to create directory '%s': %s",
                            path.c_str(), ec.message().c_str());
  }
  return iree_ok_status();
}

bool pathExists(const std::string& path) {
  std::error_code ec;
  return std::filesystem::exists(path, ec);
}

void removeDirectoryTree(const std::string& path) {
  std::error_code ec;
  std::filesystem::remove_all(path, ec);
}

iree_status_t loadLibraryFromMemory(const uint8_t* data, size_t size,
                                    void** out_handle) {
  *out_handle = nullptr;
  std::string tmp = "/tmp/torq_host_code.XXXXXX.so";
  int fd = mkstemps(&tmp[0], 3);
  if (fd == -1) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to create temporary file: %s",
                            std::strerror(errno));
  }
  ssize_t written = write(fd, data, size);
  close(fd);
  if (written < 0 || static_cast<size_t>(written) != size) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to write temporary library file");
  }
  void* handle = dlopen(tmp.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    return iree_make_status(IREE_STATUS_INTERNAL, "dlopen failed: %s", dlerror());
  }
  *out_handle = handle;
  return iree_ok_status();
}

void unloadLibrary(void* handle) {
  if (handle) dlclose(handle);
}

void* resolveSymbol(void* handle, const std::string& name) {
  if (!handle) return nullptr;
  return dlsym(handle, name.c_str());
}

#else  // !IREE_FILE_IO_ENABLE

bool writeFile(const std::string&, const void*, size_t)     { return true; }
bool writeTextFile(const std::string&, const std::string&)  { return true; }
bool appendTextFile(const std::string&, const std::string&) { return true; }
iree_status_t createDirectory(const std::string&)           { return iree_ok_status(); }
bool pathExists(const std::string&)                         { return false; }
void removeDirectoryTree(const std::string&)                {}

iree_status_t loadLibraryFromMemory(const uint8_t*, size_t, void** out_handle) {
  *out_handle = nullptr;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "dynamic library loading is not supported on bare-metal builds");
}
void unloadLibrary(void*)                                   {}
void* resolveSymbol(void*, const std::string&)              { return nullptr; }

#endif  // IREE_FILE_IO_ENABLE

}  // namespace io
}  // namespace synaptics
