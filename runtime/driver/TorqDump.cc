// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TorqDump.h"
#include "TorqIO.h"

#if IREE_FILE_IO_ENABLE
#include "iree/base/internal/flags.h"
#include "iree/base/internal/flatcc/parsing.h"
#include "torq_executable_def_reader.h"

#undef ns
#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(iree_hal_torq, x)

#include <algorithm>
#include <cassert>
#include <fstream>
#include <vector>

#include "iree/hal/api.h"
#include "iree/tooling/numpy_io.h"
#include "iree/io/stdio_stream.h"
#endif  // IREE_FILE_IO_ENABLE

namespace synaptics {
namespace dump {

#if IREE_FILE_IO_ENABLE

const char* FLAG_torq_dump_io_data_dir      = "";
const char* FLAG_torq_dump_buffers_dir      = "";
const char* FLAG_torq_dump_test_vectors_dir = "";
bool        FLAG_torq_dump_bus_logs         = false;

#if IREE_FLAGS_ENABLE_CLI == 1
IREE_STATIC_INITIALIZER(iree_flag_register_torq_dump_io_data_dir) {
  iree_flag_register(__FILE__, __LINE__, IREE_FLAG_TYPE_string,
                     (void*)&FLAG_torq_dump_io_data_dir, NULL, NULL,
                     iree_make_cstring_view("torq_dump_io_data_dir"),
                     iree_make_cstring_view("Directory for binary input/output binding dumps"));
}
IREE_STATIC_INITIALIZER(iree_flag_register_torq_dump_buffers_dir) {
  iree_flag_register(__FILE__, __LINE__, IREE_FLAG_TYPE_string,
                     (void*)&FLAG_torq_dump_buffers_dir, NULL, NULL,
                     iree_make_cstring_view("torq_dump_buffers_dir"),
                     iree_make_cstring_view("Directory for per-action intermediate buffer dumps"));
}
IREE_STATIC_INITIALIZER(iree_flag_register_torq_dump_test_vectors_dir) {
  iree_flag_register(__FILE__, __LINE__, IREE_FLAG_TYPE_string,
                     (void*)&FLAG_torq_dump_test_vectors_dir, NULL, NULL,
                     iree_make_cstring_view("torq_dump_test_vectors_dir"),
                     iree_make_cstring_view("Directory for torq_rt-compatible test vector files"));
}
IREE_STATIC_INITIALIZER(iree_flag_register_torq_dump_bus_logs) {
  iree_flag_register(__FILE__, __LINE__, IREE_FLAG_TYPE_bool,
                     (void*)&FLAG_torq_dump_bus_logs, NULL, NULL,
                     iree_make_cstring_view("torq_dump_bus_logs"),
                     iree_make_cstring_view("Collect NPU internal bus logs during test-vector generation"));
}
#endif  // IREE_FLAGS_ENABLE_CLI

#else  // !IREE_FILE_IO_ENABLE

const char* FLAG_torq_dump_io_data_dir      = "";
const char* FLAG_torq_dump_buffers_dir      = "";
const char* FLAG_torq_dump_test_vectors_dir = "";
bool        FLAG_torq_dump_bus_logs         = false;

#endif  // IREE_FILE_IO_ENABLE

#if IREE_FILE_IO_ENABLE

static std::string get_io_dump_data_dir(const std::string& executable_name) {
  if (!FLAG_torq_dump_io_data_dir[0]) return "";
  return std::string(FLAG_torq_dump_io_data_dir) + "/" + executable_name + "/";
}

static std::string get_buffer_dump_dir(const std::string& executable_name,
                                       int job_id) {
  if (!FLAG_torq_dump_buffers_dir[0]) return "";
  return std::string(FLAG_torq_dump_buffers_dir) + "/" + executable_name +
         "/action" + std::to_string(job_id) + "/";
}

static iree_hal_element_type_t to_iree_hal_element_type(
    iree_hal_torq_ElementType_enum_t et) {
  switch (et) {
    case iree_hal_torq_ElementType_I8:   return IREE_HAL_ELEMENT_TYPE_INT_8;
    case iree_hal_torq_ElementType_I16:  return IREE_HAL_ELEMENT_TYPE_INT_16;
    case iree_hal_torq_ElementType_I32:  return IREE_HAL_ELEMENT_TYPE_INT_32;
    case iree_hal_torq_ElementType_F32:  return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
    case iree_hal_torq_ElementType_BF16: return IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
    default:                             return IREE_HAL_ELEMENT_TYPE_NONE;
  }
}

static std::vector<iree_device_size_t> to_size_vec(flatbuffers_uint32_vec_t v) {
  std::vector<iree_device_size_t> out;
  for (size_t i = 0; i < flatbuffers_uint32_vec_len(v); ++i)
    out.push_back(flatbuffers_uint32_vec_at(v, i));
  return out;
}

static iree_status_t copy_to_hal_buffer(
    const std::vector<uint8_t>& src, iree_hal_allocator_t* allocator,
    const std::vector<iree_device_size_t>& shape,
    const std::vector<iree_device_size_t>& strides,
    iree_hal_element_type_t element_type, iree_hal_buffer_t** dst) {

  size_t element_size_bits = iree_hal_element_bit_count(element_type);
  assert(element_size_bits % 8 == 0 && "Element size must be a multiple of 8");
  size_t element_size_bytes = element_size_bits / 8;

  size_t total_elements = 1;
  for (auto d : shape) total_elements *= d;

  iree_hal_buffer_t* hal_buffer = nullptr;
  iree_hal_buffer_params_t params = {0};
  params.type   = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage  = IREE_HAL_BUFFER_USAGE_DEFAULT;

  iree_status_t status = iree_hal_allocator_allocate_buffer(
      allocator, params, total_elements * element_size_bytes, &hal_buffer);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return status;
  }

  for (size_t i = 0; i < total_elements; ++i) {
    size_t src_offset = 0, remaining = i;
    for (size_t j = shape.size(); j > 0; --j) {
      size_t index = remaining % shape[j - 1];
      remaining /= shape[j - 1];
      src_offset += index * strides[j - 1] * element_size_bytes;
    }
    status = iree_hal_buffer_map_write(hal_buffer, i * element_size_bytes,
                                       src.data() + src_offset,
                                       element_size_bytes);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      return status;
    }
  }

  *dst = hal_buffer;
  return iree_ok_status();
}

static iree_status_t dump_to_numpy(
    const std::string& filePath,
    iree_hal_torq_native_executable_t* executable,
    iree_hal_element_type_t element_type,
    const std::vector<uint8_t>& data,
    const std::vector<iree_device_size_t>& shape,
    const std::vector<iree_device_size_t>& strides) {

  iree_hal_allocator_t* allocator = iree_hal_device_allocator(executable->device);
  iree_hal_buffer_t* hal_buffer = nullptr;

  iree_status_t status = copy_to_hal_buffer(
      data, allocator, shape, strides, element_type, &hal_buffer);
  if (!iree_status_is_ok(status)) return status;

  iree_io_stream_t* stream = nullptr;
  iree_io_stdio_stream_mode_t mode =
      IREE_IO_STDIO_STREAM_MODE_WRITE | IREE_IO_STDIO_STREAM_MODE_DISCARD;

  status = iree_io_stdio_stream_open(mode, iree_make_cstring_view(filePath.c_str()),
                                     executable->allocator, &stream);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(hal_buffer);
    iree_status_fprint(stderr, status);
    return status;
  }

  iree_hal_buffer_view_t* buffer_view = nullptr;
  status = iree_hal_buffer_view_create(
      hal_buffer, shape.size(), shape.data(),
      element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      executable->allocator, &buffer_view);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_io_stream_release(stream);
    iree_hal_buffer_release(hal_buffer);
    return status;
  }

  status = iree_numpy_npy_save_ndarray(stream, IREE_NUMPY_NPY_SAVE_OPTION_DEFAULT,
                                       buffer_view, executable->allocator);
  if (!iree_status_is_ok(status)) iree_status_fprint(stderr, status);

  iree_hal_buffer_view_release(buffer_view);
  iree_hal_buffer_release(hal_buffer);
  iree_io_stream_release(stream);
  return status;
}

static iree_status_t create_buffers_index(
    iree_hal_torq_native_executable_t* executable) {

  auto executable_def = ns(ExecutableDef_as_root(executable->program));
  auto buffers = ns(ExecutableDef_buffers_debug_info(executable_def));
  if (!buffers) return iree_ok_status();

  std::string name = ns(ExecutableDef_executable_name(executable_def));
  std::string path = std::string(FLAG_torq_dump_buffers_dir) + "/" + name + "/buffers.csv";

  std::ofstream fp(path, std::ios::out);
  if (!fp.is_open()) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to open file %s", path.c_str());
  }

  auto actions = ns(ExecutableDef_actions(executable_def));
  fp << "id;type;shape;strides;address;size;"
        "allocation_action;last_use_action;deallocation_action;"
        "allocation_location;last_use_location;deallocation_location\n";

  for (int i = 0; i < ns(BufferDebugInfo_vec_len(buffers)); ++i) {
    auto buf = ns(BufferDebugInfo_vec_at(buffers, i));

    fp << ns(BufferDebugInfo_id(buf)) << ";";
    fp << ns(BufferType_name(ns(BufferDebugInfo_buffer_type(buf)))) << ";";

    auto shape = ns(BufferDebugInfo_shape_get(buf));
    for (int j = 0; j < (int)flatbuffers_uint32_vec_len(shape); ++j)
      fp << flatbuffers_uint32_vec_at(shape, j) << "x";

    std::string et_str = ns(ElementType_name(ns(BufferDebugInfo_element_type(buf))));
    std::transform(et_str.begin(), et_str.end(), et_str.begin(), ::tolower);
    fp << et_str << ";";

    auto strides = ns(BufferDebugInfo_strides_get(buf));
    int rank = flatbuffers_uint32_vec_len(strides);
    for (int j = 0; j < rank; ++j) {
      fp << flatbuffers_uint32_vec_at(strides, j);
      if (j < rank - 1) fp << ",";
    }

    auto alloc_id   = ns(BufferDebugInfo_allocation_action(buf));
    auto dealloc_id = ns(BufferDebugInfo_deallocation_action(buf));
    if (dealloc_id < 0) dealloc_id = ns(HostAction_vec_len(actions)) - 1;
    auto last_use   = ns(BufferDebugInfo_last_use_action(buf));

    fp << ";" << ns(BufferDebugInfo_address(buf))
       << ";" << ns(BufferDebugInfo_size(buf))
       << ";" << alloc_id
       << ";" << last_use
       << ";" << dealloc_id
       << ";\n";
  }

  fp.close();
  return iree_ok_status();
}

void dumpBinding(const std::string& executableName, uint32_t bindingId,
                 const void* data, size_t size, bool isInput) {
  if (!FLAG_torq_dump_io_data_dir[0]) return;

  const std::string io_str = isInput ? ".in" : ".out";
  const std::string dir    = get_io_dump_data_dir(executableName);
  const std::string file   = "npu" + io_str + ".host_binding." +
                             std::to_string(bindingId) + ".torq_binding." +
                             std::to_string(bindingId) + ".bin";
  io::writeFile(dir + file, data, size);
}

void dumpBuffers(TorqHw* torq, int32_t actionId,
                 iree_hal_torq_native_executable_t* executable) {
  if (!FLAG_torq_dump_buffers_dir[0]) return;

  auto executable_def  = ns(ExecutableDef_as_root(executable->program));
  auto executable_name = std::string(ns(ExecutableDef_executable_name(executable_def)));
  auto buffers         = ns(ExecutableDef_buffers_debug_info(executable_def));
  if (!buffers) return;

  const std::string dump_dir = get_buffer_dump_dir(executable_name, actionId);
  if (!io::pathExists(dump_dir)) io::createDirectory(dump_dir);

  auto actions = ns(ExecutableDef_actions(executable_def));
  auto action  = ns(HostAction_vec_at(actions, actionId));

  if (ns(HostAction_params_type(action)) == ns(HostActionParams_StartNSSParams)) {
    auto params = ns(StartNSSParams_table_t)(ns(HostAction_params(action)));
    torq->wait(true,
               ns(StartNSSParams_starts_slice1(params)),
               ns(StartNSSParams_starts_slice2(params)),
               ns(StartNSSParams_starts_dma_in(params)),
               ns(StartNSSParams_starts_dma_out(params)));
  }

  if (actionId == 0) create_buffers_index(executable);

  size_t buffer_count = ns(BufferDebugInfo_vec_len(buffers));
  for (size_t i = 0; i < buffer_count; ++i) {
    auto buf = ns(BufferDebugInfo_vec_at(buffers, i));

    if (actionId < ns(BufferDebugInfo_allocation_action(buf)) ||
        actionId > ns(BufferDebugInfo_last_use_action(buf)))
      continue;

    auto elementType = to_iree_hal_element_type(
        ns(BufferDebugInfo_element_type_get(buf)));
    if (elementType == IREE_HAL_ELEMENT_TYPE_BFLOAT_16)
      elementType = IREE_HAL_ELEMENT_TYPE_UINT_16;
    if (elementType == IREE_HAL_ELEMENT_TYPE_NONE) continue;

    uint32_t address = ns(BufferDebugInfo_address_get(buf));
    auto shape   = to_size_vec(ns(BufferDebugInfo_shape(buf)));
    auto strides = to_size_vec(ns(BufferDebugInfo_strides(buf)));
    auto elemBytes = iree_hal_element_dense_byte_count(elementType);

    size_t dataLen = 1;
    for (int d = 0; d < (int)shape.size(); ++d)
      dataLen += (shape[d] - 1) * strides[d];
    dataLen *= elemBytes;

    std::vector<uint8_t> stridedData(dataLen);
    switch (ns(BufferDebugInfo_buffer_type_get(buf))) {
      case ns(BufferType_XRAM): torq->readXram(address, dataLen, stridedData.data()); break;
      case ns(BufferType_LRAM): torq->readLram(address, dataLen, stridedData.data()); break;
      case ns(BufferType_DTCM): torq->readDtcm(address, dataLen, stridedData.data()); break;
      case ns(BufferType_ITCM): torq->readItcm(address, dataLen, stridedData.data()); break;
      default: continue;
    }

    const std::string file_path = dump_dir + "buffer_" +
                                  std::to_string(ns(BufferDebugInfo_id(buf))) + ".npy";
    iree_status_t status = dump_to_numpy(file_path, executable, elementType,
                                         stridedData, shape, strides);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      assert(false && "dump_to_numpy failed");
    }
  }
}

iree_status_t setupIODumpDirs(const std::string& executableName) {
  if (!FLAG_torq_dump_io_data_dir[0]) return iree_ok_status();
  iree_status_t status = io::createDirectory(FLAG_torq_dump_io_data_dir);
  if (!iree_status_is_ok(status)) return status;
  return io::createDirectory(get_io_dump_data_dir(executableName));
}

iree_status_t createJobDirectory(const std::string& executableName, int jobId) {
  (void)jobId;
  if (!FLAG_torq_dump_buffers_dir[0]) return iree_ok_status();
  iree_status_t status = io::createDirectory(FLAG_torq_dump_buffers_dir);
  if (!iree_status_is_ok(status)) return status;
  return io::createDirectory(std::string(FLAG_torq_dump_buffers_dir) + "/" + executableName);
}

std::unique_ptr<TestVectorWriter> createTestVectorWriter(
    const std::string& executableName) {
  if (!FLAG_torq_dump_test_vectors_dir[0]) return nullptr;
  return std::make_unique<TestVectorWriter>(executableName,
                                           FLAG_torq_dump_test_vectors_dir);
}

#else  // !IREE_FILE_IO_ENABLE

void dumpBinding(const std::string&, uint32_t, const void*, size_t, bool) {}
void dumpBuffers(TorqHw*, int32_t, iree_hal_torq_native_executable_t*)    {}
iree_status_t setupIODumpDirs(const std::string&)    { return iree_ok_status(); }
iree_status_t createJobDirectory(const std::string&, int) { return iree_ok_status(); }
std::unique_ptr<TestVectorWriter> createTestVectorWriter(const std::string&) { return nullptr; }

#endif  // IREE_FILE_IO_ENABLE

}  // namespace dump
}  // namespace synaptics
