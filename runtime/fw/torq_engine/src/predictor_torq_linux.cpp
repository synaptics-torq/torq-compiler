// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "predictor.hpp"
#include "vmfb_metadata.hpp"
#include "allocator_linux.hpp"
#include "driver/torq_allocator.h"

#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/runtime/api.h"
#include "logging_linux.hpp"

#ifndef TORQ_LINUX_DEVICE
#define TORQ_LINUX_DEVICE "torq"
#endif

namespace torq {
namespace {

constexpr const char *entry_function = "module.main";

/// Report and consume an IREE status; return true only for success.
bool succeeded(iree_status_t status)
{
    if (iree_status_is_ok(status)) return true;
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return false;
}

/// Convert a tensor storage type to its HAL representation, or NONE if unsupported.
iree_hal_element_type_t element_type(DataType type)
{
    switch (type) {
    case DataType::int8: return IREE_HAL_ELEMENT_TYPE_SINT_8;
    case DataType::uint8: return IREE_HAL_ELEMENT_TYPE_UINT_8;
    case DataType::int16: return IREE_HAL_ELEMENT_TYPE_SINT_16;
    case DataType::uint16: return IREE_HAL_ELEMENT_TYPE_UINT_16;
    case DataType::int32: return IREE_HAL_ELEMENT_TYPE_SINT_32;
    case DataType::uint32: return IREE_HAL_ELEMENT_TYPE_UINT_32;
    case DataType::int64: return IREE_HAL_ELEMENT_TYPE_SINT_64;
    case DataType::uint64: return IREE_HAL_ELEMENT_TYPE_UINT_64;
    case DataType::float16: return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
    case DataType::float32: return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
    case DataType::float64: return IREE_HAL_ELEMENT_TYPE_FLOAT_64;
    default: return IREE_HAL_ELEMENT_TYPE_NONE;
    }
}

/// Retain a tensor buffer and its HAL view until the binding is replaced.
struct BufferBinding {
    std::shared_ptr<Buffer> buffer;
    const void *address = nullptr;
    size_t size = 0;
    uint64_t generation = 0;
    bool needs_input_copy = false;
    iree_hal_buffer_view_t *view = nullptr;

    BufferBinding() = default;
    BufferBinding(const BufferBinding &) = delete;
    BufferBinding &operator=(const BufferBinding &) = delete;
    BufferBinding(BufferBinding &&other) noexcept
        : buffer(std::move(other.buffer)), address(other.address), size(other.size),
                    generation(other.generation), needs_input_copy(other.needs_input_copy),
          view(std::exchange(other.view, nullptr)) {}
    BufferBinding &operator=(BufferBinding &&other) noexcept {
        if (this != &other) {
            if (view) iree_hal_buffer_view_release(view);
            buffer = std::move(other.buffer);
            address = other.address;
            size = other.size;
            generation = other.generation;
            needs_input_copy = other.needs_input_copy;
            view = std::exchange(other.view, nullptr);
        }
        return *this;
    }
    ~BufferBinding() { if (view) iree_hal_buffer_view_release(view); }
};

}

/// Execute VMFB models through an IREE runtime session on Linux.
/// Torq DMA-BUF allocations are wrapped directly; ordinary host buffers use
/// HAL allocation and refreshed input copies. Output copies are elided only
/// when the runtime result aliases the attached destination.
class PredictorTORQ final : public Predictor {
public:
    ~PredictorTORQ() override { cleanup_runtime(); }

    /// @copydoc Predictor::load_model
    bool load_model(const void *model, size_t size, NetworkMetadata *metadata) override {
        if (!model || !size || !metadata || instance_) {
            LOGE << "Invalid model parameters or model already loaded";
            return false;
        }
        if (!model_.assign(model, size)) return false;

        LOGI << "Loading TORQ VMFB model, size: " << size << " bytes";
        if (!runtime_initialized_ && !succeeded(initialize_runtime())) {
            LOGE << "Failed to initialize TORQ runtime";
            return false;
        }
        if (!succeeded(iree_runtime_session_append_bytecode_module_from_memory(session_,
                iree_make_const_byte_span(model_.data(), model_.size()), iree_allocator_null()))) {
            LOGE << "Failed to create IREE bytecode module";
            return false;
        }
        call_initialized_ = succeeded(iree_runtime_call_initialize_by_name(session_,
            iree_make_cstring_view(entry_function), &main_call_));
        if (!call_initialized_) {
            LOGE << "Unable to resolve entry function from VM session";
            return false;
        }
        const auto declaration = iree_vm_function_lookup_attr_by_name(&main_call_.function,
            iree_make_cstring_view("iree.abi.declaration"));
        if (!parse_vmfb_declaration(declaration, *metadata)) {
            LOGE << "Unable to parse VMFB tensor metadata";
            return false;
        }
        bindings_.resize(metadata->inputs.size());
        output_bindings_.resize(metadata->outputs.size());
        output_count_ = metadata->outputs.size();
        model_loaded_ = true;
        LOGI << "TORQ VMFB meta, input size: " << metadata->inputs.size() << " entries";
        LOGI << "TORQ VMFB model loaded successfully";
        return true;
    }

    /// @copydoc Predictor::predict
    bool predict(Tensors &inputs, Tensors &outputs) override {
        if (!model_loaded_ || !session_ || inputs.size() != bindings_.size() ||
            outputs.size() != output_count_) {
            LOGE << "Model not loaded or tensor counts do not match";
            return false;
        }
        if (!set_buffers(inputs, bindings_) || !set_buffers(outputs, output_bindings_)) return false;

        LOGI << "Running TORQ inference";
        iree_runtime_call_reset(&main_call_);
        if (!succeeded(prepare_invocation_inputs())) {
            LOGE << "Failed to prepare invocation inputs";
            return false;
        }
        const bool inputs_released = sync_cpu_access(bindings_, false);
        const bool outputs_released = sync_cpu_access(output_bindings_, false);
        if (!inputs_released || !outputs_released) {
            sync_cpu_access(bindings_, true);
            sync_cpu_access(output_bindings_, true);
            return false;
        }
        const bool invoked = succeeded(iree_runtime_call_invoke(&main_call_, 0));
        const bool inputs_acquired = sync_cpu_access(bindings_, true);
        const bool outputs_acquired = sync_cpu_access(output_bindings_, true);
        if (!invoked) {
            LOGE << "TORQ inference failed";
            return false;
        }
        if (!inputs_acquired || !outputs_acquired) return false;
        if (!update_output_buffer_pointers(outputs)) {
            LOGE << "Failed to update output buffer";
            return false;
        }
        LOGI << "TORQ inference completed successfully";
        return true;
    }

private:
    /// Begin or end CPU access for every DMA-BUF, attempting all transitions.
    /// @param bindings: tensor buffer bindings to synchronize
    /// @param begin: true to reacquire CPU access, false to release it
    /// @return true if every transition succeeds
    bool sync_cpu_access(const std::vector<BufferBinding> &bindings, bool begin) {
        bool success = true;
        for (const auto &binding : bindings) {
            const bool synced = begin ? linux_buffer_begin_cpu_access(binding.address)
                                      : linux_buffer_end_cpu_access(binding.address);
            success = synced && success;
        }
        return success;
    }

    /// Create the runtime instance, build-selected device, and session.
    /// @return IREE status; partially created resources remain owned for cleanup
    iree_status_t initialize_runtime() {
        LOGI << "Initializing TORQ runtime";
        iree_runtime_instance_options_t instance_options;
        iree_runtime_instance_options_initialize(&instance_options);
        iree_runtime_instance_options_use_all_available_drivers(&instance_options);
        IREE_RETURN_IF_ERROR(iree_runtime_instance_create(
            &instance_options, iree_allocator_system(), &instance_));

        iree_hal_device_t *device = nullptr;
        IREE_RETURN_IF_ERROR(iree_runtime_instance_try_create_default_device(
            instance_, iree_make_cstring_view(TORQ_LINUX_DEVICE), &device));
        if (!device) return iree_make_status(IREE_STATUS_INTERNAL, "Failed to get TORQ device");

        iree_runtime_session_options_t session_options;
        iree_runtime_session_options_initialize(&session_options);
        iree_status_t status = iree_runtime_session_create_with_device(instance_, &session_options,
            device, iree_runtime_instance_host_allocator(instance_), &session_);
        iree_hal_device_release(device);
        if (iree_status_is_ok(status)) {
            runtime_initialized_ = true;
            LOGI << "TORQ runtime initialized successfully";
        }
        return status;
    }

    /// Release an initialized call, HAL bindings, session, and runtime instance.
    void cleanup_runtime() {
        if (call_initialized_) iree_runtime_call_deinitialize(&main_call_);
        bindings_.clear();
        output_bindings_.clear();
        if (session_) iree_runtime_session_release(session_);
        if (instance_) iree_runtime_instance_release(instance_);
        session_ = nullptr;
        instance_ = nullptr;
        call_initialized_ = false;
        runtime_initialized_ = false;
        model_loaded_ = false;
        LOGI << "TORQ runtime cleaned up";
    }

    /// Reuse current bindings or replace views whose storage identity changed.
    /// @param tensors: current tensor buffers, ordered by ABI position
    /// @param bindings: cached HAL bindings for the same tensor positions
    /// @return true if every tensor has a valid binding
    bool set_buffers(Tensors &tensors, std::vector<BufferBinding> &bindings) {
        for (size_t index = 0; index < tensors.size(); ++index) {
            auto &tensor = tensors[index];
            auto &binding = bindings[index];
            void *address = tensor.data();
            if (!address) return false;
            if (!binding.view || binding.buffer != tensor.buffer_ || binding.address != address ||
                binding.size != tensor.size() || binding.generation != linux_buffer_generation(address)) {
                BufferBinding replacement;
                if (!attach_buffer(tensor, replacement)) return false;
                binding = std::move(replacement);
                LOGI << "Buffer set for tensor " << index << ": " << tensor.name();
            }
        }
        return true;
    }

    /// Wrap a DMA-BUF or allocate and initialize fallback HAL storage.
    /// @param source: tensor storage retained by the resulting binding
    /// @param needs_input_copy: true if inputs must be refreshed before each invocation
    /// @param out_buffer: HAL buffer whose ownership passes to the caller on success
    /// @return IREE status
    iree_status_t create_hal_buffer(Buffer *source, bool &needs_input_copy, iree_hal_buffer_t **out_buffer) {
        if (!source || !out_buffer || !source->data() || !source->size())
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "Invalid buffer data");
        auto *device = iree_runtime_session_device(session_);
        if (!device) return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "No device available from session");
        auto *allocator = iree_hal_device_allocator(device);
        if (!allocator) return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "No allocator available from device");
        if (auto *buffer = linux_buffer_retain_hal_buffer(source->data())) {
            *out_buffer = buffer;
            needs_input_copy = false;
            return iree_ok_status();
        }
        const int fd = linux_buffer_fd(source->data());
        if (fd >= 0) {
            iree_hal_external_buffer_t external_buffer = {};
            external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD;
            external_buffer.size = source->size();
            external_buffer.handle.opaque_fd.fd = fd;
            iree_hal_buffer_params_t parameters = {};
            parameters.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
            parameters.access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
            parameters.usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                               IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
            parameters.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
            iree_status_t status = iree_hal_allocator_import_buffer(
                allocator, parameters, &external_buffer, iree_hal_buffer_release_callback_null(), out_buffer);
            if (iree_status_is_ok(status)) {
                needs_input_copy = false;
                LOGI << "[ZeroCopy] Imported dmabuf fd=" << fd << " size=" << source->size()
                     << " as torq HAL buffer " << *out_buffer;
                return status;
            }
            return status;
        }
        iree_hal_buffer_params_t parameters = {};
        parameters.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
        parameters.access = IREE_HAL_MEMORY_ACCESS_ALL;
        parameters.usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                           IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
        parameters.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
        IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
            allocator, parameters, source->size(), out_buffer));
        iree_status_t status = iree_hal_buffer_map_write(*out_buffer, 0, source->data(), source->size());
        if (!iree_status_is_ok(status)) {
            iree_hal_buffer_release(*out_buffer);
            *out_buffer = nullptr;
            return status;
        }
        needs_input_copy = true;
        LOGI << "Allocated HAL buffer " << *out_buffer << " for host buffer " << source
             << " size=" << source->size();
        return iree_ok_status();
    }

    /// Create a shaped HAL view and retain its source tensor buffer.
    /// @param tensor: tensor supplying storage, shape, and element type
    /// @param binding: destination owning the HAL view and source buffer reference
    /// @return true if success
    bool attach_buffer(Tensor &tensor, BufferBinding &binding) {
        iree_hal_buffer_t *buffer = nullptr;
        if (!succeeded(create_hal_buffer(tensor.buffer(), binding.needs_input_copy, &buffer))) {
            LOGE << "Failed to create HAL buffer for tensor " << tensor.name();
            return false;
        }
        const std::vector<iree_hal_dim_t> dimensions(tensor.shape().begin(), tensor.shape().end());
        const bool created = succeeded(iree_hal_buffer_view_create(buffer, dimensions.size(), dimensions.data(),
            element_type(tensor.data_type()), IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
            iree_runtime_session_host_allocator(session_), &binding.view));
        iree_hal_buffer_release(buffer);
        if (!created) return false;
        LOGI << "Created buffer view " << binding.view << " for tensor " << tensor.name();
        binding.buffer = tensor.buffer_;
        binding.address = tensor.data();
        binding.size = tensor.size();
        binding.generation = linux_buffer_generation(binding.address);
        return true;
    }

    /// Refresh copied inputs and push HAL views in ABI argument order.
    /// @return IREE status
    iree_status_t prepare_invocation_inputs() {
        for (size_t index = 0; index < bindings_.size(); ++index) {
            if (!bindings_[index].view)
                return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "Input buffer %zu not attached", index);
            const auto &binding = bindings_[index];
            if (binding.needs_input_copy) {
                IREE_RETURN_IF_ERROR(iree_hal_buffer_map_write(
                    iree_hal_buffer_view_buffer(binding.view), 0, binding.buffer->data(), binding.size));
            }
            IREE_RETURN_IF_ERROR(iree_runtime_call_inputs_push_back_buffer_view(
                &main_call_, bindings_[index].view));
        }
        return iree_ok_status();
    }

    /// Validate returned sizes and copy non-aliased results into tensor storage.
    /// Scoped mappings are released before returning; no mapped pointer escapes.
    /// @param outputs: tensor destinations, ordered by ABI result position
    /// @return true if every result was copied or aliases its destination
    bool update_output_buffer_pointers(Tensors &outputs) {
        if (iree_vm_list_size(iree_runtime_call_outputs(&main_call_)) != outputs.size()) return false;
        for (size_t index = 0; index < outputs.size(); ++index) {
            iree_hal_buffer_view_t *view = nullptr;
            if (!succeeded(iree_runtime_call_outputs_pop_front_buffer_view(&main_call_, &view))) return false;
            if (!view) return false;
            if (iree_hal_buffer_view_byte_length(view) != outputs[index].size()) {
                iree_hal_buffer_view_release(view);
                return false;
            }
            auto *runtime_buffer = iree_hal_buffer_view_buffer(view);
            auto *target_buffer = iree_hal_buffer_view_buffer(output_bindings_[index].view);
            if (iree_hal_buffer_allocated_buffer(runtime_buffer) == iree_hal_buffer_allocated_buffer(target_buffer) &&
                iree_hal_buffer_byte_offset(runtime_buffer) == iree_hal_buffer_byte_offset(target_buffer)) {
                iree_hal_buffer_view_release(view);
                LOGI << "[ZeroCopy] Output " << index << " aliases its attached buffer";
                continue;
            }
            iree_hal_buffer_mapping_t mapping;
            if (!succeeded(iree_hal_buffer_map_range(runtime_buffer,
                    IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0,
                    outputs[index].size(), &mapping))) {
                iree_hal_buffer_view_release(view);
                return false;
            }
            const auto copied_size = mapping.contents.data_length;
            const bool copied = output_bindings_[index].buffer->assign(
                static_cast<const void *>(mapping.contents.data), copied_size);
            const bool unmapped = succeeded(iree_hal_buffer_unmap_range(&mapping));
            iree_hal_buffer_view_release(view);
            if (!copied || !unmapped) return false;
            LOGI << "Copied TORQ output data to Torq buffer " << index << " ("
                 << copied_size << " bytes)";
        }
        return true;
    }

    Buffer model_;
    iree_runtime_instance_t *instance_ = nullptr;
    iree_runtime_session_t *session_ = nullptr;
    iree_runtime_call_t main_call_ = {};
    bool call_initialized_ = false;
    bool runtime_initialized_ = false;
    bool model_loaded_ = false;
    size_t output_count_ = 0;
    std::vector<BufferBinding> bindings_;
    std::vector<BufferBinding> output_bindings_;
};

std::unique_ptr<Predictor> make_torq_predictor()
{
    return std::make_unique<PredictorTORQ>();
}

}