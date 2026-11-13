// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "predictor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cache.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include "allocator_private.hpp"
#include "vmfb_metadata.hpp"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"
#include "runtime/driver/registration/driver_module.h"

LOG_MODULE_REGISTER(torq_predictor, LOG_LEVEL_INF);

/// Notify platform power management when synchronous NPU work starts or ends.
/// @param active: true while the NPU is in use
extern "C" void __attribute__((weak)) torq_npu_pm_notify(bool active)
{
    (void)active;
}

namespace torq {
namespace {

constexpr size_t vmfb_signature_size = 4;
constexpr size_t max_abi_log_length = 512;
constexpr const char *entry_function = "module.main";

K_MUTEX_DEFINE(npu_mutex);

bool succeeded(iree_status_t status, const char *operation = "IREE operation")
{
    if (iree_status_is_ok(status)) return true;
    LOG_ERR("%s failed: %s", operation, iree_status_code_string(iree_status_code(status)));
    iree_status_ignore(status);
    return false;
}

bool register_driver()
{
    static bool valid = false;
    if (k_mutex_lock(&npu_mutex, K_FOREVER) != 0) return false;
    if (!valid) {
        valid = succeeded(iree_hal_torq_driver_module_register(iree_hal_driver_registry_default()),
            "TORQ driver registration");
    }
    return k_mutex_unlock(&npu_mutex) == 0 && valid;
}

/// Own the HAL view of an imported input buffer until its storage changes.
struct InputBinding {
    const void *data = nullptr;
    size_t size = 0;
    iree_hal_buffer_view_t *view = nullptr;

    InputBinding() = default;
    InputBinding(const InputBinding &) = delete;
    InputBinding &operator=(const InputBinding &) = delete;
    InputBinding(InputBinding &&other) noexcept
        : data(other.data), size(other.size), view(std::exchange(other.view, nullptr)) {}
    InputBinding &operator=(InputBinding &&other) noexcept {
        if (this != &other) {
            if (view) iree_hal_buffer_view_release(view);
            data = other.data;
            size = other.size;
            view = std::exchange(other.view, nullptr);
        }
        return *this;
    }

    ~InputBinding() { if (view) iree_hal_buffer_view_release(view); }
};

}

/// Execute VMFB models on Zephyr with serialized NPU access and cache maintenance.
class PredictorTORQ : public Predictor {
public:
    ~PredictorTORQ() override {
        bindings_.clear();
        if (inputs_) iree_vm_list_release(inputs_);
        if (outputs_) iree_vm_list_release(outputs_);
        if (context_) iree_vm_context_release(context_);
        if (device_) iree_hal_device_release(device_);
        if (instance_) iree_vm_instance_release(instance_);
    }

    /// @copydoc Predictor::load_model
    bool load_model(const void *model, size_t size, NetworkMetadata *metadata) override {
        if (!model || !metadata || size < vmfb_signature_size || size > UINT32_MAX || instance_) {
            LOG_ERR("Invalid model data, size, or predictor state");
            return false;
        }
        auto &inputs = metadata->inputs;
        auto &outputs = metadata->outputs;
        if (!model_.assign(model, size)) {
            LOG_ERR("Unable to allocate %zu-byte model copy in cached APSRAM", size);
            return false;
        }
        if (!register_driver()) return false;
        if (cache_clean_addr(model_.data(), static_cast<uint32_t>(model_.size())) != 0) {
            LOG_ERR("Model cache clean failed");
            return false;
        }
        host_allocator_ = iree_allocator_system();
        if (!succeeded(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT, host_allocator_, &instance_),
                "VM instance creation") ||
            !succeeded(iree_hal_module_register_all_types(instance_), "HAL type registration")) return false;

        iree_hal_driver_t *driver = nullptr;
        if (!succeeded(iree_hal_driver_registry_try_create(iree_hal_driver_registry_default(),
            iree_make_cstring_view("torq"), host_allocator_, &driver), "TORQ driver creation")) return false;
        bool created = succeeded(iree_hal_driver_create_default_device(driver, host_allocator_, &device_),
            "TORQ device creation");
        iree_hal_driver_release(driver);
        if (!created) return false;

        iree_vm_module_t *hal_module = nullptr;
        if (!succeeded(iree_hal_module_create(instance_, iree_hal_module_device_policy_default(),
            1, &device_, IREE_HAL_MODULE_FLAG_SYNCHRONOUS, iree_hal_module_debug_sink_null(),
            host_allocator_, &hal_module), "HAL module creation")) return false;
        iree_vm_module_t *bytecode_module = nullptr;
        created = succeeded(iree_vm_bytecode_module_create(instance_, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
            iree_make_const_byte_span(model_.data(), model_.size()), iree_allocator_null(),
            host_allocator_, &bytecode_module),
            "VMFB bytecode loading");
        if (created) {
            iree_vm_module_t *modules[] = {hal_module, bytecode_module};
            created = succeeded(iree_vm_context_create_with_modules(instance_, IREE_VM_CONTEXT_FLAG_NONE,
            IREE_ARRAYSIZE(modules), modules, host_allocator_, &context_), "VM context creation");
        }
        if (bytecode_module) iree_vm_module_release(bytecode_module);
        iree_vm_module_release(hal_module);
        if (!created || !succeeded(iree_vm_context_resolve_function(context_,
            iree_make_cstring_view(entry_function), &main_function_), "module.main resolution")) return false;

        auto declaration = iree_vm_function_lookup_attr_by_name(&main_function_,
            iree_make_cstring_view("iree.abi.declaration"));
        if (!declaration.data || !declaration.size) {
            LOG_ERR("Model is missing iree.abi.declaration metadata");
            return false;
        }
        if (!parse_vmfb_declaration(declaration, *metadata, &input_specs_)) {
            LOG_ERR("Unsupported tensor ABI: %.*s",
                static_cast<int>(std::min(declaration.size, max_abi_log_length)), declaration.data);
            return false;
        }
        output_count_ = outputs.size();
        bindings_.resize(inputs.size());
        metadata->valid = succeeded(iree_vm_list_create(iree_vm_make_ref_type_def(iree_hal_buffer_view_type()),
                    inputs.size(), host_allocator_, &inputs_), "Input list creation") &&
            succeeded(iree_vm_list_create(iree_vm_make_ref_type_def(iree_hal_buffer_view_type()),
                    outputs.size(), host_allocator_, &outputs_), "Output list creation");
        return metadata->valid;
    }

    bool predict(Tensors &inputs, Tensors &outputs) override {
        if (!context_ || inputs.size() != bindings_.size() || outputs.size() != output_count_ ||
            !succeeded(iree_vm_list_resize(inputs_, 0)) ||
            !succeeded(iree_vm_list_resize(outputs_, 0))) return false;
        for (size_t index = 0; index < inputs.size(); ++index) {
            auto &tensor = inputs[index];
            void *address = tensor.data();
            if (!address || reinterpret_cast<uintptr_t>(address) % buffer_alignment) return false;
            auto &binding = bindings_[index];
            if (!binding.view || binding.data != address || binding.size != tensor.size()) {
                InputBinding replacement;
                if (!attach(address, tensor.size(), input_specs_[index], replacement)) return false;
                binding = std::move(replacement);
            }
            if (cache_clean_addr(address, static_cast<uint32_t>(tensor.size())) != 0) return false;
            iree_vm_ref_t reference = iree_hal_buffer_view_retain_ref(binding.view);
            bool appended = succeeded(iree_vm_list_push_ref_retain(inputs_, &reference));
            iree_vm_ref_release(&reference);
            if (!appended) return false;
        }

        if (k_mutex_lock(&npu_mutex, K_FOREVER) != 0) return false;
        torq_npu_pm_notify(true);
        bool invoked = succeeded(iree_vm_invoke(context_, main_function_, IREE_VM_INVOCATION_FLAG_NONE,
            nullptr, inputs_, outputs_, host_allocator_));
        torq_npu_pm_notify(false);
        bool copied = invoked && copy_outputs(outputs);
        bool unlocked = k_mutex_unlock(&npu_mutex) == 0;
        return copied && unlocked;
    }

private:
    bool copy_outputs(Tensors &outputs) {
        if (iree_vm_list_size(outputs_) != outputs.size()) return false;
        for (size_t index = 0; index < outputs.size(); ++index) {
            auto *view = iree_vm_list_get_buffer_view_assign(outputs_, index);
            if (!view || iree_hal_buffer_view_byte_length(view) != outputs[index].size()) return false;
            iree_hal_buffer_mapping_t mapping;
            if (!succeeded(iree_hal_buffer_map_range(iree_hal_buffer_view_buffer(view),
                    IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
                    0, iree_hal_buffer_view_byte_length(view), &mapping))) return false;
            bool copied = cache_invalidate_addr(CACHE_TYPE_DATA, mapping.contents.data,
                static_cast<uint32_t>(mapping.contents.data_length)) == 0 &&
                outputs[index].assign(mapping.contents.data, mapping.contents.data_length);
            bool unmapped = succeeded(iree_hal_buffer_unmap_range(&mapping));
            if (!copied || !unmapped) return false;
        }
        return true;
    }

    bool attach(void *address, size_t size, const TensorSpec &specification, InputBinding &binding) {
        iree_hal_buffer_params_t parameters = {};
        parameters.type = static_cast<iree_hal_memory_type_t>(
            IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
        parameters.access = IREE_HAL_MEMORY_ACCESS_ALL;
        parameters.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
        parameters.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
        parameters.min_alignment = buffer_alignment;
        iree_hal_buffer_params_t compatible_parameters = {};
        iree_device_size_t allocation_size = 0;
        auto *allocator = iree_hal_device_allocator(device_);
        auto compatibility = iree_hal_allocator_query_buffer_compatibility(allocator,
            parameters, size, &compatible_parameters, &allocation_size);
        if (!iree_all_bits_set(compatibility, IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE)) return false;

        iree_hal_external_buffer_t external = {};
        external.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
        external.size = size;
        external.handle.host_allocation.ptr = address;
        iree_hal_buffer_t *buffer = nullptr;
        if (!succeeded(iree_hal_allocator_import_buffer(allocator, compatible_parameters, &external,
                iree_hal_buffer_release_callback_null(), &buffer))) return false;
        bool created = succeeded(iree_hal_buffer_view_create(buffer, specification.dimensions.size(),
            specification.dimensions.data(), specification.element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
            host_allocator_, &binding.view));
        iree_hal_buffer_release(buffer);
        if (!created) return false;
        binding.data = address;
        binding.size = size;
        return true;
    }

    Buffer model_;
    iree_allocator_t host_allocator_ = iree_allocator_null();
    iree_vm_instance_t *instance_ = nullptr;
    iree_hal_device_t *device_ = nullptr;
    iree_vm_context_t *context_ = nullptr;
    iree_vm_function_t main_function_ = {};
    iree_vm_list_t *inputs_ = nullptr;
    iree_vm_list_t *outputs_ = nullptr;
    std::vector<TensorSpec> input_specs_;
    size_t output_count_ = 0;
    std::vector<InputBinding> bindings_;
};

std::unique_ptr<Predictor> make_torq_predictor()
{
    return std::make_unique<PredictorTORQ>();
}

}