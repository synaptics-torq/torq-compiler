#include "TorqExecutable.h"
#include "TorqUtils.h"
#include "TorqEventLog.h"
#include "TorqIO.h"
#include "TorqDump.h"
#include "torq_allocator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <functional>
#include <map>
#include <utility>

using synaptics::io::FLAG_torq_hw_type;
using synaptics::io::FLAG_torq_step_by_step;
using synaptics::io::FLAG_torq_clear_memory;
using synaptics::dump::FLAG_torq_dump_bus_logs;
using synaptics::io::FLAG_torq_explicit_dmabuf_sync;


namespace synaptics {

static synaptics::TorqExecutable* get_torq_executable(iree_hal_torq_native_executable_t* executable) {  
  return (synaptics::TorqExecutable*)executable->torq_executable;
}

static EventType toEventType(ns(HostActionParams_union_type_t) paramsType) {

  switch (paramsType) {
    case ns(HostActionParams_HostCopyParams):
      return EventType::HOST_COPY;
    case ns(HostActionParams_StartNSSParams):
      return EventType::NSS_START;
    case ns(HostActionParams_WaitNSSParams):
      return EventType::NSS_WAIT;
    case ns(HostActionParams_AllocParams):
      return EventType::ALLOC;
    case ns(HostActionParams_DeallocParams):
      return EventType::DEALLOC;
    case ns(HostActionParams_StartHostParams):
      return EventType::HOST_START;
    case ns(HostActionParams_WaitHostParams):
      return EventType::HOST_WAIT;
    default:
      assert(false && "unknown host action type");
  }
}




static iree_status_t compute_xram_footprint(ns(ExecutableDef_table_t) executable_def, uint32_t* xram_base, uint32_t* xram_size) {

  int xram_start = UINT32_MAX;
  int xram_end = 0;

  LOGD << "XRAM memory map:";

  // iterate over all the segments and compute the xram start and end address of each to find the xram base and size
  iree_hal_torq_Segment_vec_t segments = ns(ExecutableDef_code_get(executable_def));
  size_t segments_count = ns(Segment_vec_len(segments));
  
  for (size_t i = 0; i < segments_count; i++) {
    iree_hal_torq_Segment_table_t segment = ns(Segment_vec_at(segments, i));
    
    uint32_t xram_address = ns(Segment_xram_address(segment));
    auto xram_segment_size = ns(Segment_size(segment));
    auto initialized = ns(Segment_data_is_present(segment));

    if (xram_address < xram_start) {
      xram_start = xram_address;
    }

    if (xram_address + xram_segment_size > xram_end) {
      xram_end = xram_address + xram_segment_size;      
      // FIXME: we add 640 because the compiler always reads 640 bytes for each segment
      // while the segment size is stricly only the code. We should fix the compiler
      // to read only what is necessary.
      xram_end += 640;
    }

    LOGD << "Segment " << i << ": xram_address=" << xram_address 
         << ", xram_segment_size=" << xram_segment_size 
         << ", xram_start=" << xram_start 
         << ", xram_end=" << xram_end
         << ", initialized=" << initialized;
  
  }

  // iterate over all the bindings and compute the xram start and end address of each to find the xram base and size
  iree_hal_torq_Binding_vec_t bindings = ns(ExecutableDef_bindings_get(executable_def));
  size_t bindings_count = ns(Binding_vec_len(bindings));
  for (size_t i = 0; i < bindings_count; i++) {
    iree_hal_torq_Binding_table_t binding = ns(Binding_vec_at(bindings, i));
    
    uint32_t xram_address = ns(Binding_address(binding));

    uint32_t xram_size_binding = ns(Binding_size(binding));

    if (xram_address < xram_start) {
      xram_start = xram_address;
    }

    if (xram_address + xram_size_binding > xram_end) {
      xram_end = xram_address + xram_size_binding;
    }

    LOGD << "Binding " << i << ": xram_address=" << xram_address 
         << ", xram_size_binding=" << xram_size_binding 
         << ", xram_start=" << xram_start 
         << ", xram_end=" << xram_end;
  }

  // iterate over all actions and find the xram start and end address of each alloc operation
  iree_hal_torq_HostAction_vec_t actions = ns(ExecutableDef_actions_get(executable_def));
  size_t actions_count = ns(HostAction_vec_len(actions));

  for (size_t i = 0; i < actions_count; i++) {
    iree_hal_torq_HostAction_table_t action = ns(HostAction_vec_at(actions, i));

    if (ns(HostAction_params_type(action)) == ns(HostActionParams_AllocParams)) {

      auto params = (ns(AllocParams_table_t)) ns(HostAction_params_get(action));

      if (ns(AllocParams_buffer_type(params)) != ns(BufferType_XRAM)) {
        continue;
      }

      uint32_t xram_address = ns(AllocParams_address(params));
      uint32_t xram_size_alloc = ns(AllocParams_size(params));

      if (xram_address < xram_start) {
        xram_start = xram_address;
      }

      if (xram_address + xram_size_alloc > xram_end) {
        xram_end = xram_address + xram_size_alloc;
      }

      LOGD << "Alloc Action " << i << ": xram_address=" << xram_address 
           << ", xram_size_alloc=" << xram_size_alloc 
           << ", xram_start=" << xram_start 
           << ", xram_end=" << xram_end;

    }
  }

  if (xram_start == UINT32_MAX || xram_end == 0) {
    *xram_base = 0;
    *xram_size = 0;
  }
  else {
    *xram_base = xram_start;
    *xram_size = xram_end - xram_start;
  }

  return iree_ok_status();

}


typedef struct fake_npu_top_t
{
    void*    *torq_nss_regs;
    uint8_t*            lram;
    uint8_t*            xram;
} fake_npu_top_t;

static bool isZeroCopyDebugModeEnabled() {
  return dump::FLAG_torq_dump_io_data_dir[0] || dump::FLAG_torq_dump_test_vectors_dir[0];
}

// adding helper in case strategy complexity exceeds a single flag
static bool shouldExplicitlySyncZeroCopyDmaBufs() {
    return FLAG_torq_explicit_dmabuf_sync;
}

static const torq_hw_device_buffer_t* getZeroCopyDeviceBuffer(
    const iree_hal_torq_dispatch_state_v0_t* torqState,
    iree_hal_torq_Binding_table_t binding, size_t* effectiveOffset) {
    if (isZeroCopyDebugModeEnabled()) {
      return nullptr;
    }
    auto bindingId = ns(Binding_id(binding));
    if (bindingId >= torqState->dispatch_state.binding_count) {
      return nullptr;
    }
    if (!(torqState->binding_flags_storage[bindingId] &
          IREE_HAL_TORQ_BINDING_FLAG_ZERO_COPY_ELIGIBLE)) {
      return nullptr;
    }
    auto* rawBuffer = torqState->binding_buffer_storage[bindingId];
    if (!rawBuffer) {
      return nullptr;
    }
    auto* deviceBuffer =
        iree_hal_torq_buffer_device_buffer(iree_hal_buffer_allocated_buffer(rawBuffer));
    if (!deviceBuffer) {
      return nullptr;
    }
    *effectiveOffset = torqState->binding_offset_storage[bindingId] + ns(Binding_offset(binding));
    return deviceBuffer;
}

static iree_hal_buffer_t* getBindingBuffer(
    const iree_hal_torq_dispatch_state_v0_t* torqState,
    iree_hal_torq_Binding_table_t binding) {
  auto bindingId = ns(Binding_id(binding));
  if (bindingId >= torqState->dispatch_state.binding_count) {
    return nullptr;
  }
  return torqState->binding_buffer_storage[bindingId];
}

static iree_device_size_t getBindingEffectiveOffset(
    const iree_hal_torq_dispatch_state_v0_t* torqState,
    iree_hal_torq_Binding_table_t binding) {
  auto bindingId = ns(Binding_id(binding));
  return torqState->binding_offset_storage[bindingId] + ns(Binding_offset(binding));
}

static uint64_t saturatedRangeEnd(uint64_t address, uint64_t size) {
  if (size > std::numeric_limits<uint64_t>::max() - address) {
    return std::numeric_limits<uint64_t>::max();
  }
  return address + size;
}

static bool xramRangesOverlap(
    uint64_t lhsAddress, uint64_t lhsSize, uint64_t rhsAddress, uint64_t rhsSize) {
  if (lhsSize == 0 || rhsSize == 0) {
    return false;
  }
  return lhsAddress < saturatedRangeEnd(rhsAddress, rhsSize) &&
         rhsAddress < saturatedRangeEnd(lhsAddress, lhsSize);
}

static bool isImmutableInputBinding(
    const iree_hal_torq_dispatch_state_v0_t* torqState,
    iree_hal_torq_Binding_table_t binding) {
  if (!ns(Binding_is_read_only(binding)) || ns(Binding_is_write_only(binding))) {
    return false;
  }
  iree_hal_buffer_t* buffer = getBindingBuffer(torqState, binding);
  if (!buffer) {
    return false;
  }
  return iree_all_bits_set(
      iree_hal_buffer_allowed_usage(buffer), IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE);
}

static bool torqDeviceBufferNeedsWholeBufferSync(
    const torq_hw_device_buffer_t* deviceBuffer) {
  return deviceBuffer && deviceBuffer->handle != TORQ_HW_INVALID_FD &&
         deviceBuffer->hostCached;
}

static bool hasSyncedDeviceBufferFd(
    const std::vector<int>& syncedFds,
    const torq_hw_device_buffer_t* deviceBuffer) {
  // Key by the live fd owned by the dispatch buffer. Distinct fds may alias the
  // same dmabuf; in that case we only miss a coalescing opportunity.
  return torqDeviceBufferNeedsWholeBufferSync(deviceBuffer) &&
         std::find(syncedFds.begin(), syncedFds.end(), deviceBuffer->handle) !=
             syncedFds.end();
}

static void markSyncedDeviceBufferFd(
    std::vector<int>& syncedFds, const torq_hw_device_buffer_t* deviceBuffer) {
  if (torqDeviceBufferNeedsWholeBufferSync(deviceBuffer) &&
      !hasSyncedDeviceBufferFd(syncedFds, deviceBuffer)) {
    syncedFds.push_back(deviceBuffer->handle);
  }
}

static iree_status_t cleanupZeroCopyBindings(
    TorqHw* torq, ns(ExecutableDef_table_t) executableDef,
    iree_hal_executable_dispatch_state_v0_t* state,
    std::vector<uint8_t>& zeroCopyAttached) {
  auto* torqState = iree_hal_torq_dispatch_state_cast(state);
  if (!iree_hal_torq_ExecutableDef_bindings_is_present(executableDef)) {
    return iree_ok_status();
  }

  auto bindings = ns(ExecutableDef_bindings_get(executableDef));
  auto bindings_count = iree_hal_torq_Binding_vec_len(bindings);
  std::vector<int> invalidatedFds;
  for (size_t i = 0; i < bindings_count; ++i) {
    auto binding = iree_hal_torq_Binding_vec_at(bindings, i);
    auto bindingId = ns(Binding_id(binding));
    if (i >= zeroCopyAttached.size() || !zeroCopyAttached[i]) {
      continue;
    }

    size_t effectiveOffset = 0;
    const torq_hw_device_buffer_t* deviceBuffer =
          getZeroCopyDeviceBuffer(torqState, binding, &effectiveOffset);
    if (!deviceBuffer) {
      zeroCopyAttached[i] = 0;
      continue;
    }

    auto bindingAddress = ns(Binding_address(binding));
    auto bindingSize = ns(Binding_size(binding));
    if (!torq->detachBinding(*deviceBuffer, bindingAddress, effectiveOffset, bindingSize)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL, "failed to detach zero-copy binding %d", bindingId);
    }
    zeroCopyAttached[i] = 0;
    if (shouldExplicitlySyncZeroCopyDmaBufs() && !ns(Binding_is_read_only(binding))) {
      iree_status_t invalidateStatus = iree_ok_status();
      const bool syncAlreadyDone =
          hasSyncedDeviceBufferFd(invalidatedFds, deviceBuffer);
      if (!syncAlreadyDone) {
        invalidateStatus = torq_hw_device_buffer_invalidate_range(
            const_cast<torq_hw_device_buffer_t*>(deviceBuffer), effectiveOffset, bindingSize);
        if (iree_status_is_ok(invalidateStatus)) {
          markSyncedDeviceBufferFd(invalidatedFds, deviceBuffer);
        }
      }
      IREE_RETURN_IF_ERROR(invalidateStatus);
    }
  }

  return iree_ok_status();
}

void TorqExecutable::clearPersistentInputCopies() {
  for (auto& copy : persistentInputCopies_) {
    if (copy.buffer) {
      iree_hal_buffer_release(copy.buffer);
      copy.buffer = nullptr;
    }
    copy.valid = false;
  }
  persistentInputCopies_.clear();
  persistentInputCopiesByXramAddress_.clear();
}

void TorqExecutable::refreshPersistentInputCopyXramIndexPrefix(size_t firstEntry) {
  if (firstEntry >= persistentInputCopiesByXramAddress_.size()) {
    return;
  }

  uint64_t prefixMaxEnd =
      firstEntry == 0
          ? 0
          : persistentInputCopiesByXramAddress_[firstEntry - 1].prefixMaxEnd;
  for (size_t i = firstEntry; i < persistentInputCopiesByXramAddress_.size(); ++i) {
    prefixMaxEnd =
        std::max(prefixMaxEnd, persistentInputCopiesByXramAddress_[i].xramEnd);
    persistentInputCopiesByXramAddress_[i].prefixMaxEnd = prefixMaxEnd;
  }
}

void TorqExecutable::addPersistentInputCopyToXramIndex(size_t copyIndex) {
  if (copyIndex >= persistentInputCopies_.size()) {
    return;
  }

  const auto& copy = persistentInputCopies_[copyIndex];
  if (!copy.valid) {
    return;
  }

  PersistentInputCopyXramIndexEntry entry;
  entry.xramAddress = copy.xramAddress;
  entry.xramEnd = saturatedRangeEnd(copy.xramAddress, copy.bindingSize);
  entry.copyIndex = copyIndex;

  auto entryIt = std::lower_bound(
      persistentInputCopiesByXramAddress_.begin(),
      persistentInputCopiesByXramAddress_.end(),
      std::make_pair(entry.xramAddress, entry.copyIndex),
      [](const PersistentInputCopyXramIndexEntry& lhs,
         const std::pair<uint32_t, size_t>& rhs) {
        return lhs.xramAddress < rhs.first ||
               (lhs.xramAddress == rhs.first && lhs.copyIndex < rhs.second);
      });
  size_t entryPosition =
      static_cast<size_t>(entryIt - persistentInputCopiesByXramAddress_.begin());
  persistentInputCopiesByXramAddress_.insert(entryIt, entry);
  refreshPersistentInputCopyXramIndexPrefix(entryPosition);
}

void TorqExecutable::removePersistentInputCopyFromXramIndex(size_t copyIndex) {
  if (copyIndex >= persistentInputCopies_.size()) {
    return;
  }

  const auto& copy = persistentInputCopies_[copyIndex];
  if (!copy.valid) {
    return;
  }

  auto entryIt = std::lower_bound(
      persistentInputCopiesByXramAddress_.begin(),
      persistentInputCopiesByXramAddress_.end(),
      std::make_pair(copy.xramAddress, copyIndex),
      [](const PersistentInputCopyXramIndexEntry& lhs,
         const std::pair<uint32_t, size_t>& rhs) {
        return lhs.xramAddress < rhs.first ||
               (lhs.xramAddress == rhs.first && lhs.copyIndex < rhs.second);
      });
  if (entryIt == persistentInputCopiesByXramAddress_.end() ||
      entryIt->xramAddress != copy.xramAddress ||
      entryIt->copyIndex != copyIndex) {
    return;
  }

  size_t entryPosition =
      static_cast<size_t>(entryIt - persistentInputCopiesByXramAddress_.begin());
  persistentInputCopiesByXramAddress_.erase(entryIt);
  refreshPersistentInputCopyXramIndexPrefix(entryPosition);
}

void TorqExecutable::invalidatePersistentInputCopy(size_t copyIndex) {
  if (copyIndex >= persistentInputCopies_.size()) {
    return;
  }

  auto& copy = persistentInputCopies_[copyIndex];
  if (!copy.valid) {
    return;
  }

  removePersistentInputCopyFromXramIndex(copyIndex);
  if (copy.buffer) {
    iree_hal_buffer_release(copy.buffer);
    copy.buffer = nullptr;
  }
  copy.valid = false;
  copy.effectiveOffset = 0;
  copy.bindingRange = 0;
  copy.xramAddress = 0;
  copy.bindingSize = 0;
}

void TorqExecutable::invalidatePersistentInputCopiesForXramRange(
    uint32_t xramAddress, uint64_t size) {
  if (size == 0 || persistentInputCopiesByXramAddress_.empty()) {
    return;
  }

  const uint64_t xramEnd = saturatedRangeEnd(xramAddress, size);
  auto endIt = std::lower_bound(
      persistentInputCopiesByXramAddress_.begin(),
      persistentInputCopiesByXramAddress_.end(), xramEnd,
      [](const PersistentInputCopyXramIndexEntry& lhs, uint64_t rhs) {
        return lhs.xramAddress < rhs;
      });

  std::vector<size_t> copiesToInvalidate;
  auto entryIt = endIt;
  while (entryIt != persistentInputCopiesByXramAddress_.begin()) {
    --entryIt;
    if (entryIt->prefixMaxEnd <= xramAddress) {
      break;
    }

    if (entryIt->xramEnd <= xramAddress) {
      continue;
    }

    size_t copyIndex = entryIt->copyIndex;
    if (copyIndex >= persistentInputCopies_.size()) {
      continue;
    }

    const auto& copy = persistentInputCopies_[copyIndex];
    if (copy.valid &&
        xramRangesOverlap(xramAddress, size, copy.xramAddress, copy.bindingSize)) {
      copiesToInvalidate.push_back(copyIndex);
    }
  }

  for (size_t copyIndex : copiesToInvalidate) {
    invalidatePersistentInputCopy(copyIndex);
  }
}


iree_status_t TorqExecutable::syncBinding(int binding_id, iree_hal_torq_Binding_table_t binding, iree_hal_executable_dispatch_state_v0_t *state, bool from_host) {
    auto *torqState = iree_hal_torq_dispatch_state_cast(state);

    auto isReadOnly = ns(Binding_is_read_only(binding));

    if (isReadOnly && !from_host) {     
      return iree_ok_status();
    }

    auto isWriteOnly = ns(Binding_is_write_only(binding));

    if (isWriteOnly && from_host) {     
      return iree_ok_status();
    }

    auto bindingId = ns(Binding_id(binding));

    if (bindingId >= torqState->dispatch_state.binding_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "binding id %d out of range [0, %d)", bindingId, torqState->dispatch_state.binding_count);
    }

    auto bindingOffset = ns(Binding_offset(binding));
    auto bindingSize = ns(Binding_size(binding));

    if (bindingOffset >= torqState->binding_range_storage[bindingId]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "binding offset %d out of range [0, %zu)", bindingOffset, torqState->binding_range_storage[bindingId]);
    }

    if (bindingOffset + bindingSize > torqState->binding_range_storage[bindingId]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "binding size %d with offset %d out of range [0, %zu)", bindingSize, bindingOffset, torqState->binding_range_storage[bindingId]);
    }

    auto bindingAddress = ns(Binding_address(binding));

    if (!torqState->binding_ptr_storage[bindingId]) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "binding %d has no mapped host pointer", bindingId);
    }

    uint8_t *hostAddress = (uint8_t *)torqState->binding_ptr_storage[bindingId] + bindingOffset;


    if (from_host) {

      invalidatePersistentInputCopiesForXramRange(bindingAddress, bindingSize);
      if (!torq_->writeXram(bindingAddress, bindingSize, hostAddress)) {
        return iree_make_status(IREE_STATUS_INTERNAL, "failed to writeXram()");
      }

    } else {

      if (!torq_->readXram(bindingAddress, bindingSize, hostAddress)) {
        return iree_make_status(IREE_STATUS_INTERNAL, "failed to readXram()");
      }

    }

    // dump the whole binding data (input or output)
    if (from_host && !isWriteOnly) {
      dump::dumpBinding(executableName(), bindingId, hostAddress, bindingSize, true);
    } else if (!from_host && !isReadOnly) {
      dump::dumpBinding(executableName(), bindingId, hostAddress, bindingSize, false);
    }
    
    if (testVectorWriter_) {
      if (from_host) {
        testVectorWriter_->loadXram(0, bindingAddress, hostAddress, bindingSize, 4, "hex", "binding.in." + std::to_string(bindingId) + ".hex");
      } else {                
        testVectorWriter_->saveXram(nextJobId_ == 0 ? 0: nextJobId_ - 1, bindingAddress, hostAddress, bindingSize, 4, "hex", "binding.out." + std::to_string(bindingId) + ".hex");        
      }
    }

    return iree_ok_status();
}



static iree_status_t torq_execute_step_by_step(TorqHw *torq, int32_t code_entry, int32_t job_id) {

  // TODO: we reserve 640byte in lram for the code data,
  // to get this reserved value from somewhere
  static uint32_t const constexpr MAX_CODE_SIZE = 640;

  LOGD << "Reading code data from LRAM at entry point " << code_entry;

  std::vector<uint8_t> code_data_vec(MAX_CODE_SIZE);
  torq->readLram(code_entry, code_data_vec.size(), code_data_vec.data());

  LOGV << "Beginning to parse code data";

  auto cmds = parseCommandsUntilUnknown(code_data_vec);
  if (cmds.empty()) {
    return iree_make_status(IREE_STATUS_INTERNAL, "failed to parse code data");
  }

  uint32_t byte_offset = code_entry;

  std::vector<uint32_t> cmd_offsets;

  // patch the program to insert software breakpoints and collect the command offsets
  for (auto &cmd : cmds) {

    auto wasLastCommand = cmd->isLastCmd;

    cmd->isLastCmd = 1;
    cmd_offsets.push_back(byte_offset);
    byte_offset += cmd->serializedSize();

    // ignore other commands that would not be executed
    if (wasLastCommand) {
      break;
    }
  }

  auto patched_code = serializeCommands(cmds);
  auto patched_cmds = parseCommandsUntilUnknown(patched_code);

  if (patched_cmds.size() != cmds.size()) {
    LOGD << "Failed to patch code data, expected " << cmds.size() << " commands, but got " << patched_cmds.size();
    printCommands(patched_cmds);    

    return iree_make_status(IREE_STATUS_INTERNAL, "failed to parse patched code data");
  }

  // write back code with the sw breakpoints inserted
  torq->writeLram(code_entry, patched_code.size(), patched_code.data());

  // run each command and resume after each breakpoint
  for (int i = 0 ; i < cmd_offsets.size(); i++) {

    LOGV << "--------------";
    LOGV << "Stepping NPU at address " << cmd_offsets[i] << " command:";

    LOGV << "Registers before:";
    torq->printNssRegs();

    LOGV << "Executing instruction:";
    if (TorqLogger::request_log_level == TORQ_LOG_VERBOSE) {
      patched_cmds.at(i)->print();
    }

    if (!torq->start(cmd_offsets[i])) {
      LOGD << "Failed to start NPU at address " << cmd_offsets[i];
      return iree_make_status(IREE_STATUS_INTERNAL, "failed to start NPU");
    }
    
    if (!torq->wait()) {
      LOGD << "Failed to wait for NPU at address " << cmd_offsets[i];
      return iree_make_status(IREE_STATUS_INTERNAL, "failed to wait for NPU");
    }

    LOGV << "Registers after:\n";
    torq->printNssRegs();

  }

  return iree_ok_status();
}



iree_status_t TorqExecutable::initialize() {  
  
  uint32_t xram_base, xram_size;
  iree_status_t ret = iree_ok_status();
  std::unique_ptr<TorqDispatchEventLog> eventLog{nullptr};

  auto executableDef = this->executableDef();
  
  auto runtimeVersion = ns(ExecutableDef_runtime_version_get(executableDef));

  if (runtimeVersion != 1) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "executable runtime version %d does not match expected version %d", runtimeVersion, 1);
  }

  if (TorqEventLog::isProfilingEnabled()) {
    eventLog.reset(TorqEventLog::get().startDispatch(executableName(), EventType::INIT));
  }

  TORQ_ADD_PROFILING_EVENT_BEGIN(eventLog, EventType::INIT_COMPUTE_XRAM_FOOTPRINT);
  ret = compute_xram_footprint(executableDef, &xram_base, &xram_size);
  TORQ_ADD_PROFILING_EVENT_END(eventLog, EventType::INIT_COMPUTE_XRAM_FOOTPRINT);

  if (!iree_status_is_ok(ret)) {
    return ret;
  }

  LOGD << "xram_base " << xram_base << " xram_end " << (xram_base + xram_size)  << " xram_size " << xram_size;
  LOGD << "Running executable " << executableName();

  // Skip hardware initialization entirely for executables with no XRAM footprint.
  if (xram_size == 0) {
    LOGD << "No XRAM footprint, skipping hardware initialization";
    return iree_ok_status();
  }

  torq_ = newTorqHw(FLAG_torq_hw_type, xram_base, xram_size);

  if (!torq_.get()) {
      return iree_make_status(IREE_STATUS_INTERNAL, "failed to instantiate TorqHw");
  }

  auto hwId = ns(ExecutableDef_hw_id_get(executableDef));

  if (!torq_->isHwCompatible(hwId)) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE, "incompatible hardware, executable requires hw id %d", hwId);
  }

  TORQ_ADD_PROFILING_EVENT_BEGIN(eventLog, EventType::INIT_HW_OPEN);
  bool opened = torq_->open();
  TORQ_ADD_PROFILING_EVENT_END(eventLog, EventType::INIT_HW_OPEN);

  if (!opened) {
      return iree_make_status(IREE_STATUS_INTERNAL, "failed to open TorqHw");
  }

  if (FLAG_torq_clear_memory) {
    TORQ_ADD_PROFILING_EVENT_BEGIN(eventLog, EventType::INIT_CLEAR_MEMORY);
    std::vector<uint8_t> zeros(512 * 1024, 0);
    torq_->writeLram(0, zeros.size(), zeros.data());

    std::vector<uint8_t> zeros1(xram_size, 0);
    torq_->writeXram(xram_base, zeros1.size(), zeros1.data());
    TORQ_ADD_PROFILING_EVENT_END(eventLog, EventType::INIT_CLEAR_MEMORY);
  }

  testVectorWriter_ = dump::createTestVectorWriter(executableName());

  {
    TORQ_ADD_PROFILING_EVENT_BEGIN(eventLog, EventType::INIT_LOAD_CODE_SEGMENTS);
    ret = loadNpuCode();
    TORQ_ADD_PROFILING_EVENT_END(eventLog, EventType::INIT_LOAD_CODE_SEGMENTS);
  }

  if (!iree_status_is_ok(ret)) {
    return ret;
  }

  {
    TORQ_ADD_PROFILING_EVENT_BEGIN(eventLog, EventType::INIT_LOAD_HOST_CODE);
    ret = loadHostCode();
    TORQ_ADD_PROFILING_EVENT_END(eventLog, EventType::INIT_LOAD_HOST_CODE);
  }

  if (!iree_status_is_ok(ret)) {
    return ret;
  }

  // Avoid attributing this setup delay to any profiled phase.
  eventLog.reset();

  return iree_ok_status();

}


iree_status_t TorqExecutable::writeInputs(
    iree_hal_executable_dispatch_state_v0_t* state,
    std::vector<uint8_t>& zeroCopyAttached) {
  auto *torqState = iree_hal_torq_dispatch_state_cast(state);

  if (!iree_hal_torq_ExecutableDef_bindings_is_present(executableDef())) {
    if (!hasHardware()) {
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INTERNAL, "bindings are missing");
  }

  auto bindings = ns(ExecutableDef_bindings_get(executableDef()));
  auto bindings_count = iree_hal_torq_Binding_vec_len(bindings);
  if (persistentInputCopies_.size() < bindings_count) {
    persistentInputCopies_.resize(bindings_count);
  }
  std::vector<int> flushedFds;

  // simulate mapping bindings in the IO MMU of the NPU by copying the buffers in the cmodel
  for (size_t i = 0; i < bindings_count; i++) {
    iree_hal_torq_Binding_table_t binding = iree_hal_torq_Binding_vec_at(bindings, i);
    uint32_t binding_id = ns(Binding_id(binding));

    LOGD << "Loading binding " << binding_id << " index " << i;

    size_t effectiveOffset = 0;
    const torq_hw_device_buffer_t *deviceBuffer =
        getZeroCopyDeviceBuffer(torqState, binding, &effectiveOffset);
    if (hasHardware() && deviceBuffer) {
      auto bindingAddress = ns(Binding_address(binding));
      auto bindingSize = ns(Binding_size(binding));
      iree_status_t cacheStatus = iree_ok_status();
      const bool syncAlreadyDone =
          hasSyncedDeviceBufferFd(flushedFds, deviceBuffer);
      if (shouldExplicitlySyncZeroCopyDmaBufs() &&
          !ns(Binding_is_write_only(binding)) && !syncAlreadyDone) {
        cacheStatus = torq_hw_device_buffer_flush_range(
            const_cast<torq_hw_device_buffer_t *>(deviceBuffer), effectiveOffset, bindingSize);
        if (iree_status_is_ok(cacheStatus)) {
          markSyncedDeviceBufferFd(flushedFds, deviceBuffer);
        }
      }
      if (iree_status_is_ok(cacheStatus) &&
          torq_->attachBinding(*deviceBuffer, bindingAddress, effectiveOffset, bindingSize)) {
        if (i >= zeroCopyAttached.size()) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "zero-copy attach table index %zu out of range [0, %zu)",
              i, zeroCopyAttached.size());
        }
        zeroCopyAttached[i] = 1;
        continue;
      }
      iree_status_ignore(cacheStatus);
    }

    const bool immutableInput = isImmutableInputBinding(torqState, binding);
    if (immutableInput && i < persistentInputCopies_.size()) {
      auto* buffer = getBindingBuffer(torqState, binding);
      auto effectiveOffset = getBindingEffectiveOffset(torqState, binding);
      auto bindingAddress = ns(Binding_address(binding));
      auto bindingSize = ns(Binding_size(binding));
      auto bindingRange = torqState->binding_range_storage[binding_id];
      const auto& cachedCopy = persistentInputCopies_[i];
      if (cachedCopy.valid && cachedCopy.buffer == buffer &&
          cachedCopy.effectiveOffset == effectiveOffset &&
          cachedCopy.bindingRange == bindingRange &&
          cachedCopy.xramAddress == bindingAddress &&
          cachedCopy.bindingSize == bindingSize) {
        continue;
      }
    }

    iree_status_t ret = syncBinding(i, binding, state, true);

    if (!iree_status_is_ok(ret)) {
      return ret;
    }

    if (immutableInput && i < persistentInputCopies_.size()) {
      auto* buffer = getBindingBuffer(torqState, binding);
      auto& cachedCopy = persistentInputCopies_[i];
      removePersistentInputCopyFromXramIndex(i);
      if (cachedCopy.buffer != buffer) {
        if (cachedCopy.buffer) {
          iree_hal_buffer_release(cachedCopy.buffer);
        }
        cachedCopy.buffer = buffer;
        iree_hal_buffer_retain(buffer);
      }
      cachedCopy.valid = true;
      cachedCopy.effectiveOffset = getBindingEffectiveOffset(torqState, binding);
      cachedCopy.bindingRange = torqState->binding_range_storage[binding_id];
      cachedCopy.xramAddress = ns(Binding_address(binding));
      cachedCopy.bindingSize = ns(Binding_size(binding));
      addPersistentInputCopyToXramIndex(i);
    }
  }

  return iree_ok_status();

}

// reads the outputs from XRAM to the user buffers
iree_status_t TorqExecutable::readOutputs(
    iree_hal_executable_dispatch_state_v0_t* state,
    std::vector<uint8_t>& zeroCopyAttached) {
  auto *torqState = iree_hal_torq_dispatch_state_cast(state);
  
  if (!iree_hal_torq_ExecutableDef_bindings_is_present(executableDef())) {
    return iree_ok_status();
  }

  auto bindings = ns(ExecutableDef_bindings_get(executableDef()));
  auto bindings_count = iree_hal_torq_Binding_vec_len(bindings);

  iree_status_t ret = iree_ok_status();
  std::vector<int> invalidatedFds;

  // simulate mapping bindings in the IO MMU of the NPU by copying back the binding buffers from the cmodel
  for (size_t i = 0; i < bindings_count; i++) {

    iree_hal_torq_Binding_table_t binding = iree_hal_torq_Binding_vec_at(bindings, i);
    uint32_t binding_id = ns(Binding_id(binding));

    LOGD << "Reading binding " << binding_id << " index " << i;

    size_t effectiveOffset = 0;
    const torq_hw_device_buffer_t *deviceBuffer =
        getZeroCopyDeviceBuffer(torqState, binding, &effectiveOffset);
    if (deviceBuffer && !ns(Binding_is_read_only(binding)) &&
        i < zeroCopyAttached.size() && zeroCopyAttached[i]) {
      auto bindingAddress = ns(Binding_address(binding));
      auto bindingSize = ns(Binding_size(binding));
      if (!torq_->detachBinding(*deviceBuffer, bindingAddress, effectiveOffset, bindingSize)) {
        return iree_make_status(IREE_STATUS_INTERNAL, "failed to detach zero-copy binding %d", binding_id);
      }
      zeroCopyAttached[i] = 0;
      iree_status_t invalidateStatus = iree_ok_status();
      const bool syncAlreadyDone =
          hasSyncedDeviceBufferFd(invalidatedFds, deviceBuffer);
      if (shouldExplicitlySyncZeroCopyDmaBufs() && !syncAlreadyDone) {
        invalidateStatus = torq_hw_device_buffer_invalidate_range(
            const_cast<torq_hw_device_buffer_t *>(deviceBuffer), effectiveOffset, bindingSize);
        if (iree_status_is_ok(invalidateStatus)) {
          markSyncedDeviceBufferFd(invalidatedFds, deviceBuffer);
        }
      }
      IREE_RETURN_IF_ERROR(invalidateStatus);
      continue;
    }

    ret = syncBinding(i, binding, state, false);

    if (!iree_status_is_ok(ret)) {
      return ret;
    }

  }

  return iree_ok_status();

}

ns(ExecutableDef_table_t) TorqExecutable::executableDef() {
  return ns(ExecutableDef_as_root(nativeExecutable_->program));
}

std::string TorqExecutable::executableName() {
  return ns(ExecutableDef_executable_name(executableDef()));
}

// runs all the host actions in the executable, one after the other
iree_status_t TorqExecutable::executeActions(TorqDispatchEventLog* eventLog) {
  
  actionIndex_ = 0;
  nextJobId_ = 0;
  
  auto actions = ns(ExecutableDef_actions_get(executableDef()));
  size_t actionsCount = ns(HostAction_vec_len(actions));

  while(actionIndex_ < actionsCount) {

    auto action = ns(HostAction_vec_at(actions, actionIndex_));
    auto paramsType = ns(HostAction_params_type(action));

    LOGD << "Starting action " << actionIndex_;

    auto eventType = toEventType(paramsType);
    if (eventLog) {
        eventLog->addEvent(eventType, Event::BEGIN, actionIndex_);
    }

    auto status = processAction(action);
    
    if (eventLog) {
        eventLog->addEvent(eventType, Event::END, actionIndex_);
    }

    if (!iree_status_is_ok(status)) {
      return status;
    }
    
    LOGD << "Completed action " << actionIndex_;

    dump::dumpBuffers(torq_.get(), actionIndex_, nativeExecutable_);

    actionIndex_++;
  }

  return iree_ok_status();

}

iree_status_t TorqExecutable::setupDumpDirectories() {
  iree_status_t status = dump::setupIODumpDirs(executableName());
  if (!iree_status_is_ok(status)) return status;

  return dump::createJobDirectory(executableName(), 0);
}

void TorqExecutable::writeInitialStateToTestVector() {

  testVectorWriter_->beginInvocation(nextInvocationId_ - 1);

  auto segments = ns(ExecutableDef_code_get(executableDef()));
  auto segments_count = iree_hal_torq_Segment_vec_len(segments);

  for (size_t i = 0; i < segments_count; i++) {
    iree_hal_torq_Segment_table_t segment = iree_hal_torq_Segment_vec_at(segments, i);

    // skip segments that are uninitialized
    if (!iree_hal_torq_Segment_data_is_present(segment)) {
      continue;
    }

    flatbuffers_uint8_vec_t code = iree_hal_torq_Segment_data_get(segment);
    size_t code_len = flatbuffers_uint8_vec_len(code);

    if (code_len != ns(Segment_size(segment))) {
      LOGD << "Invalid buffer size for segment " << i;
      continue;
    }

    uint32_t xram_code_address = iree_hal_torq_Segment_xram_address_get(segment);

    testVectorWriter_->loadXram(0, xram_code_address, code, code_len, 4, "hex", "segment_" + std::to_string(i) + ".txt");
  }

  // For the moment we don't expect to have 1:1 corresponence between bus dumps from this excution and test vector
  // execution because we don't initialize the memory to the same state, even with the code below we can't initialize
  // DTCM/ITCM so there may be always some differences. We will focus on ensuring outputs (described in by save operations
  // in the test vector) match

  /*
  // write out the full XRAM data to ensure it is initialized in the same during tests (important for out of bound reads)
  std::vector<uint8_t> xram_data(torq_->xramSize(), 0);
  torq_->readXram(torq_->xramBase(), torq_->xramSize(), xram_data.data());
  testVectorWriter_->loadXram(0, torq_->xramBase(), xram_data.data(), xram_data.size(), 4, "hex", "initial_xram.txt");

  // write out the full LRAM data to ensure it is initialized in the same during tests (important for out of bound reads)
  std::vector<uint8_t> lram_data(torq_->lramSize(), 0);
  torq_->readLram(0, lram_data.size(), lram_data.data());
  testVectorWriter_->loadLram(0, 0, lram_data.data(), lram_data.size(), 4, "hex", "initial_lram.txt");
  */

}


iree_status_t TorqExecutable::executeDispatch(iree_hal_executable_dispatch_state_v0_t* state) {

  iree_status_t status;
  std::unique_ptr<TorqDispatchEventLog> eventLog{nullptr};

  if (TorqEventLog::isProfilingEnabled()) {
    eventLog.reset(TorqEventLog::get().startDispatch(executableName(), EventType::DISPATCH));
  }

  // setup dump directories before execution if necessary
  TORQ_ADD_PROFILING_EVENT_BEGIN(eventLog, EventType::INIT_PREPARE_OUTPUT_DIRS);
  status = setupDumpDirectories();
  TORQ_ADD_PROFILING_EVENT_END(eventLog, EventType::INIT_PREPARE_OUTPUT_DIRS);

  if (!iree_status_is_ok(status)) {
    return status;
  }

  // acquire the mutex to ensure only one execution is running, we will be changing
  // the XRAM contents with the inputs
  io::TorqMutex::ScopedLock lock(mutex_);

  // increment the invocation count id  
  nextInvocationId_++;

  if (testVectorWriter_) {
    writeInitialStateToTestVector();
  }
  
  if (hasHardware()) {
    // acquire the hardware, this won't start execution on it yet
    {
      TORQ_ADD_PROFILING_EVENT_BEGIN(eventLog, EventType::DISPATCH_ACQUIRE_HW_RESOURCES);
      const bool ressourceAcquired = torq_->acquire();
      TORQ_ADD_PROFILING_EVENT_END(eventLog, EventType::DISPATCH_ACQUIRE_HW_RESOURCES);
      if (!ressourceAcquired) {
        return iree_make_status(IREE_STATUS_INTERNAL, "failed to acquire hardware");
      }
    }

    std::vector<uint8_t> zeroCopyAttached;
    if (iree_hal_torq_ExecutableDef_bindings_is_present(executableDef())) {
      auto bindings = ns(ExecutableDef_bindings_get(executableDef()));
      zeroCopyAttached.resize(iree_hal_torq_Binding_vec_len(bindings), 0);
    }

    TORQ_ADD_PROFILING_EVENT_BEGIN(eventLog, EventType::DISPATCH_SYNC_BINDINGS_IN);
    status = writeInputs(state, zeroCopyAttached);
    TORQ_ADD_PROFILING_EVENT_END(eventLog, EventType::DISPATCH_SYNC_BINDINGS_IN);

    if (!iree_status_is_ok(status)) {
      iree_status_t cleanupStatus =
          cleanupZeroCopyBindings(torq_.get(), executableDef(), state, zeroCopyAttached);
      torq_->release();
      return iree_status_join(status, cleanupStatus);
    }

    // run all host actions using the hardware if necessary
    TORQ_ADD_PROFILING_EVENT_BEGIN(eventLog, EventType::DISPATCH_EXECUTE_ACTIONS);
    status = executeActions(eventLog.get());
    TORQ_ADD_PROFILING_EVENT_END(eventLog, EventType::DISPATCH_EXECUTE_ACTIONS);

    if (iree_status_is_ok(status)) {
      TORQ_ADD_PROFILING_EVENT_BEGIN(eventLog, EventType::DISPATCH_SYNC_BINDINGS_OUT);
      status = readOutputs(state, zeroCopyAttached);
      TORQ_ADD_PROFILING_EVENT_END(eventLog, EventType::DISPATCH_SYNC_BINDINGS_OUT);
    }

    iree_status_t cleanupStatus =
        cleanupZeroCopyBindings(torq_.get(), executableDef(), state, zeroCopyAttached);

    // release the hardware so that it can be used by another user
    TORQ_ADD_PROFILING_EVENT_BEGIN(eventLog, EventType::DISPATCH_RELEASE_HW_RESOURCES);
    torq_->release();
    TORQ_ADD_PROFILING_EVENT_END(eventLog, EventType::DISPATCH_RELEASE_HW_RESOURCES);

    if (!iree_status_is_ok(status) || !iree_status_is_ok(cleanupStatus)) {
      return iree_status_join(status, cleanupStatus);
    }
  } else {
    std::vector<uint8_t> zeroCopyAttached;
    if (iree_hal_torq_ExecutableDef_bindings_is_present(executableDef())) {
      auto bindings = ns(ExecutableDef_bindings_get(executableDef()));
      zeroCopyAttached.resize(iree_hal_torq_Binding_vec_len(bindings), 0);
    }
    TORQ_ADD_PROFILING_EVENT_BEGIN(eventLog, EventType::DISPATCH_SYNC_BINDINGS_IN);
    status = writeInputs(state, zeroCopyAttached);
    TORQ_ADD_PROFILING_EVENT_END(eventLog, EventType::DISPATCH_SYNC_BINDINGS_IN);

    if (!iree_status_is_ok(status)) {
      return status;
    }
    TORQ_ADD_PROFILING_EVENT_BEGIN(eventLog, EventType::DISPATCH_SYNC_BINDINGS_OUT);
    status = readOutputs(state, zeroCopyAttached);
    TORQ_ADD_PROFILING_EVENT_END(eventLog, EventType::DISPATCH_SYNC_BINDINGS_OUT);
    return status;
  }

  return iree_ok_status();

}

TorqExecutable::~TorqExecutable() {
  clearPersistentInputCopies();

  io::unloadLibrary(hostCodeLibHandle_);

  if (torq_) {
    torq_->close();
  }

}

iree_status_t TorqExecutable::loadNpuCode() {
  
  if (!iree_hal_torq_ExecutableDef_code_is_present(executableDef())) {
    return iree_make_status(IREE_STATUS_INTERNAL, "code is missing");
  }

  iree_hal_torq_Segment_vec_t segments = iree_hal_torq_ExecutableDef_code_get(executableDef());
  size_t segments_count = iree_hal_torq_Segment_vec_len(segments);


  for (size_t i = 0; i < segments_count; i++) {
    iree_hal_torq_Segment_table_t segment = iree_hal_torq_Segment_vec_at(segments, i);

    // skip segments that are uninitialized
    if (!iree_hal_torq_Segment_data_is_present(segment)) {
      continue;
    }

    flatbuffers_uint8_vec_t code = iree_hal_torq_Segment_data_get(segment);
    size_t code_len = flatbuffers_uint8_vec_len(code);

    if (code_len != ns(Segment_size(segment))) {
      return iree_make_status(IREE_STATUS_INTERNAL, "invalid buffer size");
    }

    uint32_t xram_code_address = iree_hal_torq_Segment_xram_address_get(segment);

    if (xram_code_address != 1 ) {
    
      LOGD << "Loading model segment at XRAM address " << xram_code_address << ", len " << code_len;

        invalidatePersistentInputCopiesForXramRange(xram_code_address, code_len);
        if (!torq_->writeXram(xram_code_address, code_len, code)) {
          return iree_make_status(IREE_STATUS_INTERNAL, "failed to writeXram()");
        }
    }
  }

  return iree_ok_status();

}

iree_status_t TorqExecutable::loadHostCode() {

  auto hostCodeData = ns(ExecutableDef_host_code(executableDef()));

  auto hostCodeLen = flatbuffers_uint8_vec_len(hostCodeData);

  // host code not present, no need to load it
  if (hostCodeLen == 0) {
    return iree_ok_status();
  }

  return io::loadLibraryFromMemory(
      reinterpret_cast<const uint8_t*>(hostCodeData), hostCodeLen,
      &hostCodeLibHandle_);
}

// Compute the dense (contiguous) strides in bytes from shape and element size.
// The element size is inferred from the innermost XRAM stride.
static std::vector<uint64_t> computeDenseStridesBytes(
    const uint32_t* xramStrides, const uint32_t* shape, int ndim) {
  std::vector<uint64_t> dense(ndim);
  if (ndim == 0) return dense;
  dense[ndim - 1] = xramStrides[ndim - 1]; // element size in bytes
  for (int i = ndim - 2; i >= 0; i--) {
    dense[i] = dense[i + 1] * shape[i + 1];
  }
  return dense;
}

// Check if any XRAM stride differs from the expected dense stride.
static bool hasNonTrivialStrides(
    const uint32_t* xramStrides, const uint32_t* shape, int ndim) {
  auto dense = computeDenseStridesBytes(xramStrides, shape, ndim);
  for (int i = 0; i < ndim; i++) {
    if (xramStrides[i] != dense[i]) return true;
  }
  return false;
}

static uint64_t computeStridedXramSpanBytes(
    const uint32_t* xramStrides, const uint32_t* shape, int ndim) {
  if (ndim <= 0) {
    return 0;
  }

  uint64_t span = xramStrides[ndim - 1];
  for (int i = 0; i < ndim; ++i) {
    if (shape[i] == 0) {
      return 0;
    }
    const uint64_t contribution = static_cast<uint64_t>(shape[i] - 1) * xramStrides[i];
    span = saturatedRangeEnd(span, contribution);
  }
  return span;
}

// Copy data from XRAM (with given strides) into a contiguous dense buffer.
// Finds the largest contiguous inner chunk to minimize the number of reads.
static void stridedXramRead(const TorqHw* torq, uint64_t baseAddress,
                            uint8_t* denseBuffer,
                            const uint32_t* xramStrides,
                            const uint32_t* shape, int ndim) {
  auto denseStrides = computeDenseStridesBytes(xramStrides, shape, ndim);

  // Find the outermost dimension where xram stride != dense stride.
  // All dimensions from splitDim onward are contiguous and can be copied as
  // a single chunk.
  int splitDim = 0;
  for (int i = ndim - 1; i >= 0; i--) {
    if (xramStrides[i] != denseStrides[i]) {
      splitDim = i + 1;
      break;
    }
  }

  // Size of each contiguous chunk (product of inner dims * element size)
  uint64_t chunkSize = xramStrides[ndim - 1]; // element size
  for (int i = splitDim; i < ndim; i++) {
    chunkSize *= shape[i];
  }

  // Number of chunks (product of outer dims)
  uint64_t numChunks = 1;
  for (int i = 0; i < splitDim; i++) {
    numChunks *= shape[i];
  }

  std::vector<uint64_t> indices(splitDim, 0);
  for (uint64_t c = 0; c < numChunks; c++) {
    uint64_t xramOffset = 0;
    uint64_t denseOffset = 0;
    for (int i = 0; i < splitDim; i++) {
      xramOffset += indices[i] * xramStrides[i];
      denseOffset += indices[i] * denseStrides[i];
    }

    torq->readXram(static_cast<uint32_t>(baseAddress + xramOffset),
                   chunkSize, denseBuffer + denseOffset);

    for (int i = splitDim - 1; i >= 0; i--) {
      indices[i]++;
      if (indices[i] < shape[i]) break;
      indices[i] = 0;
    }
  }
}

// Copy data from a contiguous dense buffer back to XRAM with given strides.
static void stridedXramWrite(TorqHw* torq, uint64_t baseAddress,
                             const uint8_t* denseBuffer,
                             const uint32_t* xramStrides,
                             const uint32_t* shape, int ndim) {
  auto denseStrides = computeDenseStridesBytes(xramStrides, shape, ndim);

  int splitDim = 0;
  for (int i = ndim - 1; i >= 0; i--) {
    if (xramStrides[i] != denseStrides[i]) {
      splitDim = i + 1;
      break;
    }
  }

  uint64_t chunkSize = xramStrides[ndim - 1];
  for (int i = splitDim; i < ndim; i++) {
    chunkSize *= shape[i];
  }

  uint64_t numChunks = 1;
  for (int i = 0; i < splitDim; i++) {
    numChunks *= shape[i];
  }

  std::vector<uint64_t> indices(splitDim, 0);
  for (uint64_t c = 0; c < numChunks; c++) {
    uint64_t xramOffset = 0;
    uint64_t denseOffset = 0;
    for (int i = 0; i < splitDim; i++) {
      xramOffset += indices[i] * xramStrides[i];
      denseOffset += indices[i] * denseStrides[i];
    }

    torq->writeXram(static_cast<uint32_t>(baseAddress + xramOffset),
                    chunkSize, denseBuffer + denseOffset);

    for (int i = splitDim - 1; i >= 0; i--) {
      indices[i]++;
      if (indices[i] < shape[i]) break;
      indices[i] = 0;
    }
  }
}

iree_status_t TorqExecutable::processStartHostAction(ns(HostAction_table_t) action, ns(StartHostParams_table_t) params) {

  std::string functionName = ns(StartHostParams_function_name(params));

  LOGD << "action: Starting host with function " << functionName;

  using HostFn = void(*)(iree_hal_executable_environment_v0_t*,
                         iree_hal_executable_dispatch_state_v0_t*,
                         iree_hal_executable_workgroup_state_v0_t*);
  auto hostFunction = reinterpret_cast<HostFn>(
      io::resolveSymbol(hostCodeLibHandle_, functionName));

  if (!hostFunction) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "Failed to find host function %s",
                            functionName.c_str());
  }

  auto args = ns(StartHostParams_args(params));
  auto sizes = ns(StartHostParams_sizes(params));

  if (flatbuffers_uint64_vec_len(args) != flatbuffers_uint64_vec_len(sizes)) {
    return iree_make_status(IREE_STATUS_INTERNAL, "Mismatched argument and size counts");
  }

  assert(sizeof(uintptr_t) == sizeof(uint64_t) && "Host pointer size not compatible");

  // Read optional stride metadata for strided XRAM access
  auto strides_fb = ns(StartHostParams_strides(params));
  auto shapes_fb = ns(StartHostParams_shapes(params));
  auto ndims_fb = ns(StartHostParams_ndims(params));
  bool hasStrideInfo = strides_fb && shapes_fb && ndims_fb;

  int numArgs = flatbuffers_uint64_vec_len(args);

  // Precompute per-arg offsets into the flattened strides/shapes arrays
  std::vector<int> argStrideOffsets(numArgs, 0);
  std::vector<int> argNdims(numArgs, 0);
  if (hasStrideInfo) {
    int offset = 0;
    for (int i = 0; i < numArgs; i++) {
      argStrideOffsets[i] = offset;
      argNdims[i] = flatbuffers_uint32_vec_at(ndims_fb, i);
      offset += argNdims[i];
    }
  }

  // read the arguments from XRAM (strided copy when strides differ from dense)
  std::vector<std::vector<uint8_t>> arguments;
  std::vector<void*> argumentAddresses;

  for (int i = 0; i < numArgs; i++) {
    auto xramAddress = flatbuffers_uint64_vec_at(args, i);
    auto size = flatbuffers_uint64_vec_at(sizes, i);
    std::vector<uint8_t> xramBuffer(size);

    bool strided = false;
    if (hasStrideInfo && argNdims[i] > 0) {
      const uint32_t* argStrides = strides_fb + argStrideOffsets[i];
      const uint32_t* argShape = shapes_fb + argStrideOffsets[i];
      if (hasNonTrivialStrides(argStrides, argShape, argNdims[i])) {
        strided = true;
        stridedXramRead(torq_.get(), xramAddress, xramBuffer.data(),
                        argStrides, argShape, argNdims[i]);
      }
    }
    if (!strided) {
      torq_->readXram(xramAddress, xramBuffer.size(), xramBuffer.data());
    }

    arguments.push_back(std::move(xramBuffer));
    argumentAddresses.push_back(arguments.back().data());

    if (testVectorWriter_) {

      // dump the inputs of the host program to the last job so that they can be checked when they are generated
      if (nextJobId_ > 0) {
        testVectorWriter_->saveXram(nextJobId_ - 1, xramAddress, arguments.back().data(), size, 4, "hex", 
          "host_program." + std::to_string(actionIndex_) + ".arg." + std::to_string(i) + ".txt");
      }

    }

  }

  // create the arguments for calling the code
  iree_hal_executable_environment_v0_t environment;
  iree_hal_executable_dispatch_state_v0_t dispatch_state;
  iree_hal_executable_workgroup_state_v0_t workgroup_state;

  memset(&environment, 0, sizeof(environment));
  memset(&dispatch_state, 0, sizeof(dispatch_state));
  memset(&workgroup_state, 0, sizeof(workgroup_state));

  dispatch_state.binding_count = argumentAddresses.size();
  dispatch_state.binding_ptrs = argumentAddresses.data();

  for (int i = 0; i < dispatch_state.binding_count; i++) {
      LOGD << "Argument " << i << ": 0x" << (uintptr_t)dispatch_state.binding_ptrs[i]
           << " xram: 0x" << flatbuffers_uint64_vec_at(args, i)
           << " size: " << flatbuffers_uint64_vec_at(sizes, i);
  }

  // call the code
  hostFunction(&environment, &dispatch_state, &workgroup_state);

  // read back the results to XRAM (strided write when strides differ from dense)
  for (int i = 0; i < numArgs; i++) {      

    auto xramAddress = flatbuffers_uint64_vec_at(args, i);
    uint64_t xramWriteSize = arguments[i].size();

    bool strided = false;
    if (hasStrideInfo && argNdims[i] > 0) {
      const uint32_t* argStrides = strides_fb + argStrideOffsets[i];
      const uint32_t* argShape = shapes_fb + argStrideOffsets[i];
      if (hasNonTrivialStrides(argStrides, argShape, argNdims[i])) {
        strided = true;
        xramWriteSize = computeStridedXramSpanBytes(argStrides, argShape, argNdims[i]);
        invalidatePersistentInputCopiesForXramRange(
            static_cast<uint32_t>(xramAddress), xramWriteSize);
        stridedXramWrite(torq_.get(), xramAddress, arguments[i].data(),
                         argStrides, argShape, argNdims[i]);
      }
    }
    if (!strided) {
      invalidatePersistentInputCopiesForXramRange(
          static_cast<uint32_t>(xramAddress), xramWriteSize);
      torq_->writeXram(xramAddress, arguments[i].size(), arguments[i].data());
    }

    if (testVectorWriter_) {

      // make sure the next job will be able to see the results of the host program in XRAM
      testVectorWriter_->loadXram(nextJobId_, xramAddress, arguments[i].data(), arguments[i].size(), 4, "hex", 
        "host_program." + std::to_string(actionIndex_) + ".result." + std::to_string(i) + ".txt");
    
    }

  }
  
  return iree_ok_status();
}

  iree_status_t TorqExecutable::processWaitHostAction(ns(HostAction_table_t) action, ns(WaitHostParams_table_t) params) {
    // host execution is synchronous for the moment so we don't need to do anything to wait
    return iree_ok_status();
  }

  iree_status_t TorqExecutable::processStartNSSAction(ns(HostAction_table_t) action, ns(StartNSSParams_table_t) params) {
      
      uint32_t code_entry = ns(StartNSSParams_start_address(params));

      LOGD << "action: Starting NPU with entry point at " << code_entry;

      if (nextJobId_ > 0) {
        auto status = dump::createJobDirectory(executableName(), nextJobId_);

        if (status != iree_ok_status()) {
          return status;
        }
      }

      if (FLAG_torq_step_by_step) {
        return torq_execute_step_by_step(torq_.get(), code_entry, actionIndex_);
      }

      LOGD << "Starting NSS job " << nextJobId_ << " with code entry " << code_entry;

      if (testVectorWriter_) {
        testVectorWriter_->writeCdescAddr(nextJobId_, code_entry);

        if (FLAG_torq_dump_bus_logs) {
          torq_->setDumpDirectory(testVectorWriter_->getJobDir(nextJobId_));
        }
      }

      if (!torq_->start(code_entry)) {
          return iree_make_status(IREE_STATUS_INTERNAL, "torq failed to start with code entry %d", code_entry);
      }
      
      nextJobId_++;
      
      return iree_ok_status();

  }

  iree_status_t TorqExecutable::processWaitNSSAction(ns(HostAction_table_t) action, ns(WaitNSSParams_table_t) params) {
    
    LOGD << "action: Waiting for NSS job " << (nextJobId_ - 1);
    
    if (!torq_->wait()) {
        return iree_make_status(IREE_STATUS_INTERNAL, "torq failed to wait");
    }

    if (!torq_->end()) {
        return iree_make_status(IREE_STATUS_INTERNAL, "torq failed to end");
    }    
    
    return iree_ok_status();
  }

  iree_status_t TorqExecutable::writeHostCopyDataToTestVector(const std::vector<uint8_t> & data, uint32_t outputAddress, uint32_t inputOffset, uint32_t outputOffset, ns(BufferType_enum_t) inputType, ns(BufferType_enum_t) outputType) {

      const std::string fileName = "host_copy." + std::to_string(actionIndex_) + ".output.offset" + std::to_string(outputOffset) + ".txt";

      // load the output buffer in the next job id so that it will see it
      if (outputType == ns(BufferType_LRAM)) {
        testVectorWriter_->loadLram(nextJobId_, outputAddress + outputOffset, data.data(), data.size(), 4, "hex", fileName);
      } else if (outputType == ns(BufferType_XRAM)) {
        testVectorWriter_->loadXram(nextJobId_, outputAddress + outputOffset, data.data(), data.size(), 4, "hex", fileName);        
      } else {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unsupported output memory type when dumping descriptors");
      }

      // if the copy happens after the NPU ran we also add a save to check the data in the previous job was correctly computed
      if (nextJobId_ > 0) {

        const std::string fileName = "host_copy." + std::to_string(actionIndex_) + ".input.offset" + std::to_string(inputOffset) + ".txt";

        if (inputType == ns(BufferType_LRAM)) {
          testVectorWriter_->saveLram(nextJobId_ - 1, inputOffset, data.data(), data.size(), 4, "hex", fileName);
        } else if (inputType == ns(BufferType_XRAM)) {
          testVectorWriter_->saveXram(nextJobId_ - 1, inputOffset, data.data(), data.size(), 4, "hex", fileName);        
        } else {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unsupported input memory type when dumping descriptors");
        }

      }

      return iree_ok_status();

  }

  iree_status_t TorqExecutable::copyElement(int inputOffset, int outputOffset, int size, ns(HostCopyParams_table_t) params) {
    std::vector<uint8_t> data(size);

    uint32_t inputAddress = ns(HostCopyParams_input_address(params));
    uint32_t outputAddress = ns(HostCopyParams_output_address(params));

    ns(BufferType_enum_t) inputType = ns(HostCopyParams_input_buffer_type(params));
    ns(BufferType_enum_t) outputType = ns(HostCopyParams_output_buffer_type(params));  

    if (inputType == ns(BufferType_XRAM) && outputType == ns(BufferType_LRAM)) {
      if (!torq_->readXram(inputAddress + inputOffset, size, data.data())) {
        return iree_make_status(IREE_STATUS_INTERNAL, "failed to read from xram");
      }
      if (!torq_->writeLram(outputAddress + outputOffset, size, data.data())) {
        return iree_make_status(IREE_STATUS_INTERNAL, "failed to write to lram");
      }
    }
    else if (inputType == ns(BufferType_LRAM) && outputType == ns(BufferType_XRAM)) {
      if (!torq_->readLram(inputAddress + inputOffset, size, data.data())) {
        return iree_make_status(IREE_STATUS_INTERNAL, "failed to read from lram");
      }
      if (!torq_->writeXram(outputAddress + outputOffset, size, data.data())) {
        return iree_make_status(IREE_STATUS_INTERNAL, "failed to write to xram");
      }  
    }
    else if (inputType == ns(BufferType_XRAM) && outputType == ns(BufferType_XRAM)) {
      if (!torq_->readXram(inputAddress + inputOffset, size, data.data())) {
        return iree_make_status(IREE_STATUS_INTERNAL, "failed to read from xram");
      }
      if (!torq_->writeXram(outputAddress + outputOffset, size, data.data())) {
        return iree_make_status(IREE_STATUS_INTERNAL, "failed to write to xram");
      }  
    }
    else {
      // XRAM to XRAM is handled by optimized path in processHostCopyAction
      // LRAM to LRAM should not happen (we must use NSS based copy for that) 
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unsupported output memory type");
    }
    
    if (testVectorWriter_) {
      auto status = writeHostCopyDataToTestVector(data, outputAddress, inputOffset, outputOffset, inputType, outputType);
      if (!iree_status_is_ok(status)) {
        return status;
      }      
    }

    return iree_ok_status();
  };

  iree_status_t TorqExecutable::copyDimension(int dim, int inputOffset, int outputOffset, ns(HostCopyParams_table_t) params) {
    auto inputStridesBytes = ns(HostCopyParams_input_strides_bytes(params));
    auto outputStridesBytes = ns(HostCopyParams_output_strides_bytes(params));
    auto shape = ns(HostCopyParams_shape(params));
    auto rank = flatbuffers_uint32_vec_len(shape);

    //cout << "dim " << dim << " rank " << rank << " Copying element: inputOffset=" << inputOffset << " outputOffset=" << outputOffset << endl;

    if (dim == rank) {
      //cout << "****** last dim == rank, copying element, dim=" << dim << " rank=" << rank << endl;
      auto elementSize = ns(HostCopyParams_element_size_bytes(params));
      auto ret = copyElement(inputOffset, outputOffset, elementSize, params);

      if (!iree_status_is_ok(ret)) {
        return ret;
      }
    } else {
      for (int i = 0; i < flatbuffers_uint32_vec_at(shape, dim); i++) {
        auto ret = copyDimension(dim + 1, inputOffset + i * flatbuffers_uint32_vec_at(inputStridesBytes, dim),
                                outputOffset + i * flatbuffers_uint32_vec_at(outputStridesBytes, dim), params);
        if (!iree_status_is_ok(ret)) {
          return ret;
        }
      }
    }
    return iree_ok_status();
  };

  iree_status_t TorqExecutable::processHostCopyAction(ns(HostAction_table_t) action, ns(HostCopyParams_table_t) params) {

    uint32_t inputAddress = ns(HostCopyParams_input_address(params));
    uint32_t outputAddress = ns(HostCopyParams_output_address(params));

    ns(BufferType_enum_t) inputType = ns(HostCopyParams_input_buffer_type(params));
    ns(BufferType_enum_t) outputType = ns(HostCopyParams_output_buffer_type(params));  

    auto inputStridesBytes = ns(HostCopyParams_input_strides_bytes(params));
    auto outputStridesBytes = ns(HostCopyParams_output_strides_bytes(params));

    auto shape = ns(HostCopyParams_shape(params));

    auto rank = flatbuffers_uint32_vec_len(shape);

    if (flatbuffers_uint32_vec_len(outputStridesBytes) != rank) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "incorrect output strides rank");
    }

    if (flatbuffers_uint32_vec_len(inputStridesBytes) != rank) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "incorrect input strides rank");
    }

    LOGD << "action: Copying " << ns(BufferType_name(inputType)) << "[" << inputAddress << 
              "] to " << ns(BufferType_name(outputType)) << "[" << outputAddress << "]";

    iree_status_t ret = iree_ok_status();
    if (outputType == ns(BufferType_XRAM)) {
      const uint64_t outputSpan =
          rank == 0 ? ns(HostCopyParams_element_size_bytes(params))
                    : computeStridedXramSpanBytes(outputStridesBytes, shape, rank);
      invalidatePersistentInputCopiesForXramRange(outputAddress, outputSpan);
    }

    // Optimized copy for XRAM to XRAM using direct memory access and linear iteration without.
    // recursion
    bool memcpySupported = torq_->getType() == TorqHw::Type::SIMULATOR || torq_->getType() == TorqHw::Type::ASTRA_MACHINA;
    if (memcpySupported && inputType == ns(BufferType_XRAM) && outputType == ns(BufferType_XRAM)) {
      auto inputAddress = static_cast<const uint8_t*>(
        torq_->startXramReadAccess(ns(HostCopyParams_input_address(params))));
      auto outputAddress = static_cast<uint8_t*>(
        torq_->startXramWriteAccess(ns(HostCopyParams_output_address(params))));
      if (!inputAddress || !outputAddress) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "XRAM access issue");
      }
      auto inputStridesBytes = ns(HostCopyParams_input_strides_bytes(params));
      auto outputStridesBytes = ns(HostCopyParams_output_strides_bytes(params));
      auto shape = ns(HostCopyParams_shape(params));
      auto rank = flatbuffers_uint32_vec_len(shape);
      auto elementSize = ns(HostCopyParams_element_size_bytes(params));

      LOGD << "Doing optimized host copy for xram to xram, rank=" << rank;
      LOGD << "Input strides bytes: " <<  inputStridesBytes;
      LOGD << "Output strides bytes: " << outputStridesBytes;
      LOGD << "Input Address: " << static_cast<const void*>(inputAddress);
      LOGD << "Output Address: " << static_cast<void*>(outputAddress);
      LOGD << "Element size bytes: " << elementSize;
      
      // Calculate total number of elements
      uint32_t totalElements = 1;
      for (int dim = 0; dim < rank; dim++) {
        totalElements *= flatbuffers_uint32_vec_at(shape, dim);
      }
      
      // Iterate through all elements sequentially
      for (uint32_t elemIdx = 0; elemIdx < totalElements; elemIdx++) {
        uint32_t inputOffset = 0;
        uint32_t outputOffset = 0;
        uint32_t remaining = elemIdx;
        
        // Convert linear index to multi-dimensional offsets
        for (int dim = rank - 1; dim >= 0; dim--) {
          uint32_t dimSize = flatbuffers_uint32_vec_at(shape, dim);
          uint32_t idx = remaining % dimSize;
          remaining /= dimSize;
          
          inputOffset += idx * flatbuffers_uint32_vec_at(inputStridesBytes, dim);
          outputOffset += idx * flatbuffers_uint32_vec_at(outputStridesBytes, dim);
        }
        
        auto src = inputAddress + inputOffset;
        auto dst = outputAddress + outputOffset;
        switch (elementSize)
        {
        case 1:
          *dst = *src;
          break;
        case 2:
          *(uint16_t*)dst = *(uint16_t*)src;
          break;
        case 4:
          *(uint32_t*)dst = *(uint32_t*)src;
          break;
        default:
          memcpy(dst, src, elementSize);
          break;
        }
      }
      if (!torq_->endXramReadAccess() || !torq_->endXramWriteAccess()) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "XRAM access issue");
      }
    }
    else {
      ret = copyDimension(0, 0, 0, params);
    }

    return ret;
  }

  iree_status_t TorqExecutable::processAllocAction(ns(HostAction_table_t) action, ns(AllocParams_table_t) params) {

    uint32_t id = ns(AllocParams_id(params));
    uint32_t address = ns(AllocParams_address(params));
    uint32_t size = ns(AllocParams_size(params));
    ns(BufferType_enum_t) bufferType = ns(AllocParams_buffer_type(params));

    LOGD << "action: Allocating buffer #" << id << ": " << ns(BufferType_name(bufferType)) << "[" << address << "] of size " << size;

    if (FLAG_torq_clear_memory) {
      std::vector<uint8_t> bufferData(size, 0);

      if (ns(BufferType_XRAM) == bufferType) {

        invalidatePersistentInputCopiesForXramRange(address, size);
        if (!torq_->writeXram(address, size, bufferData.data())) {
          return iree_make_status(IREE_STATUS_INTERNAL, "failed to write to xram");
        }

      }
    }

    return iree_ok_status();

  }

  iree_status_t TorqExecutable::processDeallocAction(ns(HostAction_table_t) action, ns(DeallocParams_table_t) params) {
    
    uint32_t id = ns(DeallocParams_id(params));

    LOGD << "action: Deallocating buffer #" << id;

    return iree_ok_status();

  }


iree_status_t TorqExecutable::processAction(ns(HostAction_table_t) action) {
    
    auto params = ns(HostAction_params_get(action));
    auto paramsType = ns(HostAction_params_type(action));

    iree_status_t result = iree_ok_status();

    switch (paramsType) {

      case ns(HostActionParams_HostCopyParams):
        result = processHostCopyAction(action, (ns(HostCopyParams_table_t)) params);
        break;

      case ns(HostActionParams_StartNSSParams):
        result = processStartNSSAction(action, (ns(StartNSSParams_table_t)) params);
        break;

      case ns(HostActionParams_WaitNSSParams):
        result = processWaitNSSAction(action, (ns(WaitNSSParams_table_t)) params);
        break;
      
      case ns(HostActionParams_AllocParams):
        result = processAllocAction(action, (ns(AllocParams_table_t)) params);
        break;

      case ns(HostActionParams_DeallocParams):
        result = processDeallocAction(action, (ns(DeallocParams_table_t)) params);
        break;

      case ns(HostActionParams_StartHostParams):
        result = processStartHostAction(action, (ns(StartHostParams_table_t)) params);
        break;

      case ns(HostActionParams_WaitHostParams):
        result = processWaitHostAction(action, (ns(WaitHostParams_table_t)) params);
        break;

      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unknown host action type %d", paramsType);
    }

    return result;

}

}  // namespace synaptics
