#include "native_executable.h"
#include "torq_command_buffer.h"
#include "TorqHw.h"
#include "TestVectorWriter.h"
#include "TorqIO.h"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

#include "iree/base/internal/flatcc/parsing.h"

#include "torq_executable_def_reader.h"

#undef ns
#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(iree_hal_torq, x) 

namespace synaptics {

class TorqDispatchEventLog;

class TorqExecutable {

public:
  TorqExecutable(iree_hal_torq_native_executable_t* nativeExecutable) 
    : torq_(nullptr), nativeExecutable_(nativeExecutable), actionIndex_(0) {}

  iree_status_t initialize();
  
  ns(ExecutableDef_table_t) executableDef();

  std::string executableName();  

  // returns true if this executable requires hardware i.e. has XRAM footprint
  bool hasHardware() const { return torq_ != nullptr; }

  // execute the dispatch using the input/output bindings in the specified state
  iree_status_t executeDispatch(iree_hal_executable_dispatch_state_v0_t* state);

  ~TorqExecutable();

private:

  std::unique_ptr<TorqHw> torq_;
  std::unique_ptr<TestVectorWriter> testVectorWriter_{nullptr};
  iree_hal_torq_native_executable_t* nativeExecutable_;   
  bool pendingHostCopies_{false}; // true if there are host copies pending and therefore bindings are not up to date yet
  int actionIndex_;
  int nextJobId_{0};
  void *hostCodeLibHandle_{nullptr};
  int nextInvocationId_{0};

  // this mutex is used to ensure we only run one dispatch at a time, this is important
  // because there is one XRAM per executable so we can only load one state at a time
  io::TorqMutex mutex_;

  struct PersistentInputCopy {
    bool valid{false};
    iree_hal_buffer_t* buffer{nullptr};
    iree_device_size_t effectiveOffset{0};
    // Part of cache identity so a changed HAL binding view cannot bypass range validation.
    iree_device_size_t bindingRange{0};
    uint32_t xramAddress{0};
    uint32_t bindingSize{0};
  };

  struct PersistentInputCopyXramIndexEntry {
    uint32_t xramAddress{0};
    uint64_t xramEnd{0};
    size_t copyIndex{0};
    uint64_t prefixMaxEnd{0};
  };

  std::vector<PersistentInputCopy> persistentInputCopies_;
  std::vector<PersistentInputCopyXramIndexEntry> persistentInputCopiesByXramAddress_;

  // writes the inputs to XRAM from the user buffers
  iree_status_t writeInputs(iree_hal_executable_dispatch_state_v0_t* state,
                            std::vector<uint8_t>& zeroCopyAttached);

  // reads the outputs from XRAM to the user buffers
  iree_status_t readOutputs(iree_hal_executable_dispatch_state_v0_t* state,
                            std::vector<uint8_t>& zeroCopyAttached);

  iree_status_t loadNpuCode();
  iree_status_t loadHostCode();

  iree_status_t processStartHostAction(ns(HostAction_table_t) action, ns(StartHostParams_table_t) params);

  iree_status_t processWaitHostAction(ns(HostAction_table_t) action, ns(WaitHostParams_table_t) params);

  iree_status_t processStartNSSAction(ns(HostAction_table_t) action, ns(StartNSSParams_table_t) params);

  iree_status_t processWaitNSSAction(ns(HostAction_table_t) action, ns(WaitNSSParams_table_t) params);

  iree_status_t processHostCopyAction(ns(HostAction_table_t) action, ns(HostCopyParams_table_t) params, const iree_hal_executable_dispatch_state_v0_t* state);

  iree_status_t processAllocAction(ns(HostAction_table_t) action, ns(AllocParams_table_t) params);

  iree_status_t processDeallocAction(ns(HostAction_table_t) action, ns(DeallocParams_table_t) params);

  iree_status_t processAction(ns(HostAction_table_t) action, const iree_hal_executable_dispatch_state_v0_t* state);
  
  iree_status_t writeHostCopyDataToTestVector(const std::vector<uint8_t> & data, uint32_t outputAddress, uint32_t inputOffset, uint32_t outputOffset, ns(BufferType_enum_t) inputType, ns(BufferType_enum_t) outputType);

  iree_status_t copyElement(int inputOffset, int outputOffset, int size, ns(HostCopyParams_table_t) params, const iree_hal_executable_dispatch_state_v0_t* state);

  iree_status_t copyDimension(int dim, int inputOffset, int outputOffset, ns(HostCopyParams_table_t) params, const iree_hal_executable_dispatch_state_v0_t* state);

  iree_status_t resolveDynamicByteOffset(ns(HostCopyParams_table_t) params, const iree_hal_executable_dispatch_state_v0_t* state, uint32_t* byteOffset);

  // runs all the host actions in the executable, one after the other
  iree_status_t executeActions(TorqDispatchEventLog* eventLog, const iree_hal_executable_dispatch_state_v0_t* state);

  iree_status_t setupDumpDirectories();

  iree_status_t syncBinding(int binding_id, iree_hal_torq_Binding_table_t binding, iree_hal_executable_dispatch_state_v0_t *state, bool from_host);

  // writes out the initial XRAM and LRAM state to the test vector dump
  void writeInitialStateToTestVector();

  void clearPersistentInputCopies();

  void addPersistentInputCopyToXramIndex(size_t copyIndex);

  void removePersistentInputCopyFromXramIndex(size_t copyIndex);

  void refreshPersistentInputCopyXramIndexPrefix(size_t firstEntry);

  void invalidatePersistentInputCopy(size_t copyIndex);

  void invalidatePersistentInputCopiesForXramRange(uint32_t xramAddress, uint64_t size);
  
};

}
