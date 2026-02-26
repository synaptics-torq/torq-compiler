#include "native_executable.h"
#include "TorqHw.h"
#include <mutex>
#include <vector>
#include <string>

#include "iree/base/internal/flatcc/parsing.h"

#include "torq_executable_def_reader.h"

#undef ns
#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(iree_hal_torq, x) 

namespace synaptics {

class TorqExecutable {

public:
  TorqExecutable(iree_hal_torq_native_executable_t* nativeExecutable) 
    : torq_(nullptr), nativeExecutable_(nativeExecutable), actionIndex_(0) {}

  iree_status_t initialize();
  
  ns(ExecutableDef_table_t) executableDef();

  std::string executableName();  

  // execute the dispatch using the input/output bindings in the specified state
  iree_status_t executeDispatch(iree_hal_executable_dispatch_state_v0_t* state);

  ~TorqExecutable();

private:

  std::unique_ptr<TorqHw> torq_;
  iree_hal_torq_native_executable_t* nativeExecutable_;   
  bool pendingHostCopies_{false}; // true if there are host copies pending and therefore bindings are not up to date yet
  int actionIndex_;
  int jobId_{0};
  void *hostCodeLibHandle_{nullptr};

  // this mutex is used to ensure we only run one dispatch at a time, this is important
  // because there is one XRAM per executable so we can only load one state at a time
  std::mutex mutex_;

  // writes the inputs to XRAM from the user buffers
  iree_status_t writeInputs(iree_hal_executable_dispatch_state_v0_t* state);

  // reads the outputs from XRAM to the user buffers
  iree_status_t readOutputs(iree_hal_executable_dispatch_state_v0_t* state);

  iree_status_t loadNpuCode();
  iree_status_t loadHostCode();

  iree_status_t processStartHostAction(ns(HostAction_table_t) action, ns(StartHostParams_table_t) params);

  iree_status_t processWaitHostAction(ns(HostAction_table_t) action, ns(WaitHostParams_table_t) params);

  iree_status_t processStartNSSAction(ns(HostAction_table_t) action, ns(StartNSSParams_table_t) params);

  iree_status_t processWaitNSSAction(ns(HostAction_table_t) action, ns(WaitNSSParams_table_t) params);

  iree_status_t processHostCopyAction(ns(HostAction_table_t) action, ns(HostCopyParams_table_t) params);

  iree_status_t processAllocAction(ns(HostAction_table_t) action, ns(AllocParams_table_t) params);

  iree_status_t processDeallocAction(ns(HostAction_table_t) action, ns(DeallocParams_table_t) params);

  iree_status_t processAction(ns(HostAction_table_t) action);
  
  iree_status_t dumpMemData(const std::vector<uint8_t> & data, uint32_t outputAddress, uint32_t outputOffset, ns(BufferType_enum_t) outputType);

  iree_status_t copyElement(int inputOffset, int outputOffset, int size, ns(HostCopyParams_table_t) params);

  iree_status_t copyDimension(int dim, int inputOffset, int outputOffset, ns(HostCopyParams_table_t) params);

  // runs all the host actions in the executable, one after the other
  iree_status_t executeActions();

  iree_status_t setupDumpDirectories();
  
  int lastCompletedJobId();

};

}