#ifndef RUNTIME_DRIVER_TESTVECTORWRITER_H_
#define RUNTIME_DRIVER_TESTVECTORWRITER_H_

#include <string>
#include "iree/base/api.h"

namespace synaptics {

// TestVectorWriter generates test vector files in the format documented in:
// extras/doc/Internal-notes/hw_test_vectors_format.md
//
// The directory structure is:
// - base_dir/executable_name/invocation<N>/job<M>/
// Each invocation represents one execution of the executable
// Each job within an invocation represents a single hardware task
//
// The format consists of:
// - Memory list files (tv.init.mem.lst, tv.exit.mem.lst)
// - Command descriptor address file (tv.cdesc_addr.txt)
// - Data files referenced by the memory lists
class TestVectorWriter {
public:
  TestVectorWriter(std::string executable_name, std::string base_dir);

  // Begin a new invocation (creates invocation directory)
  // Must be called before writing any data
  iree_status_t beginInvocation(int invocation_id);

  // Write the command descriptor address file (tv.cdesc_addr.txt)
  // The job directory will be created automatically if it doesn't exist
  iree_status_t writeCdescAddr(int job_id, uint32_t addr);

  // Write data and add entry to memory list file
  // Job directory is created automatically if needed
  // cmd: "load", "save", "xload", or "xsave"
  // fmt: "hex" or "raw"
  iree_status_t writeMemEntry(int job_id, const std::string& cmd, uint32_t addr, 
                                const void* data, size_t size, 
                                uint32_t word_width, const std::string& fmt,
                                const std::string& filename);

  // Convenience methods for specific memory types
  iree_status_t loadLram(int job_id, uint32_t addr, const void* data, size_t size, 
                         uint32_t word_width, const std::string& fmt,
                         const std::string& filename);

  iree_status_t saveLram(int job_id, uint32_t addr, const void* data, size_t size,
                         uint32_t word_width, const std::string& fmt,
                         const std::string& filename);

  iree_status_t loadXram(int job_id, uint32_t addr, const void* data, size_t size,
                         uint32_t word_width, const std::string& fmt,
                         const std::string& filename);

  iree_status_t saveXram(int job_id, uint32_t addr, const void* data, size_t size,
                         uint32_t word_width, const std::string& fmt,
                         const std::string& filename);

  std::string getJobDir(int job_id) const;
  std::string getInvocationDir() const;
  
private:
  std::string getMemListPath(int job_id, bool is_init) const;
  iree_status_t ensureJobDirectory(int job_id);

  std::string executable_name_;
  std::string base_dir_;
  std::string invocation_dir_;
  int current_invocation_id_;
};

}  // namespace synaptics

#endif  // RUNTIME_DRIVER_TESTVECTORWRITER_H_
