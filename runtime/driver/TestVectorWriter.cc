#include "TestVectorWriter.h"

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include "iree/base/api.h"

namespace synaptics {

namespace {

// Helper function to create a directory
iree_status_t createDirectory(const std::string& dir_name) {
  std::error_code ec;
  std::filesystem::create_directory(dir_name, ec);
  if (ec && ec != std::errc::file_exists) {
    return iree_make_status(IREE_STATUS_INTERNAL, 
                            "failed to create directory %s: %s", 
                            dir_name.c_str(), ec.message().c_str());
  }
  return iree_ok_status();
}

// Helper function to write binary data to a file
bool writeBinaryFile(const std::string& file_name, const void* data, size_t size) {
  std::ofstream wf;
  wf.open(file_name, std::ios::out | std::ios::binary);
  if (!wf.good()) {
    return false;
  }

  if (data) {
    wf.write(static_cast<const char*>(data), size);
  }
  wf.close();
  return wf.good();
}

// Helper function to create empty file
bool createEmptyFile(const std::string& file_name) {
  std::ofstream f(file_name);
  return f.good();
}

}  // namespace

TestVectorWriter::TestVectorWriter(std::string executable_name, std::string base_dir)
    : executable_name_(executable_name), base_dir_(base_dir), 
      current_invocation_id_(-1) {}

iree_status_t TestVectorWriter::ensureJobDirectory(int job_id) {
  std::string job_dir = invocation_dir_ + "/job" + std::to_string(job_id);
  
  // Check if job directory already exists
  if (std::filesystem::exists(job_dir)) {
    return iree_ok_status();
  }
  
  // Create job directory
  auto status = createDirectory(job_dir);
  if (!iree_status_is_ok(status)) return status;
  
  // Create empty init and exit list files
  std::string init_list = job_dir + "/tv.init.mem.lst";
  std::string exit_list = job_dir + "/tv.exit.mem.lst";
  
  if (!createEmptyFile(init_list)) {
    return iree_make_status(IREE_STATUS_INTERNAL, "failed to create %s", init_list.c_str());
  }
  
  if (!createEmptyFile(exit_list)) {
    return iree_make_status(IREE_STATUS_INTERNAL, "failed to create %s", exit_list.c_str());
  }
  
  return iree_ok_status();
}

iree_status_t TestVectorWriter::beginInvocation(int invocation_id) {
  current_invocation_id_ = invocation_id;
  invocation_dir_ = base_dir_ + "/" + executable_name_ + "/invocation" + std::to_string(invocation_id);
  
  // Remove existing invocation directory to start fresh
  std::error_code ec;
  std::filesystem::remove_all(invocation_dir_, ec);
  if (ec) {
    // Ignore errors if directory doesn't exist
  }
  
  auto status = createDirectory(base_dir_);
  if (!iree_status_is_ok(status)) return status;
  
  status = createDirectory(base_dir_ + "/" + executable_name_);
  if (!iree_status_is_ok(status)) return status;
  
  status = createDirectory(invocation_dir_);
  if (!iree_status_is_ok(status)) return status;
  
  return iree_ok_status();
}

iree_status_t TestVectorWriter::writeCdescAddr(int job_id, uint32_t addr) {
  if (current_invocation_id_ < 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, 
                            "beginInvocation must be called first");
  }
  
  auto status = ensureJobDirectory(job_id);
  if (!iree_status_is_ok(status)) return status;
  
  std::string job_dir = invocation_dir_ + "/job" + std::to_string(job_id);
  std::string path = job_dir + "/tv.cdesc_addr.txt";
  std::ofstream fp(path);
  if (!fp.is_open()) {
    return iree_make_status(IREE_STATUS_INTERNAL, "failed to open %s", path.c_str());
  }
  
  fp << "0x" << std::hex << std::setw(8) << std::setfill('0') << addr << "\n";
  fp.close();
  
  if (!fp.good()) {
    return iree_make_status(IREE_STATUS_INTERNAL, "failed to write %s", path.c_str());
  }
  
  return iree_ok_status();
}

iree_status_t TestVectorWriter::writeMemEntry(int job_id, const std::string& cmd, uint32_t addr, 
                              const void* data, size_t size, 
                              uint32_t word_width, const std::string& fmt,
                              const std::string& filename) {
  if (current_invocation_id_ < 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, 
                            "beginInvocation must be called first");
  }
  
  auto status = ensureJobDirectory(job_id);
  if (!iree_status_is_ok(status)) return status;
  
  std::string job_dir = invocation_dir_ + "/job" + std::to_string(job_id);
  
  if (size % word_width != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, 
                            "size must be divisible by word_width");
  }
  
  bool is_load = (cmd == "load" || cmd == "xload");
  bool is_init = is_load;  // load operations go to init.mem.lst
  
  // Write the data file
  std::string data_path = job_dir + "/" + filename;
  
  if (fmt == "hex") {

    std::ofstream fp(data_path);
    if (!fp.is_open()) {
      return iree_make_status(IREE_STATUS_INTERNAL, "failed to open %s", data_path.c_str());
    }
    
    if (data) {
      const unsigned char* bytes = static_cast<const unsigned char*>(data);
      size_t num_words = size / word_width;
      
      // Write each word in little endian byte order
      for (size_t word = 0; word < num_words; word++) {
        size_t word_offset = word * word_width;
        // Write bytes of this word in reverse order (little endian)
        for (size_t byte = 0; byte < word_width; byte++) {
          fp << std::hex << std::setw(2) << std::setfill('0') 
             << (int)bytes[word_offset + (word_width - 1 - byte)];
        }
        fp << '\n';
      }
    }
    fp.close();
    
    if (!fp.good()) {
      return iree_make_status(IREE_STATUS_INTERNAL, "failed to write %s", data_path.c_str());
    }
  } else if (fmt == "raw") {
    if (!writeBinaryFile(data_path, data, size)) {
      return iree_make_status(IREE_STATUS_INTERNAL, "failed to write %s", data_path.c_str());
    }
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, 
                            "fmt must be 'hex' or 'raw', got: %s", fmt.c_str());
  }
  
  // Append entry to the appropriate .lst file
  std::string lst_path = getMemListPath(job_id, is_init);
  FILE* fp = fopen(lst_path.c_str(), "a");
  if (!fp) {
    return iree_make_status(IREE_STATUS_INTERNAL, "failed to open %s", lst_path.c_str());
  }
  
  fprintf(fp, "%s  0x%08x %10zu %3d  %s  %s\n", 
          cmd.c_str(), addr, size, word_width, fmt.c_str(), filename.c_str());
  fclose(fp);
  
  return iree_ok_status();
}

iree_status_t TestVectorWriter::loadLram(int job_id, uint32_t addr, const void* data, size_t size, 
                       uint32_t word_width, const std::string& fmt,
                       const std::string& filename) {
  return writeMemEntry(job_id, "load", addr, data, size, word_width, fmt, filename);
}

iree_status_t TestVectorWriter::saveLram(int job_id, uint32_t addr, const void* data, size_t size,
                       uint32_t word_width, const std::string& fmt,
                       const std::string& filename) {
  return writeMemEntry(job_id, "save", addr, data, size, word_width, fmt, filename);
}

iree_status_t TestVectorWriter::loadXram(int job_id, uint32_t addr, const void* data, size_t size,
                       uint32_t word_width, const std::string& fmt,
                       const std::string& filename) {
  return writeMemEntry(job_id, "xload", addr, data, size, word_width, fmt, filename);
}

iree_status_t TestVectorWriter::saveXram(int job_id, uint32_t addr, const void* data, size_t size,
                       uint32_t word_width, const std::string& fmt,
                       const std::string& filename) {
  return writeMemEntry(job_id, "xsave", addr, data, size, word_width, fmt, filename);
}

std::string TestVectorWriter::getJobDir(int job_id) const {
  return invocation_dir_ + "/job" + std::to_string(job_id);
}

std::string TestVectorWriter::getInvocationDir() const {
  return invocation_dir_;
}

std::string TestVectorWriter::getMemListPath(int job_id, bool is_init) const {
  std::string job_dir = invocation_dir_ + "/job" + std::to_string(job_id);
  return job_dir + "/tv." + (is_init ? "init" : "exit") + ".mem.lst";
}

}  // namespace synaptics
