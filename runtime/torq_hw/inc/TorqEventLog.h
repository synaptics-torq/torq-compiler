#pragma once
#include <mutex>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <optional>

namespace synaptics {

enum class EventType {

    //Host events
    HOST_COPY_START,      // For HostCopyParams
    HOST_COPY_END,        // For HostCopyParams
    NSS_START,            // For StartNSSParams
    NSS_START_END,        // For EndNSSParams
    NSS_WAIT_START,       // For WaitNSSParams
    NSS_WAIT_END,         // For WaitNSSParams
    ALLOC_START,          // For AllocParams
    ALLOC_END,            // For AllocParams
    DEALLOC_START,        // For DeallocParams
    DEALLOC_END,          // For DeallocParams
    HOST_PROGRAM_START,   // For StartHostParams
    HOST_PROGRAM_END,     // For EndHostParams

};

struct Event {
    std::chrono::steady_clock::time_point timestamp;
    EventType type;
    int actionIndex = -1;      // which action this event is for
    std::string location = "";
};

class TorqEventLog {
public:
    ~TorqEventLog() {
        if (profilingFile_.is_open()) {
            profilingFile_.close();
        }
    }

    void addEvent(const Event& e) {
        std::lock_guard<std::mutex> lock(events_mutex_);
        events_.push_back(e);
    }

    bool dumpEvents(const std::string& filename);

private:    
    std::mutex events_mutex_;
    std::vector<Event> events_;
    std::ofstream profilingFile_;
};

} // namespace synaptics