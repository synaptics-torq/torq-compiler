#include "TorqEventLog.h"
#include <algorithm> // Add this line
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

using namespace std;

namespace synaptics {

bool TorqEventLog::dumpEvents(const std::string& filename){
    std::lock_guard<std::mutex> lock(events_mutex_);
    profilingFile_.open(filename);
    if (!profilingFile_.is_open()) {
        cerr << "Failed to create timing dump file: " << filename << endl;
        return false;
    }

    if (events_.empty()) {
        profilingFile_.close();
        return true;
    }
    profilingFile_ << "actionIndex, elapsed_time(us), timestamp(us), event, location\n";
    auto start = events_.front().timestamp;

    for (const auto& e : events_) {
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(e.timestamp - start).count();
        auto timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(e.timestamp.time_since_epoch()).count();

        std::ostringstream time_str;
        time_str << elapsed_us;

        std::string type_str;
        switch (e.type) {
            case EventType::HOST_COPY_START:      type_str = "HOST_COPY_START"; break;
            case EventType::HOST_COPY_END:        type_str = "HOST_COPY_END"; break;
            case EventType::NSS_START:            type_str = "NSS_START"; break;
            case EventType::NSS_START_END:        type_str = "NSS_START_END"; break;
            case EventType::NSS_WAIT_START:       type_str = "NSS_WAIT_START"; break;
            case EventType::NSS_WAIT_END:         type_str = "NSS_WAIT_END"; break;
            case EventType::ALLOC_START:          type_str = "ALLOC_START"; break;
            case EventType::ALLOC_END:            type_str = "ALLOC_END"; break;
            case EventType::DEALLOC_START:        type_str = "DEALLOC_START"; break;
            case EventType::DEALLOC_END:          type_str = "DEALLOC_END"; break;
            case EventType::HOST_PROGRAM_START:   type_str = "HOST_PROGRAM_START"; break;
            case EventType::HOST_PROGRAM_END:     type_str = "HOST_PROGRAM_END"; break;
            default:                             type_str = "Unknown"; break;
        }
        std::string sanitized_location = e.location;
        sanitized_location.erase(std::remove(sanitized_location.begin(), sanitized_location.end(), '\n'), sanitized_location.end());

        profilingFile_ << e.actionIndex << "," << elapsed_us << "," << timestamp_us << "," << type_str << "," << sanitized_location << "\n";
    }
    profilingFile_.close();
    return true;
}
}
