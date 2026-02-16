#pragma once

#include <mutex>
#include <deque>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <optional>
#include <cassert>

namespace synaptics {

enum class EventType {
    DISPATCH_BEGIN,
    DISPATCH_END,
    HOST_COPY_BEGIN,
    HOST_COPY_END,
    NSS_START_BEGIN,
    NSS_START_END,
    NSS_WAIT_BEGIN,
    NSS_WAIT_END,
    ALLOC_BEGIN,
    ALLOC_END,
    DEALLOC_BEGIN,
    DEALLOC_END,
    HOST_START_BEGIN,
    HOST_START_END,
    HOST_WAIT_BEGIN,
    HOST_WAIT_END
};


inline std::string eventTypeToString(EventType type) {
    switch (type) {
        case EventType::DISPATCH_BEGIN:
            return "DISPATCH_BEGIN";
        case EventType::DISPATCH_END:
            return "DISPATCH_END";
        case EventType::HOST_COPY_BEGIN:
            return "HOST_COPY_BEGIN";
        case EventType::HOST_COPY_END:
            return "HOST_COPY_END";
        case EventType::NSS_START_BEGIN:
            return "NSS_START_BEGIN";
        case EventType::NSS_START_END:
            return "NSS_START_END";
        case EventType::NSS_WAIT_BEGIN:
            return "NSS_WAIT_BEGIN";
        case EventType::NSS_WAIT_END:
            return "NSS_WAIT_END";
        case EventType::ALLOC_BEGIN:
            return "ALLOC_BEGIN";
        case EventType::ALLOC_END:
            return "ALLOC_END";
        case EventType::DEALLOC_BEGIN:
            return "DEALLOC_BEGIN";
        case EventType::DEALLOC_END:
            return "DEALLOC_END";
        case EventType::HOST_START_BEGIN:
            return "HOST_START_BEGIN";
        case EventType::HOST_START_END:
            return "HOST_START_END";
        case EventType::HOST_WAIT_BEGIN:
            return "HOST_WAIT_BEGIN";
        case EventType::HOST_WAIT_END:
            return "HOST_WAIT_END";
    }
    assert(false && "unknown event type");
}


struct Event {
    std::chrono::steady_clock::time_point timestamp;
    EventType type;
    int actionIndex;
};

struct DispatchEvents {
    std::string dispatchName;
    size_t dispatchIndex;
    std::vector<Event> events;
};

class TorqEventLog;

class TorqDispatchEventLog {

public:

    TorqDispatchEventLog(TorqEventLog& log, std::string dispatchName, size_t dispatchIndex);

    void close();

    void addEvent(EventType type, int actionIndex);

    ~TorqDispatchEventLog();

    TorqDispatchEventLog(const TorqDispatchEventLog&) = delete;
    TorqDispatchEventLog(TorqDispatchEventLog&&) = delete;
    TorqDispatchEventLog& operator=(const TorqDispatchEventLog&) = delete;
    TorqDispatchEventLog& operator=(TorqDispatchEventLog&&) = delete;

private:    
    TorqEventLog &log_;
    DispatchEvents *events_;
    bool closed_{false};
};

class TorqEventLog {
public:
    ~TorqEventLog() {
        close();
    }

    TorqDispatchEventLog* startDispatch(const std::string& dispatchName);

    static bool isProfilingEnabled();

    static TorqEventLog& get();

private:

    bool open(std::string profilingFile);

    void flush();

    void close();

    void flushUnlocked();

    void addDispatchEvents(DispatchEvents* events) {
        std::lock_guard<std::mutex> lock(events_mutex_);
        dispatchEvents_.push_back(events);

        // TODO: we should avoid flushing at each dispatch but 
        // this requires to hook at the right place the flushing,
        // flushing at destructor of the singleton is not good 
        // because it can be too late
        flushUnlocked();
    }

    friend class TorqDispatchEventLog;

    std::ofstream profilingFile_;
    std::mutex events_mutex_;
    std::deque<DispatchEvents *> dispatchEvents_;
    size_t nextDispatchIndex_ = 0;
};

} // namespace synaptics
