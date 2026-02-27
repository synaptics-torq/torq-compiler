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
    DISPATCH_PREPARE_OUTPUT_DIRS_BEGIN,
    DISPATCH_PREPARE_OUTPUT_DIRS_END,
    DISPATCH_COMPUTE_XRAM_FOOTPRINT_BEGIN,
    DISPATCH_COMPUTE_XRAM_FOOTPRINT_END,
    // Backend-specific runtime/session initialization (TorqHw::open()).
    DISPATCH_HW_OPEN_BEGIN,
    DISPATCH_HW_OPEN_END,
    DISPATCH_CLEAR_MEMORY_BEGIN,
    DISPATCH_CLEAR_MEMORY_END,
    // Upload executable code segments into device-accessible memory.
    DISPATCH_LOAD_CODE_SEGMENTS_BEGIN,
    DISPATCH_LOAD_CODE_SEGMENTS_END,
    DISPATCH_SYNC_BINDINGS_IN_BEGIN,
    DISPATCH_SYNC_BINDINGS_IN_END,
    DISPATCH_LOAD_HOST_CODE_BEGIN,
    DISPATCH_LOAD_HOST_CODE_END,
    DISPATCH_LOAD_HW_RESOURCES_BEGIN,
    DISPATCH_LOAD_HW_RESOURCES_END,
    HAL_CALL_BEGIN,
    HAL_CALL_END,
    HAL_QUEUE_WAIT_BEGIN,
    HAL_QUEUE_WAIT_END,
    HAL_QUEUE_APPLY_DEFERRED_BEGIN,
    HAL_QUEUE_APPLY_DEFERRED_END,
    HAL_QUEUE_SIGNAL_BEGIN,
    HAL_QUEUE_SIGNAL_END,
    HAL_COMMAND_BUFFER_COPY_BUFFER_BEGIN,
    HAL_COMMAND_BUFFER_COPY_BUFFER_END,
    HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET_BEGIN,
    HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET_END,
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
        case EventType::DISPATCH_PREPARE_OUTPUT_DIRS_BEGIN:
            return "DISPATCH_PREPARE_OUTPUT_DIRS_BEGIN";
        case EventType::DISPATCH_PREPARE_OUTPUT_DIRS_END:
            return "DISPATCH_PREPARE_OUTPUT_DIRS_END";
        case EventType::DISPATCH_COMPUTE_XRAM_FOOTPRINT_BEGIN:
            return "DISPATCH_COMPUTE_XRAM_FOOTPRINT_BEGIN";
        case EventType::DISPATCH_COMPUTE_XRAM_FOOTPRINT_END:
            return "DISPATCH_COMPUTE_XRAM_FOOTPRINT_END";
        case EventType::DISPATCH_HW_OPEN_BEGIN:
            return "DISPATCH_HW_OPEN_BEGIN";
        case EventType::DISPATCH_HW_OPEN_END:
            return "DISPATCH_HW_OPEN_END";
        case EventType::DISPATCH_CLEAR_MEMORY_BEGIN:
            return "DISPATCH_CLEAR_MEMORY_BEGIN";
        case EventType::DISPATCH_CLEAR_MEMORY_END:
            return "DISPATCH_CLEAR_MEMORY_END";
        case EventType::DISPATCH_LOAD_CODE_SEGMENTS_BEGIN:
            return "DISPATCH_LOAD_CODE_SEGMENTS_BEGIN";
        case EventType::DISPATCH_LOAD_CODE_SEGMENTS_END:
            return "DISPATCH_LOAD_CODE_SEGMENTS_END";
        case EventType::DISPATCH_SYNC_BINDINGS_IN_BEGIN:
            return "DISPATCH_SYNC_BINDINGS_IN_BEGIN";
        case EventType::DISPATCH_SYNC_BINDINGS_IN_END:
            return "DISPATCH_SYNC_BINDINGS_IN_END";
        case EventType::DISPATCH_LOAD_HOST_CODE_BEGIN:
            return "DISPATCH_LOAD_HOST_CODE_BEGIN";
        case EventType::DISPATCH_LOAD_HOST_CODE_END:
            return "DISPATCH_LOAD_HOST_CODE_END";
        case EventType::DISPATCH_LOAD_HW_RESOURCES_BEGIN:
            return "DISPATCH_LOAD_HW_RESOURCES_BEGIN";
        case EventType::DISPATCH_LOAD_HW_RESOURCES_END:
            return "DISPATCH_LOAD_HW_RESOURCES_END";
        case EventType::HAL_CALL_BEGIN:
            return "HAL_CALL_BEGIN";
        case EventType::HAL_CALL_END:
            return "HAL_CALL_END";
        case EventType::HAL_QUEUE_WAIT_BEGIN:
            return "HAL_QUEUE_WAIT_BEGIN";
        case EventType::HAL_QUEUE_WAIT_END:
            return "HAL_QUEUE_WAIT_END";
        case EventType::HAL_QUEUE_APPLY_DEFERRED_BEGIN:
            return "HAL_QUEUE_APPLY_DEFERRED_BEGIN";
        case EventType::HAL_QUEUE_APPLY_DEFERRED_END:
            return "HAL_QUEUE_APPLY_DEFERRED_END";
        case EventType::HAL_QUEUE_SIGNAL_BEGIN:
            return "HAL_QUEUE_SIGNAL_BEGIN";
        case EventType::HAL_QUEUE_SIGNAL_END:
            return "HAL_QUEUE_SIGNAL_END";
        case EventType::HAL_COMMAND_BUFFER_COPY_BUFFER_BEGIN:
            return "HAL_COMMAND_BUFFER_COPY_BUFFER_BEGIN";
        case EventType::HAL_COMMAND_BUFFER_COPY_BUFFER_END:
            return "HAL_COMMAND_BUFFER_COPY_BUFFER_END";
        case EventType::HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET_BEGIN:
            return "HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET_BEGIN";
        case EventType::HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET_END:
            return "HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET_END";
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

    TorqDispatchEventLog(
        TorqEventLog& log, std::string dispatchName, size_t dispatchIndex,
        EventType beginEventType = EventType::DISPATCH_BEGIN,
        EventType endEventType = EventType::DISPATCH_END
    );

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
    EventType beginEventType_;
    EventType endEventType_;
    bool closed_{false};
};

class TorqEventLog {
public:
    ~TorqEventLog() {
        close();
    }

    TorqDispatchEventLog* startDispatch(
        const std::string& dispatchName,
        EventType beginEventType = EventType::DISPATCH_BEGIN,
        EventType endEventType = EventType::DISPATCH_END
    );

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
    }

    friend class TorqDispatchEventLog;

    std::ofstream profilingFile_;
    std::mutex events_mutex_;
    std::deque<DispatchEvents *> dispatchEvents_;
    size_t nextDispatchIndex_ = 0;
};

} // namespace synaptics
