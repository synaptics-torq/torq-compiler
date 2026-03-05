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
    INIT,
    INIT_PREPARE_OUTPUT_DIRS,
    INIT_COMPUTE_XRAM_FOOTPRINT,
    // Backend-specific runtime/session initialization (TorqHw::open()).
    INIT_HW_OPEN,
    INIT_CLEAR_MEMORY,
    // Upload executable code segments into device-accessible memory.
    INIT_LOAD_CODE_SEGMENTS,
    INIT_LOAD_HOST_CODE,
    // 
    DISPATCH,
    DISPATCH_SYNC_BINDINGS_IN,
    DISPATCH_SYNC_BINDINGS_OUT,
    DISPATCH_ACQUIRE_HW_RESOURCES,
    DISPATCH_RELEASE_HW_RESOURCES,
    DISPATCH_EXECUTE_ACTIONS,
    HAL_CALL,
    HAL_QUEUE_WAIT,
    HAL_QUEUE_APPLY_DEFERRED,
    HAL_QUEUE_SIGNAL,
    HAL_COMMAND_BUFFER_COPY_BUFFER,
    HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET,
    HOST_COPY,
    NSS_START,
    NSS_WAIT,
    ALLOC,
    DEALLOC,
    HOST_START,
    HOST_WAIT
};

inline std::string eventTypeToString(EventType type) {
    switch (type) {
        case EventType::INIT:
            return "INIT";
        case EventType::INIT_PREPARE_OUTPUT_DIRS:
            return "INIT_PREPARE_OUTPUT_DIRS";
        case EventType::INIT_COMPUTE_XRAM_FOOTPRINT:
            return "INIT_COMPUTE_XRAM_FOOTPRINT";
        case EventType::INIT_HW_OPEN:
            return "INIT_HW_OPEN";
        case EventType::INIT_CLEAR_MEMORY:
            return "INIT_CLEAR_MEMORY";
        case EventType::INIT_LOAD_CODE_SEGMENTS:
            return "INIT_LOAD_CODE_SEGMENTS";
        case EventType::INIT_LOAD_HOST_CODE:
            return "INIT_LOAD_HOST_CODE";
        case EventType::DISPATCH:
            return "DISPATCH";
        case EventType::DISPATCH_SYNC_BINDINGS_IN:
            return "DISPATCH_SYNC_BINDINGS_IN";
        case EventType::DISPATCH_SYNC_BINDINGS_OUT:
            return "DISPATCH_SYNC_BINDINGS_OUT";
        case EventType::DISPATCH_ACQUIRE_HW_RESOURCES:
            return "DISPATCH_ACQUIRE_HW_RESOURCES";
        case EventType::DISPATCH_RELEASE_HW_RESOURCES:
            return "DISPATCH_RELEASE_HW_RESOURCES";
        case EventType::DISPATCH_EXECUTE_ACTIONS:
            return "DISPATCH_EXECUTE_ACTIONS";
        case EventType::HAL_CALL:
            return "HAL_CALL";
        case EventType::HAL_QUEUE_WAIT:
            return "HAL_QUEUE_WAIT";
        case EventType::HAL_QUEUE_APPLY_DEFERRED:
            return "HAL_QUEUE_APPLY_DEFERRED";
        case EventType::HAL_QUEUE_SIGNAL:
            return "HAL_QUEUE_SIGNAL";
        case EventType::HAL_COMMAND_BUFFER_COPY_BUFFER:
            return "HAL_COMMAND_BUFFER_COPY_BUFFER";
        case EventType::HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET:
            return "HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET";
        case EventType::HOST_COPY:
            return "HOST_COPY";
        case EventType::NSS_START:
            return "NSS_START";
        case EventType::NSS_WAIT:
            return "NSS_WAIT";
        case EventType::ALLOC:
            return "ALLOC";
        case EventType::DEALLOC:
            return "DEALLOC";
        case EventType::HOST_START:
            return "HOST_START";
        case EventType::HOST_WAIT:
            return "HOST_WAIT";
    }
    assert(false && "unknown event type");
}

struct Event {
    std::chrono::steady_clock::time_point timestamp;
    EventType type;
    enum TimeTag { BEGIN, END};
    TimeTag action;
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
        EventType eventType // = EventType::DISPATCH
    );

    void close();

    void addEvent(EventType type, Event::TimeTag action, int actionIndex);

    ~TorqDispatchEventLog();

    TorqDispatchEventLog(const TorqDispatchEventLog&) = delete;
    TorqDispatchEventLog(TorqDispatchEventLog&&) = delete;
    TorqDispatchEventLog& operator=(const TorqDispatchEventLog&) = delete;
    TorqDispatchEventLog& operator=(TorqDispatchEventLog&&) = delete;

private:    
    TorqEventLog &log_;
    DispatchEvents *events_;
    EventType eventType_;
    bool closed_{false};
};

class TorqEventLog {
public:
    ~TorqEventLog() {
        close();
    }

    TorqDispatchEventLog* startDispatch(
        const std::string& dispatchName,
        EventType eventType
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

#define TORQ_ADD_PROFILING_EVENT_BEGIN(event_log, event)   \
  if (event_log) {                                         \
      event_log->addEvent(event, Event::BEGIN, -1);        \
  }

#define TORQ_ADD_PROFILING_EVENT_END(event_log, event)     \
  if (event_log) {                                         \
      event_log->addEvent(event, Event::END, -1);          \
  }

#define TORQ_PROFILE_EVENT(event_log, event, status, expr) \
  TORQ_ADD_PROFILING_EVENT_BEGIN(event_log, event);        \
  (status) = (expr);                                       \
  TORQ_ADD_PROFILING_EVENT_END(event_log, event);          \

} // namespace synaptics
