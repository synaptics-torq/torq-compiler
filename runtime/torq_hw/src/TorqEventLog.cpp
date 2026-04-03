// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TorqEventLog.h"

#include "iree/base/internal/flags.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>

using namespace std;

namespace synaptics {

IREE_FLAG(string, torq_profile_host, "", "Create host profiling log in the specified file");

TorqEventLog& TorqEventLog::get() {
    static TorqEventLog instance;
    
    if (TorqEventLog::isProfilingEnabled() && !instance.profilingFile_.is_open()) {
        instance.open(FLAG_torq_profile_host);
    }

    return instance;
}

bool TorqEventLog::isProfilingEnabled() {
    return FLAG_torq_profile_host[0] != '\0';
}

bool TorqEventLog::open(std::string profilingFile) {
    std::lock_guard<std::mutex> lock(events_mutex_);
    profilingFile_.open(profilingFile);

    if (!profilingFile_.is_open()) {
        cerr << "Failed to create profiling file: " << profilingFile << endl;
        return false;
    }

    profilingFile_ << "dispatch_name,invocation_id,action_id,timestamp_us,event\n";

    return true;
}

/// Do a K-way merge to order events globally based on timestamps.
/// Assuming `Event` vectors are already monotonic, this is cheaper than a
/// global stable sort.
void TorqEventLog::flushUnlocked() {
    struct DispatchCursor {
        DispatchEvents* dispatch;
        size_t dispatchOrder;
        size_t eventIndex;
    };

    auto cursorCompare = [](const DispatchCursor& lhs, const DispatchCursor& rhs) {
        const Event& lhsEvent = lhs.dispatch->events[lhs.eventIndex];
        const Event& rhsEvent = rhs.dispatch->events[rhs.eventIndex];
        if (lhsEvent.timestamp != rhsEvent.timestamp) {
            // priority_queue is max-heap by default, invert to get earliest first.
            return lhsEvent.timestamp > rhsEvent.timestamp;
        }
        // Keep deterministic order for same timestamp rows.
        if (lhs.dispatchOrder != rhs.dispatchOrder) {
            return lhs.dispatchOrder > rhs.dispatchOrder;
        }
        return lhs.eventIndex > rhs.eventIndex;
    };

    std::priority_queue<DispatchCursor, std::vector<DispatchCursor>, decltype(cursorCompare)> mergeQueue(cursorCompare);

    size_t dispatchOrder = 0;
    for (auto* dispatch : dispatchEvents_) {
        if (!dispatch->events.empty()) {
            mergeQueue.push(DispatchCursor{dispatch, dispatchOrder, 0});
        }
        ++dispatchOrder;
    }

    while (!mergeQueue.empty()) {
        DispatchCursor cursor = mergeQueue.top();
        mergeQueue.pop();

        const Event& event = cursor.dispatch->events[cursor.eventIndex];
        auto timestamp_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                event.timestamp.time_since_epoch())
                .count();

        profilingFile_ << cursor.dispatch->dispatchName << ","
                      << cursor.dispatch->dispatchIndex << ","
                      << event.actionIndex << "," << timestamp_us << ","
                      << eventTypeToString(event.type)
                      << (event.action == Event::BEGIN ? "_BEGIN" : "_END") << "\n";

        ++cursor.eventIndex;
        if (cursor.eventIndex < cursor.dispatch->events.size()) {
            mergeQueue.push(cursor);
        }
    }

    while (!dispatchEvents_.empty()) {
        delete dispatchEvents_.front();
        dispatchEvents_.pop_front();
    }

    profilingFile_.flush();
}

void TorqEventLog::flush() {
    std::lock_guard<std::mutex> lock(events_mutex_);
    flushUnlocked();
}

void TorqEventLog::close() {
    std::lock_guard<std::mutex> lock(events_mutex_);

    if (profilingFile_.is_open()) {
        flushUnlocked();
        profilingFile_.close();
    }
}

TorqDispatchEventLog::TorqDispatchEventLog(
    TorqEventLog& log, std::string dispatchName, size_t dispatchIndex,
    EventType eventType
)
    : log_(log), eventType_(eventType) {
    events_ = new DispatchEvents{dispatchName, dispatchIndex, {}};
    events_->events.push_back({
        std::chrono::steady_clock::now(),
        eventType_,
        Event::BEGIN,
        -1
    });
}

void TorqDispatchEventLog::close() {
    if (closed_) {
        return;
    }

    closed_ = true;

    events_->events.push_back({
        std::chrono::steady_clock::now(),
        eventType_,
        Event::END,
        -1
    });

    log_.addDispatchEvents(events_);

}

void TorqDispatchEventLog::addEvent(EventType type, Event::TimeTag action, int actionIndex) {
    events_->events.push_back({std::chrono::steady_clock::now(), type, action, actionIndex});
}

TorqDispatchEventLog::~TorqDispatchEventLog() {
    close();
}

TorqDispatchEventLog* TorqEventLog::startDispatch(
    const std::string& dispatchName,
    EventType eventType
) {
    std::lock_guard<std::mutex> lock(events_mutex_);
    return new TorqDispatchEventLog(*this, dispatchName, nextDispatchIndex_++, eventType);
}



} // namespace synaptics
