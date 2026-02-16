// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TorqEventLog.h"

#include "iree/base/internal/flags.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
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

void TorqEventLog::flushUnlocked() {
    while(!dispatchEvents_.empty()) {
        auto events = dispatchEvents_.front();

        for (const auto &e : events->events) {

            auto timestamp_us =
                std::chrono::duration_cast<std::chrono::microseconds>(e.timestamp.time_since_epoch())
                    .count();

            std::string type_str = eventTypeToString(e.type);

            profilingFile_ << events->dispatchName << "," << events->dispatchIndex << "," << e.actionIndex << "," << timestamp_us << "," << type_str << "\n";

        }

        delete events;

        dispatchEvents_.erase(dispatchEvents_.begin());
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

TorqDispatchEventLog::TorqDispatchEventLog(TorqEventLog& log, std::string dispatchName, size_t dispatchIndex) : log_(log) {
    events_ = new DispatchEvents{dispatchName, dispatchIndex, {}};
    events_->events.push_back({
        std::chrono::steady_clock::now(),
        EventType::DISPATCH_BEGIN,
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
        EventType::DISPATCH_END,
        -1
    });

    log_.addDispatchEvents(events_);

}

void TorqDispatchEventLog::addEvent(EventType type, int actionIndex) {
    events_->events.push_back({std::chrono::steady_clock::now(), type, actionIndex});
}

TorqDispatchEventLog::~TorqDispatchEventLog() {
    close();
}

TorqDispatchEventLog* TorqEventLog::startDispatch(const std::string& dispatchName) {
    std::lock_guard<std::mutex> lock(events_mutex_);
    return new TorqDispatchEventLog(*this, dispatchName, nextDispatchIndex_++);
}



} // namespace synaptics
