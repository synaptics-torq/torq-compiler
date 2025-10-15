// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.
///
/// Duration measurement.
///

#pragma once

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace synaptics {

/// Measure time duration.
/// Timer is automatically started when object is created unless explicitely disabled.
class Timer {
    typedef std::chrono::steady_clock Clock;

public:
    typedef int64_t Duration;
    typedef std::chrono::microseconds DurationUnit;

    /// Create timer and start time measurement.
    Timer(bool auto_start = true)
    {
        if (auto_start)
            start();
    }

    /// Start time measurement.
    /// A timer can be (re)started multiple times if needed.
    void start() { _start = Clock::now(); }

    /// Get elapsed time since last start.
    /// @return duration in microseconds
    Duration get() const
    {
        auto end = Clock::now();
        return std::chrono::duration_cast<DurationUnit>(end - _start).count();
    }

private:
    Clock::time_point _start;
};


/// Print timer in ms
inline std::ostream& operator<<(std::ostream& out, const Timer& tmr)
{
    float t = tmr.get() / 1.e3;
    out << std::fixed << std::setprecision(2) << t << " ms";
    return out;
}

}  // namespace synaptics
