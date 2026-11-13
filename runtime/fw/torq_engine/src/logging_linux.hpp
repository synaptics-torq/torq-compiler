// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdlib>
#include <iostream>
#include <sstream>

#define LOGV TORQ_LOG(0)
#define LOGI TORQ_LOG(1)
#define LOGW TORQ_LOG(2)
#define LOGE TORQ_LOG(3)

#define TORQ_LOG(level) \
    torq::Logger::enabled(level) && torq::Logger{level}.out() \
        << __FUNCTION__ << "():" << __LINE__ << ": "

namespace torq {

/// Collect a message and write it to stderr with a TORQ severity prefix.
class Logger {
public:
    /// @param level: severity, from 0 (verbose) to 3 (error)
    explicit Logger(int level) : level_(level) {}

    /// @return stream used to compose the message
    std::ostream &out() { return message_; }

    /// Emit the completed message and flush stderr.
    ~Logger() {
        static constexpr char levels[] = "VIWE";
        std::cerr << levels[level_] << ":TORQ: " << message_.str() << std::endl;
    }

    /// Check TORQ_LOG_LEVEL, defaulting to warnings and errors.
    /// @param level: message severity
    /// @return true if the message meets the configured threshold
    static bool enabled(int level) {
        static const int threshold = [] {
            const char *value = std::getenv("TORQ_LOG_LEVEL");
            return value ? std::atoi(value) : 2;
        }();
        return threshold >= 0 && level >= threshold;
    }

private:
    int level_;
    std::ostringstream message_;
};

}