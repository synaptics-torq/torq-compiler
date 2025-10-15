// Copyright 2024 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <vector>
#include <iostream>
#include <sstream>
#include <variant>
#include <optional>
#include <cassert>
#include <numeric>

#include <iostream>
#include <ostream>
#include <sstream>
#include <memory>

enum TorqLogLevel {
  TORQ_LOG_ERROR = -1,
  TORQ_LOG_NONE = 0,
  TORQ_LOG_DEBUG = 1,
  TORQ_LOG_VERBOSE = 2,
};

#define LOGE TORQ_LOG(TORQ_LOG_ERROR)
#define LOGD TORQ_LOG(TORQ_LOG_DEBUG)
#define LOGV TORQ_LOG(TORQ_LOG_VERBOSE)

#define TORQ_LOG(level) \
  TorqLogger::enabled(level) && TorqLogger{level}.out() << __FUNCTION__ << "():" << __LINE__ << ": "

class TorqLogger {
public:
    TorqLogger(int level) : _level(level) {}
    std::ostream& out() { return _log_message; }

    ~TorqLogger()
    {
        static const char module[] = "Torq";
        const std::string& log_string = _log_message.str();
        std::cout << module << ": " << log_string << std::endl;
    }

    static int request_log_level;

    static bool enabled(int level)
    {
        return level <= request_log_level;
    }

private:
    int _level;
    std::ostringstream _log_message;
 };

struct Cmd {    
    bool isLastCmd;

    virtual void serialize(std::vector<uint8_t>& out) const = 0;
    virtual int serializedSize() const = 0;
    virtual void print() const = 0;
    virtual void loadFromVector(const std::vector<uint8_t>& src) = 0;
    virtual int id() const = 0;

    virtual ~Cmd() = default;
};

struct CfgCmd : public Cmd {    
    uint32_t ra;
    std::vector<uint32_t> values;

    void loadFromVector(const std::vector<uint8_t>& src) override;

    int serializedSize() const override;

    void print() const override;

    virtual int id() const override;

    void serialize(std::vector<uint8_t>& out) const override;
};

struct SynCmd : public Cmd {
    uint32_t msk;
    uint32_t rsvd;
    uint32_t op;

    void loadFromVector(const std::vector<uint8_t>& src) override;

    int serializedSize() const override;

    void print() const override;

    virtual int id() const override;
    
    void serialize(std::vector<uint8_t>& out) const override;
};

struct NxtCmd : public Cmd {
    uint32_t nxt;
    uint32_t rsvd;

    void loadFromVector(const std::vector<uint8_t>& src) override;

    int serializedSize() const override;

    void print() const override;

    virtual int id() const override;

    void serialize(std::vector<uint8_t>& out) const override;
};

using Cmds = std::vector<std::unique_ptr<Cmd>>;

Cmds parseCommandsUntilUnknown(const std::vector<uint8_t>& src);

std::vector<uint8_t> serializeCommands(const Cmds& cmds);

void printCommands(const Cmds& cmds);

