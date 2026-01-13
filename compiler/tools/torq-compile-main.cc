// Copyright 2022 The IREE Authors
// Copyright 2025 Synopsys Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/tool_entry_points_api.h"
#include <vector>
#include <string>

int main(int argc, char **argv) {
    
    // FIXME: quick hack to change defaults until create a proper tool
    std::vector<std::string> allArgs = { argv[0],
                                         "--iree-flow-enable-pad-handling",
                                         "--iree-flow-inline-constants-max-byte-length=100000000",
                                         "--iree-hal-target-backends=torq" };

    // FIXME: this is the current default but it is wrong because we should use auto-detection
    allArgs.push_back("--iree-input-type=tosa-torq");

    for (int i = 1; i < argc; ++i) {
        allArgs.push_back(argv[i]);
    }

    std::vector<char*> argPtrs;
    argPtrs.reserve(allArgs.size());
    for (auto& arg : allArgs) {
        argPtrs.push_back(const_cast<char*>(arg.c_str()));
    }

    return ireeCompilerRunMain(static_cast<int>(argPtrs.size()), argPtrs.data());
}
