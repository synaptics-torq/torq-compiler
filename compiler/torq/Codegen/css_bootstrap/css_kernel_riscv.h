#pragma once

#include <stddef.h>
#include <stdint.h>

#include <optional>
#include <string>
#include <vector>

namespace mlir::syna::torq {

struct CssObject {
    char *start;
    char *end;

    std::string as_string() const { return std::string(start, end); }
};

struct CssConfig {
    std::string name;
    std::string mabi;
    std::string march;
    std::string mattrs;
    uint32_t itcmStart;
    uint32_t dtcmStart;
    uint32_t regsStart;
    CssObject kernel;
    CssObject bootstrap;
    CssObject linkerScript;
    CssObject compiler_rt;
    CssObject libc;
    CssObject libm;
};

std::optional<CssConfig> getCssConfigByName(const std::string &name);

} // namespace mlir::syna::torq