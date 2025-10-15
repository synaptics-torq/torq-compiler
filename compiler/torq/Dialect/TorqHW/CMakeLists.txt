iree_add_all_subdirs()

iree_cc_library(
  NAME
    Dialect
  HDRS
    "TorqHWAttrs.h"
    "TorqHWDialect.h"
    "TorqHWOps.h"
  TEXTUAL_HDRS
    "TorqHWAttrs.cpp.inc"
    "TorqHWAttrs.h.inc"
    "TorqHWDialect.cpp.inc"
    "TorqHWDialect.h.inc"
    "TorqHWEnums.cpp.inc"
    "TorqHWEnums.h.inc"
    "TorqHWOps.cpp.inc"
    "TorqHWOps.h.inc"
  SRCS
    "TorqHWAttrs.cpp"
    "TorqHWDialect.cpp"
    "TorqHWOps.cpp"
  DEPS
    ::TorqHWDialectGen
    ::TorqHWOpsGen
    ::TorqHWAttrDefGen
    LLVMSupport
    MLIRControlFlowInterfaces
    MLIRIR
    MLIRLinalgDialect
    MLIRParser
    MLIRSCFDialect
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTensorDialect
    MLIRVectorDialect
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRFunctionInterfaces
  PUBLIC
)

iree_tablegen_library(
  NAME
    TorqHWDialectGen
  TD_FILE
    "TorqHWBase.td"
  OUTS
    --gen-dialect-decls TorqHWDialect.h.inc
    --gen-dialect-defs TorqHWDialect.cpp.inc
)

iree_tablegen_library(
  NAME
    TorqHWOpsGen
  TD_FILE
    "TorqHWOps.td"
  OUTS
    --gen-op-decls TorqHWOps.h.inc
    --gen-op-defs TorqHWOps.cpp.inc
)

iree_tablegen_library(
  NAME
    TorqHWAttrDefGen
  TD_FILE
    "TorqHWAttrs.td"
  OUTS
    --gen-attrdef-decls TorqHWAttrs.h.inc
    --gen-attrdef-defs TorqHWAttrs.cpp.inc
    --gen-enum-decls TorqHWEnums.h.inc
    --gen-enum-defs TorqHWEnums.cpp.inc
)
