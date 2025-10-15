set(headers
  TilingInterfaces.h
  TorqHLAttrs.h
  TorqHLDialect.h
  TorqHLOps.h
  EncodingRequirements.h
)

set(sources
  BufferizationInterfaces.cpp
  TorqHLAttrs.cpp
  TorqHLDialect.cpp
  TorqHLOps.cpp
  EncodingRequirements.cpp
)

set(deps
    ::TorqHLDialectGen
    ::TorqHLOpsGen
    ::TorqHLAttrDefGen
    LLVMSupport
    iree::compiler::Dialect::HAL::IR::HALDialect
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
)

if(TORQ_ENABLE_TORQ_GENERIC)
  message(STATUS "************************************* Enabling TorqHL GenericOp support")
  iree_tablegen_library(
    NAME
      GenericOpGen
    TD_FILE
      "GenericOp.td"
    OUTS
      --gen-op-decls GenericOp.h.inc
      --gen-op-defs GenericOp.cpp.inc
  )

  list(APPEND deps ::GenericOpGen)

  list(APPEND headers
    GenericOpBufferizableInterface.h
    GenericOp.h
    )
  list(APPEND sources
    GenericOp.cpp
    GenericOpBufferizableInterface.cpp
    TilingInterfaces.cpp
  )
  list(APPEND defines ENABLE_TORQ_GENERIC)
endif()

iree_cc_library(
  NAME
    Dialect
  DEFINES
    ${defines}
  HDRS
    ${headers}
  SRCS
    ${sources}
  DEPS
    ${deps}
  PUBLIC
)

iree_tablegen_library(
  NAME
    TorqHLDialectGen
  TD_FILE
    "TorqHLBase.td"
  OUTS
    --gen-dialect-decls TorqHLDialect.h.inc
    --gen-dialect-defs TorqHLDialect.cpp.inc
    --gen-op-interface-decls KernelInterface.h.inc
    --gen-op-interface-defs KernelInterface.cpp.inc
)

iree_tablegen_library(
  NAME
    TorqHLOpsGen
  TD_FILE
    "TorqHLOps.td"
  OUTS
    --gen-op-decls TorqHLOps.h.inc
    --gen-op-defs TorqHLOps.cpp.inc
)

iree_tablegen_library(
  NAME
    TorqHLAttrDefGen
  TD_FILE
    "TorqHLAttrs.td"
  OUTS
    --gen-attrdef-decls TorqHLAttrs.h.inc
    --gen-attrdef-defs TorqHLAttrs.cpp.inc
    --gen-enum-decls TorqHLEnums.h.inc
    --gen-enum-defs TorqHLEnums.cpp.inc
    --gen-typedef-decls TorqHLTypes.h.inc
    --gen-typedef-defs TorqHLTypes.cpp.inc
)
