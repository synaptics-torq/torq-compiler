set(sources
  ConstantFold.cpp
  Passes.cpp
  Patterns.cpp
  PatternUtils.cpp
  LinalgTilePass.cpp
  UnrollLoopPass.cpp
  ArithToTorqHLPattern.cpp
  LinalgToTorqHLPrePatternLowPrio.cpp
  LinalgToTorqHLPrePattern.cpp
  TensorToTorqOrLinalgPattern.cpp
)

set(deps
    ::PassesIncGen
    MLIRArithDialect
    MLIRTensorDialect
    MLIRFunctionInterfaces
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    MLIRLinalgTransforms
    MLIRTensorTransforms
    iree::compiler::Dialect::Util::IR
    iree::target::torq::Dialect::TorqHL::Dialect
    MLIRExecutionEngine
)

if(TORQ_ENABLE_TORQ_GENERIC)
  list(APPEND sources
    ActPatterns.cpp
    AluPatterns.cpp
    LinalgToTorqGeneric.cpp
  )

  list(APPEND deps ::LinalgToTorqGenericIncGen)
  
  iree_tablegen_library(
    NAME
      LinalgToTorqGenericIncGen
    TD_FILE
      LinalgToTorqGeneric.td
    OUTS
      --gen-pass-decls LinalgToTorqGeneric.h.inc
  )
endif()

iree_tablegen_library(
  NAME
    PassesIncGen
  TD_FILE
    Passes.td
  OUTS
    --gen-pass-decls Passes.h.inc
)

iree_cc_library(
  NAME
    LinalgToTorqHL
  HDRS
    PassesDetail.h
    Passes.h
    Passes.h.inc
    Patterns.h
    OpPatternOptions.h
  SRCS
    ${sources}
  DEPS
    ${deps}
  PUBLIC
)
