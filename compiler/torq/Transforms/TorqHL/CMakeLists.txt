set(sources
    FormDispatchRegionsPass.cpp
    Passes.cpp
    TorqHLOptimizeSegmentationPass.cpp
)

set(deps
    ::PassesIncGen
    LLVMSupport
    MLIRAffineDialect
    MLIRAffineToStandard
    MLIRAffineTransforms
    MLIRAffineUtils
    MLIRArithTransforms
    MLIRBufferizationDialect
    MLIRDestinationStyleOpInterface
    MLIRFuncDialect
    MLIRFunctionInterfaces
    MLIRIR
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFToControlFlow
    MLIRSCFTransforms
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransformUtils
    MLIRTransforms
    iree::compiler::Dialect::Util::Analysis
    iree::compiler::Dialect::Util::Conversion
    iree::compiler::Dialect::Util::Conversion::MemRefToUtil
    iree::compiler::Dialect::Util::IR
    iree::compiler::Dialect::Util::Transforms
    iree::compiler::Utils
    iree::target::torq::Conversions::TorqHLToTorqHW
    iree::target::torq::Conversions::TosaToTorqHL
    iree::target::torq::Conversions::LinalgToTorqHL
)

if(TORQ_ENABLE_TORQ_GENERIC)
  list(APPEND sources
    SimplifyGenericPass.cpp
    MergeBiasScaleTensorsPass.cpp
  )
  iree_tablegen_library(
    NAME
      GenericOpPassesIncGen
    TD_FILE
      GenericOpPasses.td
    OUTS
      --gen-pass-decls GenericOpPasses.h.inc
  )
  list(APPEND deps ::GenericOpPassesIncGen)
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
    Transforms
  HDRS
    "Passes.h"
  SRCS
    ${sources}
  DEPS
    ${deps}
  PUBLIC
)

