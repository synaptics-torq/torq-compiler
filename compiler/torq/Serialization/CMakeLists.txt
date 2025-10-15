# Note: that doesn´t work for some obscure EXPORT problem
#target_link_libraries(torq_Serialization PRIVATE torq_ct)
# so we need to add the include directories and the library manually instead

iree_cc_library(
  NAME
    Serialization
  HDRS
    Serialization.h
  SRCS
    Serialization.cpp
    DescGen.cpp
  INCLUDES
    ${TORQ_CT_INC_DIR}
  DEPS
    MLIRIR
    MLIRSupport
    MLIRTranslateLib
    LLVMSupport
    MLIRArithDialect
    MLIRArithUtils
    MLIRFuncDialect
    MLIRFunctionInterfaces
    iree::compiler::Dialect::HAL::Target
    iree::base::internal::flatcc::building
    schemas::torq_executable_def_c_fbs
    iree::target::torq::Dialect::TorqHW::Dialect
    ${TORQ_CT_LIB}
  COPTS
    -Wno-multichar
)
