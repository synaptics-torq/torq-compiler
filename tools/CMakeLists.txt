iree_cc_binary(
    NAME
        torq-mlir-lsp-server
    SRCS
        "torq-mlir-lsp-server.cc"
    DEPS
        MLIRIR
        MLIRLspServerLib
        MLIRSupport
        iree::compiler::Tools::init_passes_and_dialects
        iree::compiler::bindings::c::headers
        iree::target::torq::Utils
        iree::target::torq::Dialect::TorqHL::Dialect
        iree::target::torq::Dialect::TorqHW::Dialect
        MLIRTosaDialect
    SETUP_INSTALL_RPATH
)

iree_cc_binary(
    NAME
      torq-compile
    SRCS
      "torq-compile-main.cc"
    DEPS
      iree::compiler::bindings::c::headers
      iree::compiler::API::Impl
    DATA
      ${IREE_LLD_TARGET}
    HOSTONLY
    SETUP_INSTALL_RPATH
    INSTALL_COMPONENT IREETools-Compiler
  )
  