
iree_cc_binary(
  NAME
    torq-run-module
  SRCS
    "torq-run-module-main.cpp"
  DEPS
    iree::base
    iree::base::internal::flags
    iree::hal
    iree::io::file_handle
    iree::tooling::context_util
    iree::tooling::run_module
    iree::schemas::bytecode_module_def_c_fbs
    iree::schemas::instruments::dispatch_def_c_fbs
    iree::vm
  INSTALL_COMPONENT IREETools-Runtime
)
