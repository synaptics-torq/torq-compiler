
iree_cc_binary(
  NAME
    torq-run-module
  SRCS
    "torq-run-module-main.cpp"
  DEPS
    iree::base
    iree::base::internal::flags
    iree::hal
    iree::tooling::context_util
    iree::tooling::run_module
    iree::vm
  INSTALL_COMPONENT IREETools-Runtime
)
