// Copyright 2020 The IREE Authors
// Copyright 2026 Synaptics Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is the main entry point for the torq-run-module tool, it was copied
// from iree/tools/iree-run-module.c and modified to fit the Torq project needs.

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/run_module.h"
#include "iree/vm/api.h"

#include <string>
#include <vector>

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  // FIXME: quick hack to change defaults until create a proper tool
  std::vector<std::string> allArgs = { argv[0],
                                      "--device=torq" };
  for (int i = 1; i < argc; ++i) {
      allArgs.push_back(argv[i]);
  }

  std::vector<char*> argPtrs;
  argPtrs.reserve(allArgs.size());
  for (auto& arg : allArgs) {
      argPtrs.push_back(const_cast<char*>(arg.c_str()));
  }

  // Parse command line flags.
  iree_flags_set_usage(
      "torq-run-module",
      "Runs a function within a compiled IREE module and handles I/O parsing\n"
      "and optional expected value verification/output processing. Modules\n"
      "can be provided by file path (`--module=file.vmfb`) or read from stdin\n"
      "(`--module=-`) and the function to execute matches the original name\n"
      "provided to the compiler (`--function=foo` for `func.func @foo`).\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  // Hosting applications can provide their own allocators to pool resources or
  // track allocation statistics related to IREE code.
  iree_allocator_t host_allocator = iree_allocator_system();
  // Hosting applications should reuse instances across multiple contexts that
  // have similar composition (similar types/modules/etc). Most applications can
  // get by with a single shared instance.
  iree_vm_instance_t* instance = NULL;
  iree_status_t status =
      iree_tooling_create_instance(host_allocator, &instance);

  // Utility to run the module with the command line flags. This particular
  // method is only useful in these IREE tools that want consistent flags -
  // a real application will need to do what this is doing with its own setup
  // and I/O handling.
  int exit_code = EXIT_SUCCESS;
  if (iree_status_is_ok(status)) {
    status = iree_tooling_run_module_from_flags(instance, host_allocator,
                                                &exit_code);
  }

  iree_vm_instance_release(instance);

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
