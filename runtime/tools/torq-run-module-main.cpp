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
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/io/file_contents.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/run_module.h"
#include "iree/vm/bytecode/archive.h"
#include "iree/vm/api.h"

#include "torq_executable_def_reader.h"
#include "torq_executable_def_verifier.h"

#include <string>
#include <fstream>
#include <vector>

#include "iree/schemas/bytecode_module_def_reader.h"
#include "iree/schemas/bytecode_module_def_verifier.h"

IREE_FLAG(string, dump_dispatches, "",
          "Writes a JSON summary of Torq dispatch segments to the specified"
          " file and exits without running the model.");

static std::string iree_tooling_find_module_path(int argc, char** argv) {
  std::string module_path;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--module" && i + 1 < argc) {
      module_path = argv[i + 1];
    } else if (arg.rfind("--module=", 0) == 0) {
      module_path = arg.substr(sizeof("--module=") - 1);
    }
  }
  size_t at_pos = module_path.find('@');
  if (at_pos != std::string::npos) module_path.resize(at_pos);
  size_t query_pos = module_path.find('?');
  if (query_pos != std::string::npos) module_path.resize(query_pos);
  return module_path;
}

static void iree_tooling_write_json_string(std::ostream& stream,
                                           const std::string& value) {
  stream << '"';
  for (unsigned char c : value) {
    switch (c) {
      case '"':
        stream << "\\\"";
        break;
      case '\\':
        stream << "\\\\";
        break;
      case '\b':
        stream << "\\b";
        break;
      case '\f':
        stream << "\\f";
        break;
      case '\n':
        stream << "\\n";
        break;
      case '\r':
        stream << "\\r";
        break;
      case '\t':
        stream << "\\t";
        break;
      default:
        if (c < 0x20) {
          static const char kHex[] = "0123456789abcdef";
          stream << "\\u00" << kHex[(c >> 4) & 0xF] << kHex[c & 0xF];
        } else {
          stream << static_cast<char>(c);
        }
        break;
    }
  }
  stream << '"';
}

static iree_status_t iree_tooling_dump_dispatches(
    const std::string& module_path, const std::string& output_path,
    iree_allocator_t host_allocator) {
  if (module_path.empty()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--module= must be specified when using "
                            "--dump-dispatches");
  }

  iree_io_file_contents_t* file_contents = NULL;
  if (module_path == "-") {
    IREE_RETURN_IF_ERROR(
        iree_io_file_contents_read_stdin(host_allocator, &file_contents));
  } else {
    IREE_RETURN_IF_ERROR(iree_io_file_contents_map(
        iree_make_cstring_view(module_path.c_str()), IREE_IO_FILE_ACCESS_READ,
        host_allocator, &file_contents));
  }

  iree_const_byte_span_t flatbuffer_contents = iree_const_byte_span_empty();
  iree_host_size_t rodata_offset = 0;
  iree_status_t status = iree_vm_bytecode_archive_parse_header(
      file_contents->const_buffer, &flatbuffer_contents, &rodata_offset);
  if (iree_status_is_ok(status)) {
    int verify_ret = iree_vm_BytecodeModuleDef_verify_as_root(
        flatbuffer_contents.data, flatbuffer_contents.data_length);
    if (verify_ret != flatcc_verify_ok) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "FlatBuffer verification failed: %s",
                                flatcc_verify_error_string(verify_ret));
    }
  }

  if (iree_status_is_ok(status)) {
    iree_vm_BytecodeModuleDef_table_t module_def =
        iree_vm_BytecodeModuleDef_as_root(flatbuffer_contents.data);
    if (!module_def) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "failed to parse BytecodeModuleDef");
    } else {
      std::ofstream output(output_path.c_str(), std::ios::out | std::ios::trunc);
      if (!output.is_open()) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "failed to open file %s",
                                  output_path.c_str());
      } else {
        output << "{\n";
        output << "  \"module\": ";
        iree_tooling_write_json_string(
            output, std::string(iree_vm_BytecodeModuleDef_name(module_def)));
        output << ",\n";
        output << "  \"dispatches\": [\n";

        bool wrote_dispatch = false;
        iree_vm_RodataSegmentDef_vec_t rodata_segments =
            iree_vm_BytecodeModuleDef_rodata_segments(module_def);
        for (size_t i = 0; i < iree_vm_RodataSegmentDef_vec_len(rodata_segments);
             ++i) {
          iree_vm_RodataSegmentDef_table_t segment =
              iree_vm_RodataSegmentDef_vec_at(rodata_segments, i);
          iree_const_byte_span_t dispatch_contents = iree_const_byte_span_empty();
          
          if (iree_vm_RodataSegmentDef_embedded_data_is_present(segment)) {
            flatbuffers_uint8_vec_t data =
                iree_vm_RodataSegmentDef_embedded_data(segment);
            dispatch_contents = iree_make_const_byte_span(
                (const uint8_t*)data, flatbuffers_uint8_vec_len(data));
          } else if (iree_vm_RodataSegmentDef_external_data_length_is_present(
                         segment)) {
            const uint64_t offset =
                iree_vm_RodataSegmentDef_external_data_offset(segment);
            const uint64_t length =
                iree_vm_RodataSegmentDef_external_data_length(segment);
            dispatch_contents = iree_make_const_byte_span(
                file_contents->const_buffer.data + rodata_offset + offset,
                (size_t)length);
          }

          if (!dispatch_contents.data || dispatch_contents.data_length == 0) {
            continue;
          }

          // check the magic for ExecutableDef before trying to verify it as that, to avoid spurious
          // verification failures for non-dispatch segments.

          if (dispatch_contents.data_length < 8) {
            continue;
          }

          if (memcmp(dispatch_contents.data + 4, iree_hal_torq_ExecutableDef_identifier,
                     4) != 0) {
            continue;
          }

          int verify_ret = iree_hal_torq_ExecutableDef_verify_as_root(
              dispatch_contents.data, dispatch_contents.data_length);
          if (verify_ret != flatcc_verify_ok) {
            printf("WARNING: Segment %zu looks like a Torq ExecutableDef but failed verification: %s (code=%d)\n",
                   i, flatcc_verify_error_string(verify_ret), verify_ret);
            continue;
          }

          iree_hal_torq_ExecutableDef_table_t executable_def =
              iree_hal_torq_ExecutableDef_as_root(dispatch_contents.data);
          if (!executable_def) {
            continue;
          }

          if (wrote_dispatch) {
            output << ",\n";
          }
          wrote_dispatch = true;

          output << "    {\n";
          output << "      \"name\": ";
          iree_tooling_write_json_string(
              output, std::string(iree_hal_torq_ExecutableDef_executable_name(
                          executable_def)));
          output << ",\n";
          output << "      \"segments\": [\n";

          bool wrote_segment = false;
          iree_hal_torq_Segment_vec_t segments =
              iree_hal_torq_ExecutableDef_code_get(executable_def);
          for (size_t j = 0; j < iree_hal_torq_Segment_vec_len(segments); ++j) {
            iree_hal_torq_Segment_table_t torq_segment =
                iree_hal_torq_Segment_vec_at(segments, j);
            if (wrote_segment) {
              output << ",\n";
            }
            wrote_segment = true;
            output << "        {\"index\": " << j << ", \"initialized\": ";
            output << (iree_hal_torq_Segment_data_is_present(torq_segment) ? "true" : "false");
            output << ", \"size\": "
                   << iree_hal_torq_Segment_size(torq_segment);
            output << ", \"address\": "
                   << iree_hal_torq_Segment_xram_address(torq_segment);
            output << "}";
          }

          output << "\n      ]\n";
             output << "      ,\n";
             output << "      \"bindings\": [\n";

             bool wrote_binding = false;
             iree_hal_torq_Binding_vec_t bindings =
            iree_hal_torq_ExecutableDef_bindings_get(executable_def);
             for (size_t j = 0; j < iree_hal_torq_Binding_vec_len(bindings); ++j) {
               iree_hal_torq_Binding_table_t binding =
              iree_hal_torq_Binding_vec_at(bindings, j);
               if (wrote_binding) {
            output << ",\n";
               }
               wrote_binding = true;
               output << "        {\"index\": " << j;
               output << ", \"id\": " << iree_hal_torq_Binding_id_get(binding);
               output << ", \"address\": "
                 << iree_hal_torq_Binding_address_get(binding);
               output << ", \"offset\": "
                 << iree_hal_torq_Binding_offset_get(binding);
               output << ", \"size\": " << iree_hal_torq_Binding_size_get(binding);
               output << ", \"is_read_only\": "
                 << (iree_hal_torq_Binding_is_read_only_get(binding) ? "true"
                             : "false");
               output << ", \"is_write_only\": "
                 << (iree_hal_torq_Binding_is_write_only_get(binding) ? "true"
                              : "false");
               output << "}";
             }

             output << "\n      ]\n";
          output << "    }";
        }

        output << "\n  ]\n";
        output << "}\n";

        if (!output.good()) {
          status = iree_make_status(IREE_STATUS_INTERNAL,
                                    "failed while writing %s",
                                    output_path.c_str());
        }
      }
    }
  }

  iree_io_file_contents_free(file_contents);
  return status;
}

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
  int newArgc = static_cast<int>(argPtrs.size());
  char** newArgv = argPtrs.data();

  // Parse command line flags.
  iree_flags_set_usage(
      "torq-run-module",
      "Runs a function within a compiled IREE module and handles I/O parsing\n"
      "and optional expected value verification/output processing. Modules\n"
      "can be provided by file path (`--module=file.vmfb`) or read from stdin\n"
      "(`--module=-`) and the function to execute matches the original name\n"
      "provided to the compiler (`--function=foo` for `func.func @foo`).\n"
      "Use `--dump-dispatches=/path/to/file.json` to write Torq dispatch\n"
      "segment metadata instead of running the model.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &newArgc, &newArgv);

  std::string module_path = iree_tooling_find_module_path(argc, argv);
  if (FLAG_dump_dispatches[0]) {
    iree_allocator_t host_allocator = iree_allocator_system();
    iree_status_t status = iree_tooling_dump_dispatches(
        module_path, std::string(FLAG_dump_dispatches), host_allocator);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      IREE_TRACE_ZONE_END(z0);
      IREE_TRACE_APP_EXIT(EXIT_FAILURE);
      return EXIT_FAILURE;
    }
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(EXIT_SUCCESS);
    return EXIT_SUCCESS;
  }

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
