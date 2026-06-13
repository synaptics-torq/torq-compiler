// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace mlir::iree_compiler;

// Provided by host_x86_objects.S — the archive bytes are embedded in the
// compiler binary's .rodata via .incbin so the link step is fully
// self-contained (no external LFS / search-path dependency at runtime).
extern "C" {
extern const char _binary_libclang_rt_builtins_x86_64_a_start[];
extern const char _binary_libclang_rt_builtins_x86_64_a_end[];
}

namespace mlir::syna::torq {

namespace {

bool variantTargetsHostX86_64(IREE::HAL::ExecutableVariantOp variantOp) {
    // IREE stock LLVMCPU backend is "llvm-cpu"; Torq's internal host outline
    // uses "llvm-host". Both can produce x86_64 ELF that needs the archive.
    auto backend = variantOp.getTarget().getBackend();
    if (backend != "llvm-cpu" && backend != "llvm-host")
        return false;
    auto configAttr = variantOp.getTarget().getConfiguration();
    if (!configAttr)
        return false;
    auto tripleAttr = configAttr.getAs<StringAttr>("target_triple");
    if (!tripleAttr)
        return false;
    return llvm::Triple(tripleAttr.getValue()).getArch() == llvm::Triple::x86_64;
}

// Returns an absolute filesystem path to a process-local copy of the embedded
// archive. The bytes are written once on first call (to the OS temp dir);
// the path is reused for the lifetime of the process. IREE's link step opens
// the file like any other archive — no
// `--iree-hal-executable-object-search-path` flag needed.
const std::string &getEmbeddedArchivePath() {
    static const std::string path = [] {
        llvm::SmallString<128> model;
        llvm::sys::path::system_temp_directory(/*ErasedOnReboot=*/true, model);
        llvm::sys::path::append(model, "torq_host_x86_builtins-%%%%%%.a");
        llvm::SmallString<128> tmp;
        if (auto createEc = llvm::sys::fs::createUniqueFile(model, tmp)) {
            llvm::report_fatal_error("AttachX86CpuObjectsPass: failed to create temp archive file");
        }
        std::error_code ec;
        llvm::raw_fd_ostream os(tmp, ec, llvm::sys::fs::OF_None);
        if (ec) {
            llvm::report_fatal_error("AttachX86CpuObjectsPass: failed to open temp archive file");
        }
        const size_t size = static_cast<size_t>(
            _binary_libclang_rt_builtins_x86_64_a_end - _binary_libclang_rt_builtins_x86_64_a_start
        );
        os.write(_binary_libclang_rt_builtins_x86_64_a_start, size);
        os.close();

        // IREE's link stage reopens this file by path later in the same
        // process, so it must outlive the pass — we remove it when the process
        // exits, not here. FileRemover unlinks on a normal exit;
        // RemoveFileOnSignal covers SIGINT/SIGTERM/crash. This leaves at most
        // one temp file per process, so a long test run no longer fills
        // $TMPDIR.
        static llvm::FileRemover remover(tmp.c_str());
        llvm::sys::RemoveFileOnSignal(tmp);
        return tmp.str().str();
    }();
    return path;
}

class AttachX86CpuObjectsPass : public impl::AttachX86CpuObjectsBase<AttachX86CpuObjectsPass> {
  public:
    using AttachX86CpuObjectsBase<AttachX86CpuObjectsPass>::AttachX86CpuObjectsBase;

    void runOnOperation() override {
        auto *ctx = &getContext();
        auto pathAttr = StringAttr::get(ctx, getEmbeddedArchivePath());
        auto newObjectAttr = IREE::HAL::ExecutableObjectAttr::get(ctx, pathAttr, /*data=*/nullptr);

        getOperation().walk([&](IREE::HAL::ExecutableVariantOp variantOp) {
            if (!variantTargetsHostX86_64(variantOp))
                return;

            // Append to existing objects (if any) so we don't overwrite
            // archives attached by other sources. Skip if our archive is
            // already attached — preprocess-executables-with may fire more
            // than once per variant in some pipelines.
            SmallVector<Attribute> objects;
            if (auto existing = variantOp.getObjectsAttr()) {
                if (llvm::is_contained(existing.getValue(), newObjectAttr))
                    return;
                objects.append(existing.begin(), existing.end());
            }
            objects.push_back(newObjectAttr);
            variantOp.setObjectsAttr(ArrayAttr::get(ctx, objects));
        });
    }
};

} // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>> createAttachX86CpuObjectsPass() {
    return std::make_unique<AttachX86CpuObjectsPass>();
}

} // namespace mlir::syna::torq
