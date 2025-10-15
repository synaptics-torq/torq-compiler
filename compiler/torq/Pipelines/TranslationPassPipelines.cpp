// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TranslationPassPipelines.h"

#include "torq/Codegen/Passes.h"
#include "torq/Conversions/LinalgToTorqHL/Passes.h"
#include "torq/Conversions/TosaToTorqHL/Passes.h"
#include "torq/Transforms/Linalg/Passes.h"
#include "torq/Transforms/TorqHL/Passes.h"

#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-torq-pass-pipeline"

namespace mlir::syna::torq {

/// Populates passes needed to lower linalg/arith/math ops to TORQ ops via
/// the structured ops path.
void buildTORQCodegenPassPipeline(OpPassManager &variantPassManager) {

    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();

    modulePassManager.addPass(createTORQLowerExecutableTargetPass());

    LLVM_DEBUG({
        llvm::dbgs() << "Using TORQ pass pipeline:\n";
        variantPassManager.printAsTextualPipeline(llvm::dbgs());
        llvm::dbgs() << "\n";
    });
}

} // namespace mlir::syna::torq
