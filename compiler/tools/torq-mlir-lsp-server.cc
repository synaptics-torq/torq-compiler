// Copyright 2025 Synaptics Inc.

#include "iree/compiler/Tools/init_dialects.h"
#include "iree/compiler/tool_entry_points_api.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHW/TorqHWDialect.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"


#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"


int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    
    mlir::iree_compiler::registerAllDialects(registry);
    
    registry.insert<mlir::syna::torq_hl::TorqHLDialect>();
    registry.insert<mlir::syna::torq_hw::TorqHWDialect>();
    registry.insert<mlir::tosa::TosaDialect>();

    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::torch::Torch::TorchDialect>();
    registry.insert<mlir::torch::TorchConversion::TorchConversionDialect>();
    registry.insert<mlir::torch::TMTensor::TMTensorDialect>();

    return failed(mlir::MlirLspServerMain(argc, argv, registry));
}
