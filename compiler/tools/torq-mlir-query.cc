// Copyright 2025 Synaptics Inc.

#include "iree/compiler/Tools/init_dialects.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Query/Matcher/Registry.h"
#include "mlir/Query/Matcher/SliceMatchers.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-query/MlirQueryMain.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHW/TorqHWDialect.h"

#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

using namespace mlir;

// This is needed because these matchers are defined as overloaded functions.
using HasOpAttrName = detail::AttrOpMatcher(StringRef);
using HasOpName = detail::NameOpMatcher(StringRef);
using IsConstantOp = detail::constant_op_matcher();

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

    query::matcher::Registry matcherRegistry;

    // Matchers registered in alphabetical order for consistency:
    matcherRegistry.registerMatcher("allOf", query::matcher::internal::allOf);
    matcherRegistry.registerMatcher("anyOf", query::matcher::internal::anyOf);
    matcherRegistry.registerMatcher(
        "getAllDefinitions",
        query::matcher::m_GetAllDefinitions<query::matcher::DynMatcher>);
    matcherRegistry.registerMatcher(
        "getDefinitions",
        query::matcher::m_GetDefinitions<query::matcher::DynMatcher>);
    matcherRegistry.registerMatcher(
        "getDefinitionsByPredicate",
        query::matcher::m_GetDefinitionsByPredicate<query::matcher::DynMatcher,
                                                    query::matcher::DynMatcher>);
    matcherRegistry.registerMatcher(
        "getUsersByPredicate",
        query::matcher::m_GetUsersByPredicate<query::matcher::DynMatcher,
                                              query::matcher::DynMatcher>);
    matcherRegistry.registerMatcher("hasOpAttrName",
                                    static_cast<HasOpAttrName *>(m_Attr));
    matcherRegistry.registerMatcher("hasOpName", static_cast<HasOpName *>(m_Op));
    matcherRegistry.registerMatcher("isConstantOp",
                                    static_cast<IsConstantOp *>(m_Constant));
    matcherRegistry.registerMatcher("isNegInfFloat", m_NegInfFloat);
    matcherRegistry.registerMatcher("isNegZeroFloat", m_NegZeroFloat);
    matcherRegistry.registerMatcher("isNonZero", m_NonZero);
    matcherRegistry.registerMatcher("isOne", m_One);
    matcherRegistry.registerMatcher("isOneFloat", m_OneFloat);
    matcherRegistry.registerMatcher("isPosInfFloat", m_PosInfFloat);
    matcherRegistry.registerMatcher("isPosZeroFloat", m_PosZeroFloat);
    matcherRegistry.registerMatcher("isZero", m_Zero);
    matcherRegistry.registerMatcher("isZeroFloat", m_AnyZeroFloat);

    MLIRContext context(registry);

    return failed(mlirQueryMain(argc, argv, context, matcherRegistry));
}
