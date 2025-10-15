# Copyright 2024 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    Pipelines
  HDRS
    TranslationPassPipelines.h
    InputConversionPassPipelines.h
  SRCS
    TranslationPassPipelines.cpp
    InputConversionPassPipelines.cpp
  DEPS
    MLIRSupport
    iree::target::torq::Conversions::TosaToTorqHL
    iree::target::torq::Conversions::LinalgToTorqHL
    iree::target::torq::Conversions::TorqHLToTorqHW
    iree::target::torq::Transforms::Linalg::Transforms
    iree::compiler::plugins::input::Torch::InputConversion
    iree::compiler::plugins::input::Torch::torch-mlir::TorchConversionDialectIR
    iree::compiler::plugins::input::Torch::torch-mlir::TorchDialectIR
    iree::compiler::plugins::input::Torch::torch-mlir::TorchDialectPasses
    iree::compiler::plugins::input::Torch::torch-mlir::TorchOnnxToTorchPasses
    iree::compiler::plugins::input::Torch::torch-mlir::ConversionPasses
    iree::compiler::plugins::input::Torch::torch-mlir-dialects::TMTensorDialectIR
)
