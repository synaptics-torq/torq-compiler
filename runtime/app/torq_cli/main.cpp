// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/classifier.hpp"
#include "torq/network.hpp"
#include "torq/preprocessor.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: torq_cli MODEL.vmfb INPUT.bin [OUTPUT_SCALE OUTPUT_ZERO_POINT]\n";
        return EXIT_SUCCESS;
    }
    if (argc != 3 && argc != 5) {
        std::cerr << "Usage: torq_cli MODEL.vmfb INPUT.bin [OUTPUT_SCALE OUTPUT_ZERO_POINT]\n";
        return EXIT_FAILURE;
    }

    try {
        torq::Network network;
        if (!network.load_model(std::string(argv[1]))) {
            throw std::runtime_error("Unable to load VMFB model and reflected tensor metadata");
        }
        if (network.inputs.size() != 1 || network.outputs.size() != 1) {
            throw std::runtime_error("MBV2 requires one input and one output tensor");
        }

        float output_scale = torq::kDefaultScale;
        int32_t output_zero_point = network.outputs[0].data_type() == torq::DataType::uint8
            ? 0 : torq::kDefaultZeroPoint;
        if (argc == 5) {
            size_t parsed = 0;
            output_scale = std::stof(argv[3], &parsed);
            if (parsed != std::string(argv[3]).size() || !std::isfinite(output_scale) || output_scale <= 0) {
                throw std::runtime_error("OUTPUT_SCALE must be a finite positive number");
            }
            output_zero_point = std::stoi(argv[4], &parsed);
            if (parsed != std::string(argv[4]).size()) {
                throw std::runtime_error("OUTPUT_ZERO_POINT must be an integer");
            }
        }

        std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
        if (!input || input.tellg() != static_cast<std::streamoff>(network.inputs[0].size())) {
            throw std::runtime_error("Input file must match the model input tensor size (" +
                                     std::to_string(network.inputs[0].size()) + " bytes)");
        }
        std::vector<char> input_data(network.inputs[0].size());
        torq::Preprocessor preprocessor;
        input.seekg(0);
        if (!input.read(input_data.data(), static_cast<std::streamsize>(input_data.size())) ||
            !preprocessor.assign(network.inputs[0], torq::InputData(input_data.data(), input_data.size()))) {
            throw std::runtime_error("Unable to read or assign raw input tensor");
        }

        const auto start = std::chrono::steady_clock::now();
        if (!network.predict()) {
            throw std::runtime_error("Network prediction failed");
        }
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start);
        torq::Classifier classifier(5, output_scale, output_zero_point);
        const auto result = classifier.process(network.outputs);
        if (!result.success) {
            throw std::runtime_error("Classification failed");
        }

        std::cout << "Inference: " << std::fixed << std::setprecision(3)
                  << elapsed.count() << " ms\n";
        for (const auto& item : result.items) {
            std::cout << "class_index=" << item.class_index
                      << " confidence=" << item.confidence << '\n';
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "torq_cli: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}