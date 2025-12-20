cmake_minimum_required(VERSION 3.21)

project(TORQ ASM C CXX)

option(TORQ_HW_DEBUG "Enable verbose debugging logs of HW" OFF)

option(TORQ_ENABLE_SOC_FPGA "Enable SoC FPGA support" OFF)
option(TORQ_ENABLE_ASTRA_MACHINA "Enable Astra Machina platform support" OFF)
option(TORQ_ENABLE_AWS_FPGA "Enable AWS FPGA support" OFF)
option(TORQ_ENABLE_SIMULATOR "Enable Simulator support" ON)
option(TORQ_ENABLE_CORALNPU_SIMULATOR "Enable CoralNPU CSS Simulator support" OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TORQ_HW_DIR ${CMAKE_CURRENT_LIST_DIR}/third_party/torq-hw)
add_subdirectory(third_party/torq-hw/ct)
add_subdirectory(third_party/torq-hw/cm)
add_subdirectory(third_party/torq-hw/rt)

# we don't checkout all submodules of iree as we don't use them
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF)
# we want to have assertions even if we are building for Release or RelWithDebInfo
set(IREE_ENABLE_ASSERTIONS ON)
# add the cmake for our IREE plugin
set(IREE_CMAKE_PLUGIN_PATHS ${CMAKE_CURRENT_LIST_DIR})
# disable IREE compiler backend plugins by default
set(IREE_TARGET_BACKEND_DEFAULTS OFF)
# enable IREE llvm-cpu backed plugin that is used to compile code for CSS
set(IREE_TARGET_BACKEND_LLVM_CPU ON)
# disable the default runtime HAL drivers
set(IREE_HAL_DRIVER_DEFAULTS OFF)
# enable our hal driver in the runtime
set(IREE_EXTERNAL_HAL_DRIVERS torq)
# disable the input pipeline for stableHLO which is not supported at the moment
set(IREE_INPUT_STABLEHLO OFF)
add_subdirectory(third_party/iree)

add_custom_target(torq
  DEPENDS torq_rt_cm iree-compile torq-compile iree-run-module iree-opt third_party/iree/compiler/bindings/python/all third_party/iree/runtime/bindings/python/all
  COMMENT "Building Torq project with IREE"
)
