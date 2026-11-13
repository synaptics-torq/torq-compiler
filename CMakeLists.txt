cmake_minimum_required(VERSION 3.21)

project(TORQ ASM C CXX)

option(TORQ_OS_RTOS "Build the TORQ runtime for RTOS instead of Linux" OFF)
if(TORQ_OS_RTOS)
  set(TORQ_OS_LINUX OFF)
else()
  set(TORQ_OS_LINUX ON)
endif()

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
set(IREE_INPUT_STABLEHLO ON)
if(TORQ_OS_RTOS)
  set(IREE_ALLOCATOR_SYSTEM "rtos" CACHE STRING "IREE system allocator" FORCE)
  set(IREE_ALLOCATOR_SYSTEM_CTL "rtos_allocator_system_ctl" CACHE STRING "IREE system allocator callback" FORCE)
  set(IREE_ALLOCATOR_RTOS_CTL rtos_allocator_system_ctl)
  set(IREE_ALLOCATOR_RTOS_SRCS "${CMAKE_CURRENT_LIST_DIR}/runtime/driver/rtos_allocator.c")
  set(IREE_ALLOCATOR_RTOS_DEPS
    "$<BUILD_LOCAL_INTERFACE:zephyr_interface>"
    "$<BUILD_LOCAL_INTERFACE:zephyr_generated_headers>"
  )
endif()
add_subdirectory(third_party/iree)

if(TORQ_OS_RTOS)
  add_dependencies(iree_base_base.objects zephyr_generated_headers)
endif()

option(TORQ_BUILD_CLI "Build the Torq engine MobileNetV2 Linux sample" OFF)
if(TORQ_BUILD_CLI)
  add_subdirectory(runtime/app)
endif()

add_custom_target(torq
  DEPENDS torq_rt_cm iree-compile torq-compile torq-mlir-lsp-server iree-run-module torq-run-module iree-c-embed-data iree-flatcc-cli iree-opt third_party/iree/compiler/bindings/python/all third_party/iree/runtime/bindings/python/all
  COMMENT "Building Torq project with IREE"
)
