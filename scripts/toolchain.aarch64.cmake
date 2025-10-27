# This toolchain file cross-compiles for aarch64-linux-gnu using clang
# The binaries produced are statically linked and can be run on any host without dependecies

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Check if clang is installed
find_program(CLANG_EXECUTABLE clang)

if(NOT CLANG_EXECUTABLE)
  message(FATAL_ERROR "Clang compiler not found. Please install clang (on Ubuntu run the command sudo apt-get install clang).")
endif()

# Check that gcc ld for aarch64 is installed
find_program(GCC_LD_EXECUTABLE aarch64-linux-gnu-ld)
if(NOT GCC_LD_EXECUTABLE)
  message(FATAL_ERROR "aarch64-linux-gnu-ld not found. Please install gcc for aarch64 (on Ubuntu run the command sudo apt-get install binutils-aarch64-linux-gnu).")
endif()

# Check that libc for aarch64 is installed
if(NOT EXISTS "/usr/aarch64-linux-gnu/lib/libc.so")
  message(FATAL_ERROR "libc.so not found. Please install libc for aarch64 (on Ubuntu run the command sudo apt-get install libc6-dev-arm64-cross).")
endif()

# Check that static libstdc++ for aarch64 is installed
if(NOT EXISTS "/usr/lib/gcc-cross/aarch64-linux-gnu/13/libstdc++.a")
  message(FATAL_ERROR "libstdc++.a not found. Please install libc for aarch64 (on Ubuntu run the command sudo apt-get install libstdc++-13-dev-arm64-cross).")
endif()

# Check that libgcc for aarch64 is installed
if(NOT EXISTS "/usr/aarch64-linux-gnu/lib/libgcc_s.so.1")
  message(FATAL_ERROR " not found. Please install libc for aarch64 (on Ubuntu run the command sudo apt-get install libgcc-s1-arm64-cross).")
endif()

# Check that libgcc_eh for aarch64 is installed
if(NOT EXISTS "/usr/lib/gcc-cross/aarch64-linux-gnu/13/libgcc_eh.a")
  message(FATAL_ERROR "libgcc_eh.a not found. Please install libgcc_eh for arm-linux-gnueabihf (on Ubuntu run the command sudo apt-get install libgcc-13-dev-arm64-cross).")
endif()

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_C_FLAGS "--target=aarch64-linux-gnu -static-libgcc" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "--target=aarch64-linux-gnu -static-libstdc++ -static-libgcc" CACHE STRING "" FORCE)

set(CMAKE_EXE_LINKER_FLAGS "--target=aarch64-linux-gnu --sysroot=${CMAKE_SYSROOT} -static-libstdc++ -static-libgcc" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" CACHE STRING "" FORCE)

set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
