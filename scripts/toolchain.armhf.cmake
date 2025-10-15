# This toolchain file cross-compiles for arm-linux-gnueabihf using clang
# The binaries produced are statically linked and can be run on any host without dependecies

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Check if clang is installed
find_program(CLANG_EXECUTABLE clang)

if(NOT CLANG_EXECUTABLE)
  message(FATAL_ERROR "Clang compiler not found. Please install clang (on Ubuntu run the command sudo apt-get install clang).")
endif()

# Check that gcc ld for armhf is installed
find_program(GCC_LD_EXECUTABLE arm-linux-gnueabihf-ld)
if(NOT GCC_LD_EXECUTABLE)
  message(FATAL_ERROR "/usr/bin/arm-linux-gnueabihf-ld not found. Please install gcc for armhf (on Ubuntu run the command sudo apt-get install binutils-arm-linux-gnueabihf).")
endif()

# Check that libc for armhf is installed
if(NOT EXISTS "/usr/arm-linux-gnueabihf/lib/libc.so")
  message(FATAL_ERROR "libc.so not found. Please install libc for arm-linux-gnueabihf (on Ubuntu run the command sudo apt-get install libc6-dev-armhf-cross).")
endif()

# Check that static libstdc++ for armhf is installed
if(NOT EXISTS "/usr/lib/gcc-cross/arm-linux-gnueabihf/13/libstdc++.a")
  message(FATAL_ERROR "libstdc++.a not found. Please install libc for arm-linux-gnueabihf (on Ubuntu run the command sudo apt-get install libstdc++-13-dev-armhf-cross).")
endif()

# Check that libgcc for armhf is installed
if(NOT EXISTS "/usr/arm-linux-gnueabihf/lib/libgcc_s.so.1")
  message(FATAL_ERROR "libgcc_s.so.1 not found. Please install libgcc for arm-linux-gnueabihf (on Ubuntu run the command sudo apt-get install libgcc-s1-armhf-cross).")
endif()

# Check that libgcc_eh for armhf is installed
if(NOT EXISTS "/usr/lib/gcc-cross/arm-linux-gnueabihf/13/libgcc_eh.a")
  message(FATAL_ERROR "libgcc_eh.a not found. Please install libgcc_eh for arm-linux-gnueabihf (on Ubuntu run the command sudo apt-get install libgcc-13-dev-armhf-cross).")
endif()

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_C_FLAGS "--target=arm-linux-gnueabihf -static-libgcc" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "--target=arm-linux-gnueabihf -static-libstdc++ -static-libgcc" CACHE STRING "" FORCE)

#set(CMAKE_SYSROOT /path/to/arm/sysroot)

set(CMAKE_EXE_LINKER_FLAGS "--target=arm-linux-gnueabihf --sysroot=${CMAKE_SYSROOT} -static -static-libstdc++ -static-libgcc" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" CACHE STRING "" FORCE)

set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)