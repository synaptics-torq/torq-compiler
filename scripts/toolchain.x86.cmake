set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES IREE_BUILD_DIR)

set(CMAKE_C_COMPILER ${IREE_BUILD_DIR}/third_party/iree/llvm-project/bin/clang)
set(CMAKE_CXX_COMPILER ${IREE_BUILD_DIR}/third_party/iree/llvm-project/bin/clang++)
set(CMAKE_LINKER ${IREE_BUILD_DIR}/third_party/iree/llvm-project/bin/ld.lld)

set(CMAKE_C_FLAGS "--target=x86_64-unknown-elf -ffreestanding -nostdlib" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "--target=x86_64-unknown-elf -ffreestanding -nostdlib" CACHE STRING "" FORCE)
set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "" )
set(CMAKE_EXE_LINKER_FLAGS   "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld -nostartfiles -nodefaultlibs " )

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
