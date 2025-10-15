# compile all bootstrap code for the css kernel for riscv_32 and then create
# host object files containing the corresponding binary data

set(TORQ_RISCV_SOURCES
    css_kernel_kelvin.c
    css_kernel_qemu.c
    css_kernel.s
)

foreach(TORQ_RISCV_SOURCE ${TORQ_RISCV_SOURCES})

    set(TORQ_RISCV_OBJ "${CMAKE_CURRENT_BINARY_DIR}/${TORQ_RISCV_SOURCE}.o.data")

    # set the default options used both for .c and .s files
    set(TORQ_CLANG_OPTIONS -c -target riscv32-unknown-elf -march=rv32im -mabi=ilp32 -Werror -Winline -g)

    # if we are compiling a .c file we need to add a few more options
    if (TORQ_RISCV_SOURCE MATCHES ".c")
    
        set(TORQ_CLANG_OPTIONS ${TORQ_CLANG_OPTIONS}
                -I${TORQ_HW_DIR}  -I${TORQ_HW_DIR}/reg 
                -I${CMAKE_SOURCE_DIR}/third_party/iree/runtime/src
                -ffreestanding -fno-stack-protector -fno-builtin -nostdlib -O2)

        # add the stubs directory that contains fake header files that provide some base
        # functionality that is not available in a freestanding environment
        set(TORQ_CLANG_OPTIONS ${TORQ_CLANG_OPTIONS} -I${CMAKE_CURRENT_SOURCE_DIR}/stubs )
    endif()

    # add a rule to compile the source file to an object file
    add_custom_command(
        OUTPUT ${TORQ_RISCV_OBJ}
        COMMAND "${IREE_CLANG_BINARY}" ${TORQ_CLANG_OPTIONS}
                        -o ${TORQ_RISCV_OBJ} ${CMAKE_CURRENT_SOURCE_DIR}/${TORQ_RISCV_SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${TORQ_RISCV_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/css_kernel.c.inc
    )

    list(APPEND TORQ_RISCV_OBJS ${TORQ_RISCV_OBJ})

endforeach()

# create dependencies for the pre-built runtime libraries
set(TORQ_RISCV_PREBUILT_RUNTIME_LIBS libc.a libclang_rt.builtins-riscv32.a libm.a)

foreach(TORQ_RISCV_PREBUILT_RUNTIME_LIB ${TORQ_RISCV_PREBUILT_RUNTIME_LIBS})
    set(TORQ_RISCV_PREBUILT_RUNTIME_LIB_PATH "${CMAKE_SOURCE_DIR}/third_party/runtimes-prebuilt/${TORQ_RISCV_PREBUILT_RUNTIME_LIB}")
    list(APPEND TORQ_RISCV_OBJS ${TORQ_RISCV_PREBUILT_RUNTIME_LIB_PATH})
endforeach()

set_source_files_properties(css_kernel_riscv.S PROPERTIES OBJECT_DEPENDS "${TORQ_RISCV_OBJS}")

add_library(torq_kernel_riscv STATIC css_kernel_riscv.S)
target_compile_definitions(torq_kernel_riscv PRIVATE CSS_PREBUILT_DIR="${CMAKE_SOURCE_DIR}/third_party/runtimes-prebuilt")

# we can't use the macro iree_bitcode_library here because it uses the wrong machine config for riscv_32
target_include_directories(torq_kernel_riscv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# this is used to find the generated .o included as binary blobs in the .s file
target_include_directories(torq_kernel_riscv PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_sources(torq_kernel_riscv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/css_kernel_riscv.h)
