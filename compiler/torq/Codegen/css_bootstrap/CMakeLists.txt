function(add_binary_obj TORQ_RISCV_SOURCE EXTRA_CLANG_OPTIONS)

    set(TORQ_RISCV_OBJ "${CMAKE_CURRENT_BINARY_DIR}/${TORQ_RISCV_SOURCE}_${CSS_CONFIG_NAME}.o.data")

    # set the default options used both for .c and .s files
    set(TORQ_CLANG_OPTIONS -c -target riscv32-unknown-elf -march=${CSS_CONFIG_MARCH} -mabi=${CSS_CONFIG_MABI} -Werror -Winline -g ${EXTRA_CLANG_OPTIONS})
    
    # add a rule to compile the source file to an object file
    add_custom_command(
        OUTPUT ${TORQ_RISCV_OBJ}
        COMMAND "${IREE_CLANG_BINARY}" ${TORQ_CLANG_OPTIONS}
                        -o ${TORQ_RISCV_OBJ} ${CMAKE_CURRENT_SOURCE_DIR}/${TORQ_RISCV_SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${TORQ_RISCV_SOURCE}
    )

    list(APPEND TORQ_RISCV_OBJS ${TORQ_RISCV_OBJ})

    set(TORQ_RISCV_OBJS ${TORQ_RISCV_OBJS} PARENT_SCOPE)

    string(REPLACE "." "_" SYMBOL_NAME ${TORQ_RISCV_SOURCE})
    set(SYMBOL_NAME "${CSS_CONFIG_NAME}_${SYMBOL_NAME}_o")
    
    set(CSS_BINARY_OBJS "${CSS_BINARY_OBJS}\nBINARY_OBJ css_binary${CSS_BINARY_OBJ_IDX}, ${SYMBOL_NAME}, \".\", \"${TORQ_RISCV_SOURCE}_${CSS_CONFIG_NAME}.o.data\"" PARENT_SCOPE)
    set(CSS_BINARY_OBJ_DEF "${CSS_BINARY_OBJ_DEF}\nBINARY_OBJ(${SYMBOL_NAME})" PARENT_SCOPE)

    math(EXPR CSS_BINARY_OBJ_IDX "${CSS_BINARY_OBJ_IDX} + 1")

    set(CSS_BINARY_OBJ_IDX ${CSS_BINARY_OBJ_IDX} PARENT_SCOPE)    

endfunction()

# find all files in the configs subdirectory that match the pattern *.cmake
file(GLOB TORQ_RISCV_CONFIGS "${CMAKE_CURRENT_SOURCE_DIR}/configs/*.cmake")
file(GLOB TORQ_RISCV_CONFIGS_EXTRA "${CMAKE_SOURCE_DIR}/extras/css_configs/*.cmake")

list(APPEND TORQ_RISCV_CONFIGS ${TORQ_RISCV_CONFIGS_EXTRA})

set(TORQ_RISCV_CONFS_NAMES "")

set(CSS_BINARY_OBJS "")
set(CSS_BINARY_OBJ_DEF "")
set(CSS_CONFIGS_DESCS "")

set(CSS_BINARY_OBJ_IDX 0)

foreach(TORQ_CSS_CONFIG ${TORQ_RISCV_CONFIGS})

    # include the config to set the CSS_CONFIG_* variables
    include(${TORQ_CSS_CONFIG})

    # get the file name form the path
    get_filename_component(CSS_CONFIG_NAME ${TORQ_CSS_CONFIG} NAME_WE)

    list(APPEND TORQ_RISCV_CONFS_NAMES ${CSS_CONFIG_NAME})

    set(C_CODE_OPTIONS ${CSS_CONFIG_CFLAGS}
                -I${TORQ_HW_DIR}  -I${TORQ_HW_DIR}/reg 
                -I${CMAKE_SOURCE_DIR}/third_party/iree/runtime/src
                -ffreestanding -fno-stack-protector -fno-builtin -nostdlib -O2)

    # add the stubs directory that contains fake header files that provide some base
    # functionality that is not available in a freestanding environment
    set(C_CODE_OPTIONS ${C_CODE_OPTIONS} -I${CMAKE_CURRENT_SOURCE_DIR}/stubs )

    add_binary_obj(css_kernel.c "${C_CODE_OPTIONS}")
    
    add_binary_obj(css_kernel.s "")

    list(APPEND CSS_CONFIGS_DESCS "CssConfig{\"${CSS_CONFIG_NAME}\", \"${CSS_CONFIG_MABI}\", \"${CSS_CONFIG_MARCH}\", \"${CSS_CONFIG_MATTRS}\", \
                                  ${CSS_ITCM_START}, ${CSS_DTCM_START}, ${CSS_REGS_START}, \
                                  BINARY_DESC(${CSS_CONFIG_NAME}_css_kernel_c_o), BINARY_DESC(${CSS_CONFIG_NAME}_css_kernel_s_o)}")

endforeach()

string (JOIN ",\n  " CSS_CONFIGS_DESCS_STR ${CSS_CONFIGS_DESCS})

message(STATUS "CSS configurations: ${TORQ_RISCV_CONFS_NAMES}")

configure_file("css_kernel_riscv.S.in" "${CMAKE_CURRENT_BINARY_DIR}/css_kernel_riscv.S" @ONLY)

configure_file("css_kernel_riscv.h.in" "${CMAKE_CURRENT_BINARY_DIR}/css_kernel_riscv.h" @ONLY)

# create dependencies for the pre-built runtime libraries
set(TORQ_RISCV_PREBUILT_RUNTIME_LIBS libc.a libclang_rt.builtins-riscv32.a libm.a)

foreach(TORQ_RISCV_PREBUILT_RUNTIME_LIB ${TORQ_RISCV_PREBUILT_RUNTIME_LIBS})
    set(TORQ_RISCV_PREBUILT_RUNTIME_LIB_PATH "${CMAKE_SOURCE_DIR}/third_party/runtimes-prebuilt/${TORQ_RISCV_PREBUILT_RUNTIME_LIB}")
    list(APPEND TORQ_RISCV_OBJS ${TORQ_RISCV_PREBUILT_RUNTIME_LIB_PATH})
endforeach()

set_source_files_properties(css_kernel_riscv.S PROPERTIES OBJECT_DEPENDS "${TORQ_RISCV_OBJS}")

add_library(torq_kernel_riscv STATIC "${CMAKE_CURRENT_BINARY_DIR}/css_kernel_riscv.S")
target_compile_definitions(torq_kernel_riscv PRIVATE CSS_PREBUILT_DIR="${CMAKE_SOURCE_DIR}/third_party/runtimes-prebuilt")

# we can't use the macro iree_bitcode_library here because it uses the wrong machine config for riscv_32
target_include_directories(torq_kernel_riscv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# this is used to find the generated .o included as binary blobs in the .s file
target_include_directories(torq_kernel_riscv PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_sources(torq_kernel_riscv PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/css_kernel_riscv.h")
