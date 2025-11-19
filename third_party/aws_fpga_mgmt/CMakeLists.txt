include(ExternalProject)

ExternalProject_Add(
    aws_fpga_mgmt_proj
    URL https://github.com/aws/aws-fpga/archive/refs/tags/v2.2.2.tar.gz
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cd <SOURCE_DIR>/sdk/userspace && ./mkall_fpga_mgmt_tools.sh
    
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1

    BUILD_BYPRODUCTS
    <SOURCE_DIR>/sdk/userspace/lib/libfpga_pci.a
    <SOURCE_DIR>/sdk/userspace/lib/libfpga_dma.a
    <SOURCE_DIR>/sdk/userspace/lib/libfpga_clkgen.a
    <SOURCE_DIR>/sdk/userspace/lib/libfpga_mgmt.a
)

ExternalProject_Get_Property(aws_fpga_mgmt_proj SOURCE_DIR)
set(AWS_FPGA_MGMT_LIBS libfpga_pci libfpga_dma libfpga_clkgen libfpga_mgmt)

set(AWS_FPGA_INC_DIR ${SOURCE_DIR}/sdk/userspace/include)

# we need to create the directory at configure time as the imported targets need to
# have the include directories set at that time
file(MAKE_DIRECTORY ${AWS_FPGA_INC_DIR})

foreach(LIB_NAME ${AWS_FPGA_MGMT_LIBS})

    add_library(${LIB_NAME} STATIC IMPORTED)

    add_dependencies(${LIB_NAME} aws_fpga_mgmt_proj)

    set_target_properties(${LIB_NAME} PROPERTIES
        IMPORTED_LOCATION ${SOURCE_DIR}/sdk/userspace/lib/${LIB_NAME}.a
        INTERFACE_INCLUDE_DIRECTORIES ${AWS_FPGA_INC_DIR}
    )

endforeach()

add_library(aws_fpga_mgmt STATIC dummy.c)

target_include_directories(aws_fpga_mgmt INTERFACE ${AWS_FPGA_INC_DIR})

target_link_libraries(aws_fpga_mgmt INTERFACE ${AWS_FPGA_MGMT_LIBS})

iree_install_targets(TARGETS aws_fpga_mgmt HDRS ${AWS_FPGA_INC_DIR}
                     EXPORT_SET Runtime COMPONENT IREEDevLibraries-Runtime FIX_INCLUDE_DIRS)
