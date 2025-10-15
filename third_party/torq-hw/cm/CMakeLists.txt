set(CMODEL_SRC_DIR "" CACHE PATH "Directory with c-model sources")

if (CMODEL_SRC_DIR STREQUAL "")
    add_library(torq_cm INTERFACE)
    target_include_directories(torq_cm INTERFACE ${CMAKE_CURRENT_LIST_DIR})
    set(torq_cm_lib ${CMAKE_CURRENT_LIST_DIR}/${CMAKE_HOST_SYSTEM_PROCESSOR}-${CMAKE_HOST_SYSTEM_NAME}/libtorq_cm.a)
    target_link_libraries(torq_cm INTERFACE ${torq_cm_lib} m)

    # WORKAROUND: see runtime/torq_hw/CMakeLists.txt
    set(TORQ_SIMULATOR_INC_DIR ${CMAKE_CURRENT_LIST_DIR} PARENT_SCOPE)
    set(TORQ_SIMULATOR_LIB ${torq_cm_lib} m PARENT_SCOPE)
    return()
endif()

set(torq_cm_hdr ${CMODEL_SRC_DIR}/top/inc/torq_cm.h)

if( NOT EXISTS ${torq_cm_hdr} )
    message(FATAL_ERROR "C-model source not found in ${CMODEL_SRC_DIR}")
endif()

message(STATUS "*** Using C-model sources found in ${CMODEL_SRC_DIR} ***")

file(GLOB torq_cm_src
    ${CMODEL_SRC_DIR}/css/src/*.c
    ${CMODEL_SRC_DIR}/nss/src/*.c
    ${CMODEL_SRC_DIR}/top/src/*.c
)
add_library(torq_cm ${torq_cm_src})
target_link_libraries(torq_cm PUBLIC m)

# -w ensure we build the C-model with no warning as errors (it won't compile otherwise)
# -g put debug symbols in C-model so that we can easily debug it
# -O0 disable optimization so that we can easily debug it
# -UNDEBUG disable release build otherwise assertion will be removed and cmodel won't work correctl
target_compile_options(torq_cm PRIVATE -w -g -O0 -UNDEBUG)

target_include_directories(torq_cm
  PUBLIC ${CMODEL_SRC_DIR}/top/inc
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../reg
  PRIVATE ${CMODEL_SRC_DIR}/css/inc
  PRIVATE ${CMODEL_SRC_DIR}/nss/inc
)

# WORKAROUND: see runtime/torq_hw/CMakeLists.txt
set(TORQ_SIMULATOR_INC_DIR ${CMODEL_SRC_DIR}/top/inc PARENT_SCOPE)
set(TORQ_SIMULATOR_LIB ${CMAKE_CURRENT_BINARY_DIR}/libtorq_cm.a m PARENT_SCOPE)