cmake_minimum_required(VERSION 3.16)
project(softmax_op)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_TOOLKIT_HOME "/usr/local/Ascend/ascend-toolkit/latest")
else()
    set(ASCEND_TOOLKIT_HOME $ENV{ASCEND_TOOLKIT_HOME})
endif()

set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_TOOLKIT_HOME})
set(SOC_VERSION "Ascend910B2" CACHE STRING "SoC Version")

message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "SOC_VERSION: ${SOC_VERSION}")

set(ASCENDC_INC_PATH ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
set(TIKCPP_PATH ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp)
set(HCC_PATH /usr/local/Ascend/ascend-toolkit/8.3.RC1/toolkit/toolchain/hcc)
set(CCEC ${ASCEND_CANN_PACKAGE_PATH}/compiler/ccec_compiler/bin/ccec)

set(KERNEL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/softmax_custom.cpp)
set(KERNEL_BIN ${CMAKE_CURRENT_BINARY_DIR}/softmax_custom_kernel.o)

# Compile kernel to binary
add_custom_command(
    OUTPUT ${KERNEL_BIN}
    COMMAND ${CCEC} -xcce -O2 --npu-soc=${SOC_VERSION} --cce-aicore-only -std=c++17
        -I${HCC_PATH}/aarch64-target-linux-gnu/include/c++/7.3.0
        -I${HCC_PATH}/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu
        -I${TIKCPP_PATH}/tikcfw
        -I${TIKCPP_PATH}/tikcfw/impl
        -I${ASCENDC_INC_PATH}/basic_api
        -I${ASCENDC_INC_PATH}/basic_api/impl
        -I${ASCENDC_INC_PATH}/basic_api/interface
        -I${ASCENDC_INC_PATH}/highlevel_api/lib
        -I${ASCEND_CANN_PACKAGE_PATH}/include
        -c ${KERNEL_SOURCE} -o ${KERNEL_BIN}
    DEPENDS ${KERNEL_SOURCE}
    COMMENT "Compiling kernel softmax_custom.cpp"
)

add_custom_target(kernel_compile ALL DEPENDS ${KERNEL_BIN})

# Compile main program with GCC
include_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/include
)
link_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

add_executable(softmax_npu main.cpp)
add_dependencies(softmax_npu kernel_compile)
target_link_libraries(softmax_npu ascendcl runtime)
