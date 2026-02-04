cmake_minimum_required(VERSION 3.16)
project(resnet18_npu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN Ascend Toolkit path
set(ASCEND_PATH $ENV{ASCEND_INSTALL_PATH})
if(NOT ASCEND_PATH)
    set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

message(STATUS "Using ASCEND_PATH: ${ASCEND_PATH}")

# Include directories
include_directories(
    ${ASCEND_PATH}/include
    ${ASCEND_PATH}/include/aclnn
)

# Library directories
link_directories(
    ${ASCEND_PATH}/lib64
)

# Executable
add_executable(resnet18_npu resnet18_npu.cpp)

# Link libraries
target_link_libraries(resnet18_npu
    ascendcl
    nnopbase
    opapi
    acl_op_compiler
)

# Installation
install(TARGETS resnet18_npu DESTINATION bin)
