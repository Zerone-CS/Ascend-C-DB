cmake_minimum_required(VERSION 3.16)
project(alexnet_ascendc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN 环境
if(NOT DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_TOOLKIT_HOME "/usr/local/Ascend/ascend-toolkit/latest")
else()
    set(ASCEND_TOOLKIT_HOME $ENV{ASCEND_TOOLKIT_HOME})
endif()

set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_TOOLKIT_HOME})
set(SOC_VERSION "Ascend910B2" CACHE STRING "SoC Version")

message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "SOC_VERSION: ${SOC_VERSION}")

# 包含路径
include_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
)

# 链接路径
link_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/opp/vendors/customize/op_api/lib
)

# ========== 测试目标 ==========

# ReLU aclnn 测试
add_executable(test_relu_aclnn tests/test_relu_aclnn.cpp)
target_link_libraries(test_relu_aclnn ascendcl nnopbase opapi cust_opapi)

# AlexNet 完整测试
add_executable(alexnet_full examples/cpp/alexnet_full_npu.cpp)
target_link_libraries(alexnet_full ascendcl nnopbase opapi cust_opapi)
