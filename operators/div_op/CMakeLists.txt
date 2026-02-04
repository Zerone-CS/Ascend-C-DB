cmake_minimum_required(VERSION 3.16)
project(div_op)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_TOOLKIT_HOME "/usr/local/Ascend/ascend-toolkit/latest")
else()
    set(ASCEND_TOOLKIT_HOME $ENV{ASCEND_TOOLKIT_HOME})
endif()

set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_TOOLKIT_HOME})
message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")

include_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/opp/vendors/customize/op_api/include
    ${CMAKE_CURRENT_SOURCE_DIR}/DivCustom/build_out/autogen
)

link_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/opp/vendors/customize/op_api/lib
)

add_executable(div_npu_aclnn main_aclnn.cpp)
target_link_libraries(div_npu_aclnn ascendcl nnopbase cust_opapi)
