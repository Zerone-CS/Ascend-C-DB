cmake_minimum_required(VERSION 3.16)
project(resnet_ascendc_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(ASCEND_PATH /usr/local/Ascend/ascend-toolkit/latest)

include_directories(
    ${ASCEND_PATH}/include
    ${ASCEND_PATH}/include/aclnn
    ${CMAKE_CURRENT_SOURCE_DIR}
)

link_directories(
    ${ASCEND_PATH}/lib64
    ${ASCEND_PATH}/opp/vendors/customize/op_api/lib
)

add_executable(test_basic_block test_basic_block.cpp)
target_link_libraries(test_basic_block
    ascendcl
    nnopbase
    opapi
    cust_opapi
)
