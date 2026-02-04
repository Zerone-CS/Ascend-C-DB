cmake_minimum_required(VERSION 3.16)
project(matmul_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ascend CANN paths
set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest")
set(ASCEND_INCLUDE "${ASCEND_HOME}/aarch64-linux/include")
set(ASCEND_LIB "${ASCEND_HOME}/aarch64-linux/lib64")
set(CUSTOM_OP_PATH "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api")

# Include directories
include_directories(
    ${ASCEND_INCLUDE}
    ${ASCEND_INCLUDE}/aclnn
    ${CUSTOM_OP_PATH}/include
)

# Link directories
link_directories(
    ${ASCEND_LIB}
    ${CUSTOM_OP_PATH}/lib
)

# Test executable
add_executable(test_matmul_aclnn main_aclnn.cpp)

target_link_libraries(test_matmul_aclnn
    ascendcl
    nnopbase
    acl_op_compiler
    cust_opapi
)

# Copy to output
set_target_properties(test_matmul_aclnn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
