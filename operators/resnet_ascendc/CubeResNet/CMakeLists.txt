cmake_minimum_required(VERSION 3.15)
project(cube_resnet)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
else()
  set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(CMAKE_CCEC_COMPILER ccec)
set(ACT_PATH "${ASCEND_HOME_PATH}/aarch64-linux/ascendc/act")

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CCEC_COMPILER_OPTIONS
    -O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220
    --cce-aicore-block-local-init
    --cce-auto-sync
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -DL2_CACHE_HINT

    -I${ASCEND_HOME_PATH}/compiler/tikcpp
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    -I${ASCEND_HOME_PATH}/include
    -I${ASCEND_HOME_PATH}/include/experiment
    -I${ASCEND_HOME_PATH}/include/experiment/runtime
    -I${ASCEND_HOME_PATH}/include/experiment/msprof

    -I${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api
    -I${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api
    -I${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api/impl
    -I${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api/impl
    -I${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api/interface

    -I${ACT_PATH}
    -I${CMAKE_SOURCE_DIR}

    -L${ASCEND_HOME_PATH}/lib64
    -Wno-macro-redefined -Wno-ignored-attributes
    -lruntime -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase
)

message("ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}")
message("ACT_PATH: ${ACT_PATH}")

add_custom_target(cube_resnet)

function(add_ccec_executable NAME)
    add_custom_command(
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}
        COMMAND ${CMAKE_CCEC_COMPILER} ${CCEC_COMPILER_OPTIONS} ${ARGN} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}
        DEPENDS ${ARGN}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling Cube kernel: ${NAME}"
    )
    add_custom_target(${NAME} ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME})
    add_dependencies(cube_resnet ${NAME})
endfunction()

# Test Cube MatMul
add_ccec_executable(test_cube_matmul test_cube_matmul.cpp)

# Test Cube ResNet
add_ccec_executable(test_cube_resnet test_cube_resnet.cpp)

# Full ResNet-18
add_ccec_executable(test_resnet18 test_resnet18.cpp)

# Benchmark
add_ccec_executable(benchmark_matmul benchmark_matmul.cpp)
