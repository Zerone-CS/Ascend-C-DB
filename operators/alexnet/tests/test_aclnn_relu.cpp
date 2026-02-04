#include <iostream>
#include <cstring>
#include <acl/acl.h>
#include "aclnnop/aclnn_relu.h"

#define CHECK_RET(cond, msg) \
    do { if (!(cond)) { std::cerr << msg << std::endl; return -1; } } while(0)

int main() {
    std::cout << "=== AlexNet ReLU - True NPU Execution ==="  << std::endl;
    
    // Init
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, "aclInit failed");
    ret = aclrtSetDevice(0);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSetDevice failed");
    
    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtCreateStream failed");
    
    std::cout << "[NPU] Device initialized" << std::endl;
    
    // Data
    const int64_t shape[] = {4, 8};
    const int ndim = 2;
    const int64_t size = 4 * 8;
    const int64_t byteSize = size * sizeof(float);
    
    float hostInput[size];
    float hostOutput[size];
    
    // Initialize input: [-2, -1, 0, 1, 2, 3, 4, 5, ...]
    for (int i = 0; i < size; i++) {
        hostInput[i] = (float)(i - size/2);
    }
    
    std::cout << "Input[0:8]: ";
    for (int i = 0; i < 8; i++) std::cout << hostInput[i] << " ";
    std::cout << std::endl;
    
    // Allocate device memory
    void *devInput, *devOutput;
    ret = aclrtMalloc(&devInput, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "malloc input failed");
    ret = aclrtMalloc(&devOutput, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "malloc output failed");
    
    // H2D
    ret = aclrtMemcpy(devInput, byteSize, hostInput, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, "H2D memcpy failed");
    std::cout << "[NPU] H2D transfer done" << std::endl;
    
    // Create tensor descriptors
    aclTensor *inputTensor = nullptr, *outputTensor = nullptr;
    int64_t strides[] = {8, 1};
    
    inputTensor = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, ndim, devInput);
    outputTensor = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, ndim, devOutput);
    CHECK_RET(inputTensor && outputTensor, "aclCreateTensor failed");
    
    // Get workspace size
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnReluGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnReluGetWorkspaceSize failed");
    std::cout << "[NPU] Workspace size: " << workspaceSize << " bytes" << std::endl;
    
    // Allocate workspace
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, "malloc workspace failed");
    }
    
    // Execute ReLU on NPU!
    std::cout << "[NPU] Executing ReLU on NPU..." << std::endl;
    ret = aclnnRelu(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnRelu failed");
    
    // Synchronize
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed");
    std::cout << "[NPU] Kernel execution completed" << std::endl;
    
    // D2H
    ret = aclrtMemcpy(hostOutput, byteSize, devOutput, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, "D2H memcpy failed");
    std::cout << "[NPU] D2H transfer done" << std::endl;
    
    // Verify
    std::cout << "Output[0:8]: ";
    for (int i = 0; i < 8; i++) std::cout << hostOutput[i] << " ";
    std::cout << std::endl;
    
    bool correct = true;
    for (int i = 0; i < size; i++) {
        float expected = hostInput[i] > 0 ? hostInput[i] : 0;
        if (std::abs(hostOutput[i] - expected) > 1e-5) {
            correct = false;
            std::cout << "Mismatch at " << i << ": expected " << expected << ", got " << hostOutput[i] << std::endl;
        }
    }
    
    if (correct) {
        std::cout << "\n*** ReLU NPU Execution SUCCESS! ***" << std::endl;
    } else {
        std::cout << "\n*** VERIFICATION FAILED ***" << std::endl;
    }
    
    // Cleanup
    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);
    if (workspace) aclrtFree(workspace);
    aclrtFree(devInput);
    aclrtFree(devOutput);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    return correct ? 0 : 1;
}
