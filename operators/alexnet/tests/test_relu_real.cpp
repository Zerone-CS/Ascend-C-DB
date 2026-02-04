#include <iostream>
#include <cstring>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_relu.h"

#define CHECK_RET(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "Error: " << msg << std::endl; \
        std::cerr << "ACL Error: " << aclGetRecentErrMsg() << std::endl; \
        return -1; \
    } \
} while(0)

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "   ReLU on NPU - True NPU Execution" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Initialize ACL
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, "aclInit failed");
    
    ret = aclrtSetDevice(0);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSetDevice failed");
    
    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtCreateStream failed");
    
    std::cout << "[NPU] Device 0 initialized" << std::endl;
    
    // Prepare data
    const int64_t shape[] = {4, 8};
    const int ndim = 2;
    const int64_t numElements = 4 * 8;
    const int64_t byteSize = numElements * sizeof(float);
    
    float* hostInput = new float[numElements];
    float* hostOutput = new float[numElements];
    
    // Initialize: values from -16 to 15
    for (int64_t i = 0; i < numElements; i++) {
        hostInput[i] = static_cast<float>(i - numElements/2);
    }
    
    std::cout << "\nInput (first 8 values): ";
    for (int i = 0; i < 8; i++) std::cout << hostInput[i] << " ";
    std::cout << std::endl;
    
    // Allocate device memory
    void *devInput = nullptr, *devOutput = nullptr;
    ret = aclrtMalloc(&devInput, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "malloc devInput failed");
    
    ret = aclrtMalloc(&devOutput, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "malloc devOutput failed");
    
    // H2D transfer
    ret = aclrtMemcpy(devInput, byteSize, hostInput, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, "H2D memcpy failed");
    std::cout << "[NPU] H2D transfer completed" << std::endl;
    
    // Create aclTensor
    int64_t strides[] = {8, 1};
    aclTensor* inputTensor = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0, 
                                              ACL_FORMAT_ND, shape, ndim, devInput);
    aclTensor* outputTensor = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0, 
                                               ACL_FORMAT_ND, shape, ndim, devOutput);
    CHECK_RET(inputTensor && outputTensor, "aclCreateTensor failed");
    
    // Get workspace size for ReLU
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnReluGetWorkspaceSize(inputTensor, outputTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnReluGetWorkspaceSize failed");
    
    std::cout << "[NPU] Workspace size: " << workspaceSize << " bytes" << std::endl;
    
    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, "malloc workspace failed");
    }
    
    // Execute ReLU on NPU!
    std::cout << "[NPU] Executing ReLU kernel on NPU..." << std::endl;
    ret = aclnnRelu(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnRelu failed");
    
    // Synchronize
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed");
    std::cout << "[NPU] Kernel execution completed" << std::endl;
    
    // D2H transfer
    ret = aclrtMemcpy(hostOutput, byteSize, devOutput, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, "D2H memcpy failed");
    std::cout << "[NPU] D2H transfer completed" << std::endl;
    
    // Print results
    std::cout << "\nOutput (first 8 values): ";
    for (int i = 0; i < 8; i++) std::cout << hostOutput[i] << " ";
    std::cout << std::endl;
    
    // Verify results
    bool correct = true;
    for (int64_t i = 0; i < numElements; i++) {
        float expected = hostInput[i] > 0 ? hostInput[i] : 0;
        if (std::abs(hostOutput[i] - expected) > 1e-5) {
            correct = false;
            std::cerr << "Mismatch at " << i << ": expected " << expected 
                      << ", got " << hostOutput[i] << std::endl;
        }
    }
    
    std::cout << "\n============================================" << std::endl;
    if (correct) {
        std::cout << "*** ReLU NPU Execution SUCCESS! ***" << std::endl;
    } else {
        std::cout << "*** VERIFICATION FAILED ***" << std::endl;
    }
    std::cout << "============================================" << std::endl;
    
    // Cleanup
    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);
    if (workspace) aclrtFree(workspace);
    aclrtFree(devInput);
    aclrtFree(devOutput);
    delete[] hostInput;
    delete[] hostOutput;
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    return correct ? 0 : 1;
}
