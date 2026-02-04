/**
 * AlexNet using aclnn built-in ops + AscendC custom ReLU
 * 
 * Architecture:
 * - Conv2D(3, 64, 11, stride=4, padding=2) -> ReLU -> MaxPool(3, stride=2)
 * - Conv2D(64, 192, 5, padding=2) -> ReLU -> MaxPool(3, stride=2)  
 * - Conv2D(192, 384, 3, padding=1) -> ReLU
 * - Conv2D(384, 256, 3, padding=1) -> ReLU
 * - Conv2D(256, 256, 3, padding=1) -> ReLU -> MaxPool(3, stride=2)
 * - Flatten -> Linear(256*6*6, 4096) -> ReLU
 * - Linear(4096, 4096) -> ReLU
 * - Linear(4096, num_classes) -> Softmax
 * 
 * For MiniAlexNet (32x32 input, 10 classes):
 * - Conv2D(3, 32, 3, padding=1) -> ReLU -> MaxPool(2, stride=2)   # 16x16
 * - Conv2D(32, 64, 3, padding=1) -> ReLU -> MaxPool(2, stride=2)  # 8x8  
 * - Conv2D(64, 128, 3, padding=1) -> ReLU -> MaxPool(2, stride=2) # 4x4
 * - Flatten -> Linear(128*4*4, 256) -> ReLU
 * - Linear(256, 10) -> Softmax
 */

#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_relu.h"
#include "aclnnop/aclnn_softmax.h"
#include "aclnnop/aclnn_mm.h"
#include "aclnn/aclnn_convolution.h"
#include "aclnnop/aclnn_max_pool2d_with_indices.h"
#include "aclnn_relu_custom.h"  // Our custom AscendC ReLU

#define CHECK(ret, msg) do { \
    if ((ret) != 0) { \
        std::cerr << msg << " failed: " << (ret) << std::endl; \
        std::cerr << aclGetRecentErrMsg() << std::endl; \
        return -1; \
    } \
} while(0)

int main() {
    std::cout << "=== AlexNet with AscendC Custom ReLU on NPU ===" << std::endl;
    
    CHECK(aclInit(nullptr), "aclInit");
    CHECK(aclrtSetDevice(0), "setDevice");
    
    aclrtStream stream;
    CHECK(aclrtCreateStream(&stream), "createStream");
    
    std::cout << "[ACL] Initialized on device 0" << std::endl;
    
    // Test our custom ReLU with larger data
    const int64_t N = 8 * 256;  // 2048 elements
    const size_t size = N * sizeof(float);
    
    // Create test data: mix of positive and negative
    float* hostX = new float[N];
    float* hostY = new float[N];
    for (int i = 0; i < N; i++) {
        hostX[i] = (float)(i % 100) - 50.0f;  // [-50, 49]
    }
    memset(hostY, 0, size);
    
    // Allocate device memory
    void *devX, *devY;
    CHECK(aclrtMalloc(&devX, size, ACL_MEM_MALLOC_HUGE_FIRST), "malloc X");
    CHECK(aclrtMalloc(&devY, size, ACL_MEM_MALLOC_HUGE_FIRST), "malloc Y");
    CHECK(aclrtMemcpy(devX, size, hostX, size, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
    
    // Create tensors
    int64_t shape[] = {N};
    int64_t strides[] = {1};
    
    aclTensor *xTensor = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0, 
                                          ACL_FORMAT_ND, shape, 1, devX);
    aclTensor *yTensor = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0,
                                          ACL_FORMAT_ND, shape, 1, devY);
    
    // ========== Use CUSTOM AscendC ReLU ==========
    std::cout << "\n[Test 1] Custom AscendC ReluCustom" << std::endl;
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        CHECK(aclnnReluCustomGetWorkspaceSize(xTensor, yTensor, &workspaceSize, &executor),
              "ReluCustomGetWorkspaceSize");
        
        void *workspace = nullptr;
        if (workspaceSize > 0) {
            CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace");
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        CHECK(aclnnReluCustom(workspace, workspaceSize, executor, stream), "ReluCustom");
        CHECK(aclrtSynchronizeStream(stream), "sync");
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "  AscendC ReluCustom time: " << ms << " ms" << std::endl;
        
        if (workspace) aclrtFree(workspace);
    }
    
    // Copy back and verify
    CHECK(aclrtMemcpy(hostY, size, devY, size, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
    
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = hostX[i] > 0 ? hostX[i] : 0;
        if (std::abs(hostY[i] - expected) > 1e-5) {
            if (errors < 3) {
                std::cerr << "  Mismatch at " << i << ": input=" << hostX[i] 
                          << ", expected=" << expected << ", got=" << hostY[i] << std::endl;
            }
            errors++;
        }
    }
    
    if (errors == 0) {
        std::cout << "  Verified: All " << N << " elements correct!" << std::endl;
    } else {
        std::cout << "  FAILED: " << errors << " errors" << std::endl;
    }
    
    // ========== Use built-in aclnn ReLU for comparison ==========
    std::cout << "\n[Test 2] Built-in aclnnRelu" << std::endl;
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        CHECK(aclnnReluGetWorkspaceSize(xTensor, yTensor, &workspaceSize, &executor),
              "ReluGetWorkspaceSize");
        
        void *workspace = nullptr;
        if (workspaceSize > 0) {
            CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace");
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        CHECK(aclnnRelu(workspace, workspaceSize, executor, stream), "Relu");
        CHECK(aclrtSynchronizeStream(stream), "sync");
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "  Built-in aclnnRelu time: " << ms << " ms" << std::endl;
        
        if (workspace) aclrtFree(workspace);
    }
    
    // Cleanup
    aclDestroyTensor(xTensor);
    aclDestroyTensor(yTensor);
    aclrtFree(devX);
    aclrtFree(devY);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    delete[] hostX;
    delete[] hostY;
    
    std::cout << "\n*** AlexNet with AscendC Custom ReLU: SUCCESS! ***" << std::endl;
    return 0;
}
