#include <iostream>
#include <cstring>
#include <cmath>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn_relu_custom.h"

#define CHECK(ret, msg) do { \
    if ((ret) != ACL_SUCCESS) { \
        std::cerr << msg << " failed: " << (ret) << std::endl; \
        std::cerr << aclGetRecentErrMsg() << std::endl; \
        return -1; \
    } \
} while(0)

#define CHECK_ACLNN(ret, msg) do { \
    if ((ret) != 0) { \
        std::cerr << msg << " failed: " << (ret) << std::endl; \
        std::cerr << aclGetRecentErrMsg() << std::endl; \
        return -1; \
    } \
} while(0)

int main() {
    std::cout << "=== Testing AscendC ReluCustom via aclnn ===" << std::endl;
    
    CHECK(aclInit(nullptr), "aclInit");
    CHECK(aclrtSetDevice(0), "setDevice");
    
    aclrtStream stream;
    CHECK(aclrtCreateStream(&stream), "createStream");
    std::cout << "[ACL] Initialized" << std::endl;
    
    // Test data: 8*256 elements (divisible by blockDim*tileSize)
    const int64_t N = 8 * 256;
    const size_t size = N * sizeof(float);
    
    float hostX[N], hostY[N];
    for (int i = 0; i < N; i++) {
        hostX[i] = (float)i - N/2;  // [-1024, 1023]
    }
    memset(hostY, 0, size);
    
    std::cout << "Input[0:8]: ";
    for (int i = 0; i < 8; i++) std::cout << hostX[i] << " ";
    std::cout << std::endl;
    
    // Allocate device memory
    void *devX, *devY;
    CHECK(aclrtMalloc(&devX, size, ACL_MEM_MALLOC_HUGE_FIRST), "malloc X");
    CHECK(aclrtMalloc(&devY, size, ACL_MEM_MALLOC_HUGE_FIRST), "malloc Y");
    CHECK(aclrtMemcpy(devX, size, hostX, size, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
    CHECK(aclrtMemset(devY, size, 0, size), "memset Y");
    
    // Create aclTensor
    int64_t shape[] = {N};
    int64_t strides[] = {1};
    
    aclTensor *xTensor = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0, 
                                          ACL_FORMAT_ND, shape, 1, devX);
    aclTensor *yTensor = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0,
                                          ACL_FORMAT_ND, shape, 1, devY);
    
    if (!xTensor || !yTensor) {
        std::cerr << "Failed to create tensors" << std::endl;
        return -1;
    }
    std::cout << "[ACL] Tensors created" << std::endl;
    
    // Get workspace size
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    
    aclnnStatus ret = aclnnReluCustomGetWorkspaceSize(xTensor, yTensor, &workspaceSize, &executor);
    CHECK_ACLNN(ret, "aclnnReluCustomGetWorkspaceSize");
    
    std::cout << "[ACL] Workspace size: " << workspaceSize << std::endl;
    
    // Allocate workspace if needed
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace");
    }
    
    // Execute the operator
    ret = aclnnReluCustom(workspace, workspaceSize, executor, stream);
    CHECK_ACLNN(ret, "aclnnReluCustom");
    
    CHECK(aclrtSynchronizeStream(stream), "sync");
    std::cout << "[ACL] Kernel executed" << std::endl;
    
    // Copy output back
    CHECK(aclrtMemcpy(hostY, size, devY, size, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
    
    std::cout << "Output[0:8]: ";
    for (int i = 0; i < 8; i++) std::cout << hostY[i] << " ";
    std::cout << std::endl;
    
    // Verify
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = hostX[i] > 0 ? hostX[i] : 0;
        if (std::abs(hostY[i] - expected) > 1e-5) {
            std::cerr << "Mismatch at " << i << ": expected " << expected << ", got " << hostY[i] << std::endl;
            ok = false;
            break;
        }
    }
    
    // Cleanup
    aclDestroyTensor(xTensor);
    aclDestroyTensor(yTensor);
    if (workspace) aclrtFree(workspace);
    aclrtFree(devX);
    aclrtFree(devY);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    if (ok) {
        std::cout << "\n*** AscendC ReluCustom SUCCESS! ***" << std::endl;
        return 0;
    } else {
        std::cout << "\n*** FAILED ***" << std::endl;
        return 1;
    }
}
