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
    std::cout << "=== Full Test: AscendC ReluCustom via aclnn ===" << std::endl;
    
    CHECK(aclInit(nullptr), "aclInit");
    CHECK(aclrtSetDevice(0), "setDevice");
    
    aclrtStream stream;
    CHECK(aclrtCreateStream(&stream), "createStream");
    
    const int64_t N = 8 * 256;
    const size_t size = N * sizeof(float);
    
    float* hostX = new float[N];
    float* hostY = new float[N];
    
    // Mix of negative and positive values
    for (int i = 0; i < N; i++) {
        hostX[i] = (float)(i % 100) - 50.0f;  // [-50, 49]
    }
    memset(hostY, 0, size);
    
    std::cout << "Input range: [-50, 49]" << std::endl;
    std::cout << "Input[0:10]: ";
    for (int i = 0; i < 10; i++) std::cout << hostX[i] << " ";
    std::cout << std::endl;
    
    void *devX, *devY;
    CHECK(aclrtMalloc(&devX, size, ACL_MEM_MALLOC_HUGE_FIRST), "malloc X");
    CHECK(aclrtMalloc(&devY, size, ACL_MEM_MALLOC_HUGE_FIRST), "malloc Y");
    CHECK(aclrtMemcpy(devX, size, hostX, size, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
    CHECK(aclrtMemset(devY, size, 0, size), "memset Y");
    
    int64_t shape[] = {N};
    int64_t strides[] = {1};
    
    aclTensor *xTensor = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0, 
                                          ACL_FORMAT_ND, shape, 1, devX);
    aclTensor *yTensor = aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0,
                                          ACL_FORMAT_ND, shape, 1, devY);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    
    CHECK_ACLNN(aclnnReluCustomGetWorkspaceSize(xTensor, yTensor, &workspaceSize, &executor),
                "GetWorkspaceSize");
    
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace");
    }
    
    CHECK_ACLNN(aclnnReluCustom(workspace, workspaceSize, executor, stream), "aclnnReluCustom");
    CHECK(aclrtSynchronizeStream(stream), "sync");
    
    CHECK(aclrtMemcpy(hostY, size, devY, size, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
    
    std::cout << "Output[0:10]: ";
    for (int i = 0; i < 10; i++) std::cout << hostY[i] << " ";
    std::cout << std::endl;
    
    // Full verification
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = hostX[i] > 0 ? hostX[i] : 0;
        if (std::abs(hostY[i] - expected) > 1e-5) {
            if (errors < 5) {
                std::cerr << "Mismatch at " << i << ": input=" << hostX[i] 
                          << ", expected=" << expected << ", got=" << hostY[i] << std::endl;
            }
            errors++;
        }
    }
    
    aclDestroyTensor(xTensor);
    aclDestroyTensor(yTensor);
    if (workspace) aclrtFree(workspace);
    aclrtFree(devX);
    aclrtFree(devY);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    delete[] hostX;
    delete[] hostY;
    
    if (errors == 0) {
        std::cout << "\n*** ALL " << N << " ELEMENTS VERIFIED! AscendC ReluCustom SUCCESS! ***" << std::endl;
        return 0;
    } else {
        std::cout << "\n*** FAILED: " << errors << " errors ***" << std::endl;
        return 1;
    }
}
