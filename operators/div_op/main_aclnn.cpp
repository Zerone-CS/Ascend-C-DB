#include <stdio.h>
#include <string.h>
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "DivCustom/build_out/autogen/aclnn_div_custom.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const int64_t* shape, uint64_t shapeLen) {
    int64_t shapeSize = 1;
    for (uint64_t i = 0; i < shapeLen; i++) {
        shapeSize *= shape[i];
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                   aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape.data(), shape.size()) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    int32_t deviceId = 2;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> shape = {8, 256};
    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* zDeviceAddr = nullptr;
    aclTensor* xTensor = nullptr;
    aclTensor* yTensor = nullptr;
    aclTensor* zTensor = nullptr;

    int64_t totalSize = 8 * 256;
    std::vector<float> xHostData(totalSize);
    std::vector<float> yHostData(totalSize);
    std::vector<float> zHostData(totalSize, 0);
    
    for (int i = 0; i < totalSize; i++) {
        xHostData[i] = (float)(i % 20) - 10.0f;
        yHostData[i] = (float)(i % 7) + 1.0f;
    }

    ret = CreateAclTensor(xHostData, shape, &xDeviceAddr, aclDataType::ACL_FLOAT, &xTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, shape, &yDeviceAddr, aclDataType::ACL_FLOAT, &yTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(zHostData, shape, &zDeviceAddr, aclDataType::ACL_FLOAT, &zTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    
    ret = aclnnDivCustomGetWorkspaceSize(xTensor, yTensor, zTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivCustomGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("workspaceSize=%lu\n", workspaceSize);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnDivCustom(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivCustom failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(shape.data(), shape.size()) * sizeof(float);
    ret = aclrtMemcpy(zHostData.data(), size, zDeviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("\n=== Div Custom Operator Test ===\n");
    LOG_PRINT("Input x (first 10): ");
    for (int i = 0; i < 10; i++) {
        LOG_PRINT("%.2f ", xHostData[i]);
    }
    LOG_PRINT("\nInput y (first 10): ");
    for (int i = 0; i < 10; i++) {
        LOG_PRINT("%.2f ", yHostData[i]);
    }
    LOG_PRINT("\nNPU Output:        ");
    for (int i = 0; i < 10; i++) {
        LOG_PRINT("%.4f ", zHostData[i]);
    }
    LOG_PRINT("\nCPU Expected:      ");
    for (int i = 0; i < 10; i++) {
        LOG_PRINT("%.4f ", xHostData[i] / yHostData[i]);
    }
    LOG_PRINT("\n");

    float maxError = 0.0f;
    int errorCount = 0;
    for (int i = 0; i < totalSize; i++) {
        float expected = xHostData[i] / yHostData[i];
        float error = fabsf(zHostData[i] - expected);
        if (error > maxError) maxError = error;
        if (error > 1e-5) errorCount++;
    }
    LOG_PRINT("\nMax error: %.6f\n", maxError);
    LOG_PRINT("Error count (>1e-5): %d/%ld\n", errorCount, totalSize);
    
    bool passed = (maxError < 1e-4);
    LOG_PRINT("\nTest: %s\n", passed ? "PASSED" : "FAILED");

    aclDestroyTensor(xTensor);
    aclDestroyTensor(yTensor);
    aclDestroyTensor(zTensor);
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(zDeviceAddr);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return passed ? 0 : 1;
}
