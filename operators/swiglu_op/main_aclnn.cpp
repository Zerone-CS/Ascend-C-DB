#include <stdio.h>
#include <string.h>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn_swi_glu_custom.h"

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
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> xShape = {8, 256};
    std::vector<int64_t> gateShape = {8, 256};
    std::vector<int64_t> outShape = {8, 256};
    void* xDeviceAddr = nullptr;
    void* gateDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* gate = nullptr;
    aclTensor* out = nullptr;

    int64_t totalSize = 8 * 256;
    std::vector<float> xHostData(totalSize);
    std::vector<float> gateHostData(totalSize);
    std::vector<float> outHostData(totalSize, 0);
    std::vector<float> goldenData(totalSize);
    
    // Initialize input data with random-like values
    for (int i = 0; i < totalSize; i++) {
        xHostData[i] = sinf(i * 0.1f);
        gateHostData[i] = cosf(i * 0.1f);
        // SwiGLU: y = x * gate * sigmoid(gate)
        float sig = 1.0f / (1.0f + expf(-gateHostData[i]));
        goldenData[i] = xHostData[i] * gateHostData[i] * sig;
    }

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gateHostData, gateShape, &gateDeviceAddr, aclDataType::ACL_FLOAT, &gate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    
    ret = aclnnSwiGLUCustomGetWorkspaceSize(x, gate, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwiGLUCustomGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("workspaceSize=%lu\n", workspaceSize);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnSwiGLUCustom(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwiGLUCustom failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(outShape.data(), outShape.size()) * sizeof(float);
    ret = aclrtMemcpy(outHostData.data(), size, outDeviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("\nInput x[0:4]: ");
    for (int i = 0; i < 4; i++) {
        LOG_PRINT("%.6f ", xHostData[i]);
    }
    LOG_PRINT("\nInput gate[0:4]: ");
    for (int i = 0; i < 4; i++) {
        LOG_PRINT("%.6f ", gateHostData[i]);
    }
    LOG_PRINT("\nOutput[0:4]: ");
    for (int i = 0; i < 4; i++) {
        LOG_PRINT("%.6f ", outHostData[i]);
    }
    LOG_PRINT("\nGolden[0:4]: ");
    for (int i = 0; i < 4; i++) {
        LOG_PRINT("%.6f ", goldenData[i]);
    }
    LOG_PRINT("\n");
    
    // Verify results
    float maxErr = 0.0f;
    for (int i = 0; i < totalSize; i++) {
        float err = fabsf(outHostData[i] - goldenData[i]);
        if (err > maxErr) maxErr = err;
    }
    LOG_PRINT("\nMax error: %.2e\n", maxErr);
    if (maxErr < 1e-5f) {
        LOG_PRINT("\033[32m\u2705 SwiGLU TEST PASSED!\033[0m\n");
    } else {
        LOG_PRINT("\033[31m\u274c SwiGLU TEST FAILED!\033[0m\n");
    }

    aclDestroyTensor(x);
    aclDestroyTensor(gate);
    aclDestroyTensor(out);
    aclrtFree(xDeviceAddr);
    aclrtFree(gateDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("SwiGLU custom operator test completed!\n");
    return 0;
}
