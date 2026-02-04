/**
 * 测试 AscendC 自定义 ReLU 算子
 */
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "ReluCustom/build_out/autogen/aclnn_relu_custom.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (auto s : shape) size *= s;
    return size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclInit failed: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtSetDevice failed: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtCreateStream failed: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, 
                   void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtMalloc failed: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtMemcpy H2D failed: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 
                              0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    printf("\n=== AscendC 自定义 ReLU 算子测试 ===\n\n");
    
    int32_t deviceId = 2;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, return ret);

    // 准备测试数据: 包含正负数
    std::vector<int64_t> shape = {2, 1024};
    int64_t totalSize = GetShapeSize(shape);
    std::vector<float> xHostData(totalSize);
    std::vector<float> outHostData(totalSize, 0);
    
    for (int i = 0; i < totalSize; i++) {
        xHostData[i] = (float)(i % 41) - 20.0f;  // -20 到 +20
    }

    void* xDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* out = nullptr;

    ret = CreateAclTensor(xHostData, shape, &xDeviceAddr, ACL_FLOAT, &x);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, ACL_FLOAT, &out);
    CHECK_RET(ret == 0, return ret);

    // 调用自定义 ReLU 算子
    printf("[第1阶段] aclnnReluCustomGetWorkspaceSize...\n");
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnReluCustomGetWorkspaceSize(x, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, printf("GetWorkspaceSize failed: %d, %s\n", ret, aclGetRecentErrMsg()); return ret);
    printf("  workspaceSize = %lu bytes\n", workspaceSize);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, printf("alloc workspace failed: %d\n", ret); return ret);
    }

    printf("[第2阶段] aclnnReluCustom 执行计算...\n");
    ret = aclnnReluCustom(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclnnReluCustom failed: %d, %s\n", ret, aclGetRecentErrMsg()); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtSynchronizeStream failed: %d\n", ret); return ret);

    // 读取结果
    auto size = GetShapeSize(shape) * sizeof(float);
    ret = aclrtMemcpy(outHostData.data(), size, outDeviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtMemcpy D2H failed: %d\n", ret); return ret);

    // 验证结果
    printf("\n=== 结果验证 ===\n");
    printf("输入 (10 个): ");
    for (int i = 0; i < 10; i++) printf("%.1f ", xHostData[i]);
    printf("\n输出 (10 个): ");
    for (int i = 0; i < 10; i++) printf("%.1f ", outHostData[i]);
    printf("\n期望 (10 个): ");
    for (int i = 0; i < 10; i++) printf("%.1f ", std::max(0.0f, xHostData[i]));
    printf("\n");

    float maxError = 0.0f;
    int errorCount = 0;
    for (int i = 0; i < totalSize; i++) {
        float expected = std::max(0.0f, xHostData[i]);
        float error = fabsf(outHostData[i] - expected);
        if (error > maxError) maxError = error;
        if (error > 1e-5) errorCount++;
    }
    printf("\n最大误差: %.6f\n", maxError);
    printf("错误数量 (>1e-5): %d/%ld\n", errorCount, totalSize);
    
    bool passed = (maxError < 1e-4);
    printf("\n测试: %s\n\n", passed ? "✅ PASSED" : "❌ FAILED");

    // 清理
    aclDestroyTensor(x);
    aclDestroyTensor(out);
    aclrtFree(xDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return passed ? 0 : 1;
}
