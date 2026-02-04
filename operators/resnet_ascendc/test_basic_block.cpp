/**
 * 测试 ResNet BasicBlock - 使用所有 AscendC 自定义算子
 * BasicBlock: Conv -> BN -> ReLU -> Conv -> BN -> Add(residual) -> ReLU
 */
#include <stdio.h>
#include <cmath>
#include <vector>
#include <random>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

// AscendC 自定义算子头文件
#include "ResNetOps/build_out/autogen/aclnn_conv2d_custom.h"
#include "ResNetOps/build_out/autogen/aclnn_bn_custom.h"
#include "ResNetOps/build_out/autogen/aclnn_relu_custom.h"
#include "ResNetOps/build_out/autogen/aclnn_add_custom.h"

#define CHECK_ACL(ret, msg) do { \
    if ((ret) != ACL_SUCCESS) { \
        printf("[ERROR] %s, ret=%d\n", msg, ret); \
        printf("  ACL: %s\n", aclGetRecentErrMsg()); \
        return -1; \
    } \
} while(0)

class NPUTensor {
public:
    std::vector<int64_t> shape;
    int64_t numElements = 0;
    void* devPtr = nullptr;
    aclTensor* tensor = nullptr;

    int create(const std::vector<int64_t>& sh) {
        shape = sh;
        numElements = 1;
        for (auto s : shape) numElements *= s;
        int64_t byteSize = numElements * sizeof(float);

        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = shape.size() - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];

        auto ret = aclrtMalloc(&devPtr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return ret;
        tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
                                 strides.data(), 0, ACL_FORMAT_ND,
                                 shape.data(), shape.size(), devPtr);
        return tensor ? 0 : -1;
    }

    int fromHost(const float* data) {
        return aclrtMemcpy(devPtr, numElements * sizeof(float), data,
                           numElements * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    }
    int toHost(float* data) {
        return aclrtMemcpy(data, numElements * sizeof(float), devPtr,
                           numElements * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    void destroy() {
        if (tensor) { aclDestroyTensor(tensor); tensor = nullptr; }
        if (devPtr) { aclrtFree(devPtr); devPtr = nullptr; }
    }
};

int runConv2d(NPUTensor& x, NPUTensor& w, NPUTensor& y, int stride, int pad, aclrtStream stream) {
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnConv2dCustomGetWorkspaceSize(x.tensor, w.tensor, stride, pad, y.tensor, &ws, &exec);
    if (ret != ACL_SUCCESS) { printf("Conv2d GetWS failed: %d\n", ret); return ret; }
    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnConv2dCustom(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

int runBn(NPUTensor& x, NPUTensor& gamma, NPUTensor& beta, NPUTensor& mean, NPUTensor& var,
          NPUTensor& y, aclrtStream stream) {
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnBnCustomGetWorkspaceSize(x.tensor, gamma.tensor, beta.tensor, mean.tensor, var.tensor,
                                              y.tensor, &ws, &exec);
    if (ret != ACL_SUCCESS) { printf("BN GetWS failed: %d\n", ret); return ret; }
    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnBnCustom(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

int runRelu(NPUTensor& x, NPUTensor& y, aclrtStream stream) {
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnReluCustomGetWorkspaceSize(x.tensor, y.tensor, &ws, &exec);
    if (ret != ACL_SUCCESS) { printf("ReLU GetWS failed: %d\n", ret); return ret; }
    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnReluCustom(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

int runAdd(NPUTensor& x1, NPUTensor& x2, NPUTensor& y, aclrtStream stream) {
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    auto ret = aclnnAddCustomGetWorkspaceSize(x1.tensor, x2.tensor, y.tensor, &ws, &exec);
    if (ret != ACL_SUCCESS) { printf("Add GetWS failed: %d\n", ret); return ret; }
    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnAddCustom(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

int main() {
    printf("\n=== ResNet BasicBlock Test (Pure AscendC) ===\n");
    printf("BasicBlock: Conv->BN->ReLU->Conv->BN + residual -> ReLU\n\n");

    int deviceId = 2;
    auto ret = aclInit(nullptr);
    CHECK_ACL(ret, "aclInit");
    ret = aclrtSetDevice(deviceId);
    CHECK_ACL(ret, "aclrtSetDevice");
    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    CHECK_ACL(ret, "aclrtCreateStream");

    // BasicBlock 参数: 64 channels, 3x3 conv
    int N = 1, C = 64, H = 56, W = 56;
    int outC = 64, kH = 3, kW = 3;
    int stride = 1, pad = 1;

    printf("[1] 初始化张量...\n");
    NPUTensor input, conv1_w, conv1_out, bn1_out, relu1_out;
    NPUTensor conv2_w, conv2_out, bn2_out, add_out, final_out;
    NPUTensor bn_gamma, bn_beta, bn_mean, bn_var;

    // 输入
    input.create({N, C, H, W});
    std::vector<float> inputData(input.numElements);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : inputData) v = dist(gen);
    input.fromHost(inputData.data());

    // Conv1 weights
    conv1_w.create({outC, C, kH, kW});
    std::vector<float> w1(conv1_w.numElements);
    float stddev = std::sqrt(2.0f / (C * kH * kW));
    for (auto& v : w1) v = dist(gen) * stddev;
    conv1_w.fromHost(w1.data());

    // Conv2 weights
    conv2_w.create({outC, outC, kH, kW});
    std::vector<float> w2(conv2_w.numElements);
    for (auto& v : w2) v = dist(gen) * stddev;
    conv2_w.fromHost(w2.data());

    // BN params
    bn_gamma.create({outC});
    bn_beta.create({outC});
    bn_mean.create({outC});
    bn_var.create({outC});
    std::vector<float> ones(outC, 1.0f), zeros(outC, 0.0f);
    bn_gamma.fromHost(ones.data());
    bn_beta.fromHost(zeros.data());
    bn_mean.fromHost(zeros.data());
    bn_var.fromHost(ones.data());

    // 中间张量
    int outH = (H + 2*pad - kH) / stride + 1;
    int outW = (W + 2*pad - kW) / stride + 1;
    conv1_out.create({N, outC, outH, outW});
    bn1_out.create({N, outC, outH, outW});
    relu1_out.create({N, outC, outH, outW});
    conv2_out.create({N, outC, outH, outW});
    bn2_out.create({N, outC, outH, outW});
    add_out.create({N, outC, outH, outW});
    final_out.create({N, outC, outH, outW});

    printf("[2] 执行 BasicBlock...\n");

    // Conv1
    printf("  -> Conv1 [%d,%d,%d,%d] * [%d,%d,%d,%d]...\n", N, C, H, W, outC, C, kH, kW);
    ret = runConv2d(input, conv1_w, conv1_out, stride, pad, stream);
    CHECK_ACL(ret, "Conv1");

    // BN1
    printf("  -> BN1...\n");
    ret = runBn(conv1_out, bn_gamma, bn_beta, bn_mean, bn_var, bn1_out, stream);
    CHECK_ACL(ret, "BN1");

    // ReLU1
    printf("  -> ReLU1...\n");
    ret = runRelu(bn1_out, relu1_out, stream);
    CHECK_ACL(ret, "ReLU1");

    // Conv2
    printf("  -> Conv2...\n");
    ret = runConv2d(relu1_out, conv2_w, conv2_out, stride, pad, stream);
    CHECK_ACL(ret, "Conv2");

    // BN2
    printf("  -> BN2...\n");
    ret = runBn(conv2_out, bn_gamma, bn_beta, bn_mean, bn_var, bn2_out, stream);
    CHECK_ACL(ret, "BN2");

    // Add (residual)
    printf("  -> Add (residual)...\n");
    ret = runAdd(bn2_out, input, add_out, stream);
    CHECK_ACL(ret, "Add");

    // ReLU2
    printf("  -> ReLU2 (final)...\n");
    ret = runRelu(add_out, final_out, stream);
    CHECK_ACL(ret, "ReLU2");

    aclrtSynchronizeStream(stream);

    // 读取结果
    std::vector<float> result(final_out.numElements);
    final_out.toHost(result.data());

    printf("\n[3] 结果验证:\n");
    printf("  输入 shape: [%d, %d, %d, %d]\n", N, C, H, W);
    printf("  输出 shape: [%d, %d, %d, %d]\n", N, outC, outH, outW);
    
    float minVal = result[0], maxVal = result[0], sum = 0;
    for (auto v : result) {
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
        sum += v;
    }
    printf("  输出统计: min=%.4f, max=%.4f, mean=%.4f\n", minVal, maxVal, sum/result.size());
    printf("  前10个值: ");
    for (int i = 0; i < 10 && i < result.size(); i++) printf("%.3f ", result[i]);
    printf("\n");

    // 清理
    input.destroy(); conv1_w.destroy(); conv1_out.destroy();
    bn1_out.destroy(); relu1_out.destroy();
    conv2_w.destroy(); conv2_out.destroy(); bn2_out.destroy();
    add_out.destroy(); final_out.destroy();
    bn_gamma.destroy(); bn_beta.destroy(); bn_mean.destroy(); bn_var.destroy();
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    printf("\n=== BasicBlock 测试完成 ===\n");
    return 0;
}
