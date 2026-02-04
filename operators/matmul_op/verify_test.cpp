/*
 * Verify MatMul custom operator correctness
 */
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#include "acl/acl.h"
#include "aclnn_matmul_custom.h"

int main()
{
    std::cout << "=== MatMul Verification Test ===\n";
    
    aclError ret = aclInit(nullptr);
    ret = aclrtSetDevice(1);
    
    aclrtStream stream;
    aclrtCreateStream(&stream);
    
    // 4x4 矩阵便于手动验算
    int64_t M = 4, K = 4, N = 4;
    
    std::cout << "Matrix: A[" << M << "," << K << "] x B[" << K << "," << N << "]\n";
    
    size_t aSize = M * K * sizeof(aclFloat16);
    size_t bSize = K * N * sizeof(aclFloat16);
    size_t cSize = M * N * sizeof(float);
    
    // 创建测试数据
    // A = [[1,1,1,1], [2,2,2,2], [3,3,3,3], [4,4,4,4]]
    // B = I (4x4 单位阵)
    // C 应该等于 A
    std::vector<aclFloat16> aHost(M * K);
    std::vector<aclFloat16> bHost(K * N);
    std::vector<float> cExpected(M * N);
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            aHost[i * K + j] = aclFloatToFloat16((float)(i + 1));
        }
    }
    
    // B = 单位矩阵
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            bHost[i * N + j] = aclFloatToFloat16(i == j ? 1.0f : 0.0f);
        }
    }
    
    // 计算期望结果: C = A * B = A (因为 B 是单位阵)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += aclFloat16ToFloat(aHost[i * K + k]) * aclFloat16ToFloat(bHost[k * N + j]);
            }
            cExpected[i * N + j] = sum;
        }
    }
    
    std::cout << "\nExpected C:\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << cExpected[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    
    // 分配设备内存
    void* aDevice = nullptr;
    void* bDevice = nullptr;
    void* cDevice = nullptr;
    
    aclrtMalloc(&aDevice, aSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&bDevice, bSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&cDevice, cSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    
    aclrtMemcpy(aDevice, aSize, aHost.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(bDevice, bSize, bHost.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemset(cDevice, cSize, 0, cSize);
    
    // 创建 tensor
    int64_t aShape[] = {M, K};
    int64_t bShape[] = {K, N};
    int64_t cShape[] = {M, N};
    int64_t aStrides[] = {K, 1};
    int64_t bStrides[] = {N, 1};
    int64_t cStrides[] = {N, 1};
    
    aclTensor* aTensor = aclCreateTensor(aShape, 2, ACL_FLOAT16, aStrides, 0,
                                          ACL_FORMAT_ND, aShape, 2, aDevice);
    aclTensor* bTensor = aclCreateTensor(bShape, 2, ACL_FLOAT16, bStrides, 0,
                                          ACL_FORMAT_ND, bShape, 2, bDevice);
    aclTensor* cTensor = aclCreateTensor(cShape, 2, ACL_FLOAT, cStrides, 0,
                                          ACL_FORMAT_ND, cShape, 2, cDevice);
    
    // 调用算子
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    aclnnMatmulCustomGetWorkspaceSize(aTensor, bTensor, cTensor, &workspaceSize, &executor);
    
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    }
    
    aclnnMatmulCustom(workspace, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    // 读取结果
    std::vector<float> cResult(M * N);
    aclrtMemcpy(cResult.data(), cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::cout << "\nActual C:\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << cResult[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    
    // 验证结果
    bool pass = true;
    float maxError = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = std::abs(cResult[i] - cExpected[i]);
        maxError = std::max(maxError, error);
        if (error > 0.01f) {
            pass = false;
        }
    }
    
    std::cout << "\nMax error: " << maxError << "\n";
    std::cout << "Test: " << (pass ? "PASSED" : "FAILED") << "\n";
    
    // 清理
    aclDestroyTensor(aTensor);
    aclDestroyTensor(bTensor);
    aclDestroyTensor(cTensor);
    if (workspace) aclrtFree(workspace);
    aclrtFree(aDevice);
    aclrtFree(bDevice);
    aclrtFree(cDevice);
    aclrtDestroyStream(stream);
    aclrtResetDevice(1);
    aclFinalize();
    
    return pass ? 0 : 1;
}
