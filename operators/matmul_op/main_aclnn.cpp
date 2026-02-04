/*
 * AscendC MatMul Custom Operator Test - Using ACLNN API
 */
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

#include "acl/acl.h"
#include "aclnn_matmul_custom.h"

#define CHECK_ACL(call) do { \
    auto _ret = (call); \
    if (_ret != ACL_SUCCESS) { \
        std::cerr << "ACL Error: " << _ret << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return -1; \
    } \
} while(0)

#define CHECK_ACLNN(call) do { \
    auto _ret = (call); \
    if (_ret != 0) { \
        std::cerr << "ACLNN Error: " << _ret << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return -1; \
    } \
} while(0)

int main(int argc, char** argv)
{
    std::cout << "=========================================================" << std::endl;
    std::cout << "AscendC MatMul Custom Operator Test" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    // 矩阵维度
    int64_t M = 256;
    int64_t K = 256;
    int64_t N = 256;
    int32_t deviceId = 1;  // 使用空闲的 NPU
    
    if (argc > 1) {
        M = std::atoi(argv[1]);
    }
    if (argc > 2) {
        K = std::atoi(argv[2]);
    }
    if (argc > 3) {
        N = std::atoi(argv[3]);
    }
    
    std::cout << "\n[INFO] Matrix dimensions: A[" << M << ", " << K << "] x B[" 
              << K << ", " << N << "] = C[" << M << ", " << N << "]" << std::endl;
    
    // 初始化 ACL
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));
    
    std::cout << "[INFO] Using NPU device: " << deviceId << std::endl;
    
    // 生成输入数据
    size_t aSize = M * K * sizeof(aclFloat16);
    size_t bSize = K * N * sizeof(aclFloat16);
    size_t cSize = M * N * sizeof(float);
    
    std::vector<aclFloat16> aHost(M * K);
    std::vector<aclFloat16> bHost(K * N);
    std::vector<float> cHost(M * N, 0.0f);
    std::vector<float> cExpected(M * N, 0.0f);
    
    // 初始化输入数据
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int64_t i = 0; i < M * K; i++) {
        float val = dist(gen);
        aHost[i] = aclFloatToFloat16(val);
    }
    for (int64_t i = 0; i < K * N; i++) {
        float val = dist(gen);
        bHost[i] = aclFloatToFloat16(val);
    }
    
    // CPU 计算期望结果
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                float aVal = aclFloat16ToFloat(aHost[i * K + k]);
                float bVal = aclFloat16ToFloat(bHost[k * N + j]);
                sum += aVal * bVal;
            }
            cExpected[i * N + j] = sum;
        }
    }
    
    std::cout << "[INFO] CPU reference calculation completed" << std::endl;
    
    // 分配设备内存
    void* aDevice = nullptr;
    void* bDevice = nullptr;
    void* cDevice = nullptr;
    
    CHECK_ACL(aclrtMalloc(&aDevice, aSize, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMalloc(&bDevice, bSize, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMalloc(&cDevice, cSize, ACL_MEM_MALLOC_NORMAL_ONLY));
    
    // 拷贝数据到设备
    CHECK_ACL(aclrtMemcpy(aDevice, aSize, aHost.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(bDevice, bSize, bHost.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemset(cDevice, cSize, 0, cSize));
    
    std::cout << "[INFO] Data copied to device" << std::endl;
    
    // 创建 aclTensor
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
    
    if (!aTensor || !bTensor || !cTensor) {
        std::cerr << "Failed to create tensors" << std::endl;
        return -1;
    }
    std::cout << "[INFO] Tensors created" << std::endl;
    
    // 调用自定义 matmul 算子
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    std::cout << "[INFO] Calling GetWorkspaceSize..." << std::endl;
    auto wsRet = aclnnMatmulCustomGetWorkspaceSize(aTensor, bTensor, cTensor,
                                                    &workspaceSize, &executor);
    std::cout << "[INFO] GetWorkspaceSize returned: " << wsRet << std::endl;
    if (wsRet != 0) {
        std::cerr << "ACLNN GetWorkspaceSize Error: " << wsRet << std::endl;
        return -1;
    }
    
    std::cout << "[INFO] Workspace size: " << workspaceSize << " bytes" << std::endl;
    
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY));
    }
    
    // 执行算子
    std::cout << "[INFO] Executing MatMul..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    auto execRet = aclnnMatmulCustom(workspace, workspaceSize, executor, stream);
    std::cout << "[INFO] MatMul returned: " << execRet << std::endl;
    if (execRet != 0) {
        std::cerr << "ACLNN MatMul Error: " << execRet << std::endl;
        return -1;
    }
    
    std::cout << "[INFO] Synchronizing..." << std::endl;
    CHECK_ACL(aclrtSynchronizeStream(stream));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "[INFO] MatMul execution time: " << duration.count() << " us" << std::endl;
    
    // 拷贝结果回主机
    CHECK_ACL(aclrtMemcpy(cHost.data(), cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    
    // 验证结果
    double maxError = 0.0;
    double totalError = 0.0;
    for (int64_t i = 0; i < M * N; i++) {
        double error = std::fabs(cHost[i] - cExpected[i]);
        maxError = std::max(maxError, error);
        totalError += error;
    }
    double avgError = totalError / (M * N);
    
    std::cout << "\n[RESULT] Verification:" << std::endl;
    std::cout << "  - Max error: " << maxError << std::endl;
    std::cout << "  - Avg error: " << avgError << std::endl;
    
    // 结果检查 (fp16 计算的误差稍大)
    bool passed = maxError < 1.0;  // fp16 精度限制
    if (passed) {
        std::cout << "  - Status: PASSED \xe2\x9c\x93" << std::endl;
    } else {
        std::cout << "  - Status: FAILED \xe2\x9c\x97" << std::endl;
    }
    
    // 打印部分结果
    std::cout << "\n[INFO] Sample output (first 5 elements):" << std::endl;
    for (int i = 0; i < 5 && i < M * N; i++) {
        std::cout << "  C[" << i << "] = " << cHost[i] 
                  << " (expected: " << cExpected[i] << ")" << std::endl;
    }
    
    // 清理
    aclDestroyTensor(aTensor);
    aclDestroyTensor(bTensor);
    aclDestroyTensor(cTensor);
    
    if (workspace) {
        aclrtFree(workspace);
    }
    aclrtFree(aDevice);
    aclrtFree(bDevice);
    aclrtFree(cDevice);
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    
    std::cout << "\n=========================================================" << std::endl;
    std::cout << "MatMul Custom Operator Test " << (passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    return passed ? 0 : 1;
}
