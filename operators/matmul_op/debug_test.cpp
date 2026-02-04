/*
 * Debug test for MatMul custom operator
 */
#include <iostream>
#include <vector>
#include <cstring>

#include "acl/acl.h"
#include "aclnn_matmul_custom.h"

int main()
{
    std::cout << "=== MatMul Debug Test ===\n";
    
    // 初始化
    aclError ret = aclInit(nullptr);
    std::cout << "aclInit: " << ret << std::endl;
    
    ret = aclrtSetDevice(1);
    std::cout << "aclrtSetDevice: " << ret << std::endl;
    
    // 获取最后一次错误信息
    const char* errMsg = aclGetRecentErrMsg();
    if (errMsg) {
        std::cout << "Recent error: " << errMsg << std::endl;
    }
    
    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    std::cout << "aclrtCreateStream: " << ret << std::endl;
    
    // 使用较小的矩阵测试: 16x16
    int64_t M = 16, K = 16, N = 16;
    
    std::cout << "\nMatrix: A[" << M << "," << K << "] x B[" << K << "," << N << "]\n";
    
    size_t aSize = M * K * sizeof(aclFloat16);
    size_t bSize = K * N * sizeof(aclFloat16);
    size_t cSize = M * N * sizeof(float);
    
    // 分配主机内存
    std::vector<aclFloat16> aHost(M * K);
    std::vector<aclFloat16> bHost(K * N);
    
    // 初始化为简单值
    for (int i = 0; i < M * K; i++) {
        aHost[i] = aclFloatToFloat16(1.0f);
    }
    for (int i = 0; i < K * N; i++) {
        bHost[i] = aclFloatToFloat16(1.0f);
    }
    
    // 分配设备内存
    void* aDevice = nullptr;
    void* bDevice = nullptr;
    void* cDevice = nullptr;
    
    ret = aclrtMalloc(&aDevice, aSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    std::cout << "aclrtMalloc A: " << ret << ", ptr=" << aDevice << std::endl;
    
    ret = aclrtMalloc(&bDevice, bSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    std::cout << "aclrtMalloc B: " << ret << ", ptr=" << bDevice << std::endl;
    
    ret = aclrtMalloc(&cDevice, cSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    std::cout << "aclrtMalloc C: " << ret << ", ptr=" << cDevice << std::endl;
    
    // 拷贝数据
    ret = aclrtMemcpy(aDevice, aSize, aHost.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    std::cout << "memcpy A: " << ret << std::endl;
    
    ret = aclrtMemcpy(bDevice, bSize, bHost.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    std::cout << "memcpy B: " << ret << std::endl;
    
    ret = aclrtMemset(cDevice, cSize, 0, cSize);
    std::cout << "memset C: " << ret << std::endl;
    
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
    
    std::cout << "Tensors created: A=" << aTensor << " B=" << bTensor << " C=" << cTensor << std::endl;
    
    // 调用算子
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    std::cout << "\n--- Calling GetWorkspaceSize ---\n";
    aclnnStatus wsRet = aclnnMatmulCustomGetWorkspaceSize(aTensor, bTensor, cTensor,
                                                          &workspaceSize, &executor);
    std::cout << "GetWorkspaceSize ret: " << wsRet << ", workspace: " << workspaceSize << std::endl;
    
    errMsg = aclGetRecentErrMsg();
    if (errMsg && strlen(errMsg) > 0) {
        std::cout << "Error after GetWorkspaceSize: " << errMsg << std::endl;
    }
    
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    }
    
    std::cout << "\n--- Calling MatmulCustom ---\n";
    aclnnStatus execRet = aclnnMatmulCustom(workspace, workspaceSize, executor, stream);
    std::cout << "MatmulCustom ret: " << execRet << std::endl;
    
    errMsg = aclGetRecentErrMsg();
    if (errMsg && strlen(errMsg) > 0) {
        std::cout << "Error after MatmulCustom: " << errMsg << std::endl;
    }
    
    std::cout << "\n--- Synchronizing ---\n";
    ret = aclrtSynchronizeStream(stream);
    std::cout << "Synchronize ret: " << ret << " (0x" << std::hex << ret << std::dec << ")" << std::endl;
    
    errMsg = aclGetRecentErrMsg();
    if (errMsg && strlen(errMsg) > 0) {
        std::cout << "Error after Sync: " << errMsg << std::endl;
    }
    
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
    
    return 0;
}
