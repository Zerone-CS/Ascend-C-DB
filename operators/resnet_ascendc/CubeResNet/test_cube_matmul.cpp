/*
 * Test Cube-based MatMul on Ascend 910B
 * Using ACT library for high-performance matrix multiplication
 */

#include <iostream>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <acl/acl.h>

#include "tiling/platform/platform_ascendc.h"
#include "include/matmul/block/block_scheduler_policy.h"
#include "include/matmul/block/block_mmad_builder.h"
#include "include/matmul/kernel/kernel_matmul.h"
#include "include/matmul/device/device_matmul.h"
#include "include/utils/host_utils.h"
#include "include/utils/layout_utils.h"
#include "include/utils/status_utils.h"

using namespace Act;
using namespace Act::Gemm;

#define CHECK_ACL(x) do { \
    aclError err = (x); \
    if (err != ACL_SUCCESS) { \
        std::cerr << "ACL error: " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// L1 and L0 TileShape for 910B
using L1TileShape = AscendC::Shape<_128, _256, _256>;
using L0TileShape = AscendC::Shape<_128, _256, _64>;

using AType = half;
using BType = half;
using CType = half;

using LayoutA = layout::RowMajor;
using LayoutB = layout::RowMajor;
using LayoutC = layout::RowMajor;

using BlockScheduler = IterateKScheduler;

using BlockMmad = Block::BlockMmadBuilder<
    AType, LayoutA, BType, LayoutB, CType, LayoutC, CType, LayoutC,
    L1TileShape, L0TileShape, BlockScheduler, MatmulMultiBlockWithLayout<>>;

using BlockEpilogue = Block::BlockEpilogueEmpty;
using ProblemShape = MatmulShape;

using MatmulKernel = Kernel::KernelMatmul<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
using Arguments = typename MatmulKernel::Arguments;
using DeviceMatmul = Device::DeviceMatmul<MatmulKernel>;

bool CubeMatMulOp(uint8_t* A, uint8_t* B, uint8_t* C,
                  int64_t M, int64_t N, int64_t K) {
    uint8_t* workspaceDevice;
    MatmulShape shape{M, N, K, 1};

    Arguments args = {
        shape,
        {A, B, C, nullptr},
        {}
    };

    DeviceMatmul mm;
    size_t workspaceSize = DeviceMatmul::GetWorkspaceSize(args);

    CHECK_ACL(aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    Status status = mm.CanImplement(args);
    if (status != Status::success) {
        std::cerr << "Cannot implement matmul: " << GetStatusString(status);
        aclrtFree(workspaceDevice);
        return false;
    }

    mm.InitParams(args, workspaceDevice);
    mm();

    CHECK_ACL(aclrtFree(workspaceDevice));
    return true;
}

int main(int argc, char* argv[]) {
    // Matrix dimensions
    int64_t M = 512, N = 512, K = 512;
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Cube-based MatMul Test on Ascend 910B" << std::endl;
    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "========================================" << std::endl;

    // Initialize ACL
    int64_t deviceId = 2;  // Use device 2 as specified
    aclrtContext context;
    aclrtStream stream;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    std::cout << "ACL initialized on device " << deviceId << std::endl;

    // Allocate memory
    size_t sizeA = M * K * sizeof(half);
    size_t sizeB = K * N * sizeof(half);
    size_t sizeC = M * N * sizeof(half);

    half *hostA, *hostB, *hostC;
    uint8_t *devA, *devB, *devC;

    CHECK_ACL(aclrtMallocHost((void**)&hostA, sizeA));
    CHECK_ACL(aclrtMallocHost((void**)&hostB, sizeB));
    CHECK_ACL(aclrtMallocHost((void**)&hostC, sizeC));

    CHECK_ACL(aclrtMalloc((void**)&devA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&devB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&devC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    // Initialize matrices
    std::cout << "Initializing matrices..." << std::endl;
    for (int64_t i = 0; i < M * K; i++) {
        hostA[i] = (half)(0.01f * (i % 100));
    }
    for (int64_t i = 0; i < K * N; i++) {
        hostB[i] = (half)(0.01f * ((i + 50) % 100));
    }

    CHECK_ACL(aclrtMemcpy(devA, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(devB, sizeB, hostB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    // Warmup
    std::cout << "Warmup run..." << std::endl;
    if (!CubeMatMulOp(devA, devB, devC, M, N, K)) {
        std::cerr << "Warmup failed" << std::endl;
        return 1;
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // Benchmark
    int numIters = 10;
    std::cout << "Running " << numIters << " iterations..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIters; i++) {
        CubeMatMulOp(devA, devB, devC, M, N, K);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));
    auto end = std::chrono::high_resolution_clock::now();

    double totalMs = std::chrono::duration<double, std::milli>(end - start).count();
    double avgMs = totalMs / numIters;

    // Calculate TFLOPS
    double flops = 2.0 * M * N * K;  // MatMul has 2*M*N*K FLOPs
    double tflops = (flops / avgMs) / 1e9;  // TFLOPS

    // Get result
    CHECK_ACL(aclrtMemcpy(hostC, sizeC, devC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    // Verify a few elements
    std::cout << "\n========================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "Average time: " << avgMs << " ms" << std::endl;
    std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;
    std::cout << "\nOutput sample (first 5 elements):" << std::endl;
    for (int i = 0; i < 5 && i < M * N; i++) {
        std::cout << "  C[" << i << "] = " << (float)hostC[i] << std::endl;
    }
    std::cout << "========================================" << std::endl;

    // Cleanup
    aclrtFree(devA);
    aclrtFree(devB);
    aclrtFree(devC);
    aclrtFreeHost(hostA);
    aclrtFreeHost(hostB);
    aclrtFreeHost(hostC);

    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
