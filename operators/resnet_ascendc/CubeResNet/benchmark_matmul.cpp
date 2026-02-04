/*
 * Benchmark: Cube MatMul vs Vector-based MatMul
 * Compare performance on Ascend 910B
 */

#include <iostream>
#include <cstdint>
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
        std::cerr << "ACL error: " << err << std::endl; \
        exit(1); \
    } \
} while(0)

// Cube MatMul setup
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
using MK = Kernel::KernelMatmul<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
using MArgs = typename MK::Arguments;
using DM = Device::DeviceMatmul<MK>;

void CubeMatMul(uint8_t* A, uint8_t* B, uint8_t* C, int64_t M, int64_t N, int64_t K) {
    uint8_t* ws;
    MatmulShape shape{M, N, K, 1};
    MArgs args = {shape, {A, B, C, nullptr}, {}};
    DM mm;
    size_t wsz = DM::GetWorkspaceSize(args);
    aclrtMalloc((void**)&ws, wsz, ACL_MEM_MALLOC_HUGE_FIRST);
    if (mm.CanImplement(args) == Status::success) {
        mm.InitParams(args, ws);
        mm();
    }
    aclrtFree(ws);
}

// Vector-based naive MatMul (CPU simulation)
void NaiveMatMulCPU(half* A, half* B, half* C, int64_t M, int64_t N, int64_t K) {
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += (float)A[m * K + k] * (float)B[k * N + n];
            }
            C[m * N + n] = (half)sum;
        }
    }
}

struct BenchResult {
    double timeMs;
    double tflops;
};

BenchResult BenchmarkCube(int64_t M, int64_t N, int64_t K, int iters) {
    uint8_t *dA, *dB, *dC;
    aclrtMalloc((void**)&dA, M * K * sizeof(half), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dB, K * N * sizeof(half), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dC, M * N * sizeof(half), ACL_MEM_MALLOC_HUGE_FIRST);

    // Warmup
    CubeMatMul(dA, dB, dC, M, N, K);
    aclrtSynchronizeDevice();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        CubeMatMul(dA, dB, dC, M, N, K);
    }
    aclrtSynchronizeDevice();
    auto end = std::chrono::high_resolution_clock::now();

    double totalMs = std::chrono::duration<double, std::milli>(end - start).count();
    double avgMs = totalMs / iters;
    double flops = 2.0 * M * N * K;
    double tflops = (flops / avgMs) / 1e9;

    aclrtFree(dA); aclrtFree(dB); aclrtFree(dC);
    return {avgMs, tflops};
}

BenchResult BenchmarkNaiveCPU(int64_t M, int64_t N, int64_t K, int iters) {
    half *hA, *hB, *hC;
    aclrtMallocHost((void**)&hA, M * K * sizeof(half));
    aclrtMallocHost((void**)&hB, K * N * sizeof(half));
    aclrtMallocHost((void**)&hC, M * N * sizeof(half));

    // Init
    for (int64_t i = 0; i < M * K; i++) hA[i] = (half)(0.01f * (i % 100));
    for (int64_t i = 0; i < K * N; i++) hB[i] = (half)(0.01f * ((i + 50) % 100));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        NaiveMatMulCPU(hA, hB, hC, M, N, K);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double totalMs = std::chrono::duration<double, std::milli>(end - start).count();
    double avgMs = totalMs / iters;
    double flops = 2.0 * M * N * K;
    double tflops = (flops / avgMs) / 1e9;

    aclrtFreeHost(hA); aclrtFreeHost(hB); aclrtFreeHost(hC);
    return {avgMs, tflops};
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "  MatMul Performance Benchmark: Cube vs Naive (CPU baseline)" << std::endl;
    std::cout << "  Device: Ascend 910B" << std::endl;
    std::cout << "============================================================\n" << std::endl;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(2));

    // Test sizes
    struct TestCase { int64_t M, N, K; };
    TestCase tests[] = {
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
    };

    std::cout << "Size (MxNxK)       | Cube (ms)   | Cube (TFLOPS) | Naive CPU (ms) | Speedup" << std::endl;
    std::cout << "-------------------|-------------|---------------|----------------|--------" << std::endl;

    for (auto& t : tests) {
        int cubeIters = 20;
        int cpuIters = (t.M <= 256) ? 3 : 1;  // CPU is slow for large sizes

        auto cubeRes = BenchmarkCube(t.M, t.N, t.K, cubeIters);

        // Only run CPU benchmark for small sizes
        double cpuMs = 0, speedup = 0;
        if (t.M <= 512) {
            auto cpuRes = BenchmarkNaiveCPU(t.M, t.N, t.K, cpuIters);
            cpuMs = cpuRes.timeMs;
            speedup = cpuMs / cubeRes.timeMs;
        }

        printf("%4ldx%4ldx%4ld     | %10.4f  | %13.2f | ", t.M, t.N, t.K, cubeRes.timeMs, cubeRes.tflops);
        if (cpuMs > 0) {
            printf("%14.2f | %6.1fx\n", cpuMs, speedup);
        } else {
            printf("       (skip)    | N/A\n");
        }
    }

    std::cout << "\n============================================================" << std::endl;
    std::cout << "Note: Cube unit uses ACT library's optimized MatMul kernel" << std::endl;
    std::cout << "      CPU baseline is triple-nested loop (extremely slow)" << std::endl;
    std::cout << "============================================================" << std::endl;

    aclrtResetDevice(2);
    aclFinalize();
    return 0;
}
