/*
 * Complete ResNet-18 using Cube Unit on Ascend 910B
 * Conv2D: im2col + Cube MatMul
 * FC: Cube MatMul
 * Auxiliary ops (ReLU, BN, Pool): host-side computation
 */

#include <iostream>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
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

// ============================================================================
// Cube MatMul Setup
// ============================================================================
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
    CHECK_ACL(aclrtMalloc((void**)&ws, wsz, ACL_MEM_MALLOC_HUGE_FIRST));
    if (mm.CanImplement(args) == Status::success) {
        mm.InitParams(args, ws);
        mm();
    }
    CHECK_ACL(aclrtFree(ws));
}

// ============================================================================
// Helper: random init
// ============================================================================
std::mt19937 g_gen(42);
std::normal_distribution<float> g_dist(0.0f, 0.02f);

half* AllocHost(int64_t count) {
    half* p;
    CHECK_ACL(aclrtMallocHost((void**)&p, count * sizeof(half)));
    return p;
}

void InitRandom(half* p, int64_t count) {
    for (int64_t i = 0; i < count; i++) p[i] = (half)g_dist(g_gen);
}

void InitOnes(half* p, int64_t count) {
    for (int64_t i = 0; i < count; i++) p[i] = (half)1.0f;
}

void InitZeros(half* p, int64_t count) {
    for (int64_t i = 0; i < count; i++) p[i] = (half)0.0f;
}

// ============================================================================
// im2col / col2im
// ============================================================================
void Im2Col(const half* in, half* out, int N, int C, int H, int W,
            int kH, int kW, int padH, int padW, int sH, int sW) {
    int oH = (H + 2*padH - kH) / sH + 1;
    int oW = (W + 2*padW - kW) / sW + 1;
    int colW = C * kH * kW;
    for (int n = 0; n < N; n++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++) {
                int row = n * oH * oW + oh * oW + ow;
                for (int c = 0; c < C; c++)
                    for (int fh = 0; fh < kH; fh++)
                        for (int fw = 0; fw < kW; fw++) {
                            int ih = oh * sH - padH + fh;
                            int iw = ow * sW - padW + fw;
                            int col = c * kH * kW + fh * kW + fw;
                            out[row * colW + col] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                ? in[n*C*H*W + c*H*W + ih*W + iw] : (half)0.0f;
                        }
            }
}

void Col2Im(const half* in, half* out, int N, int K, int oH, int oW) {
    for (int n = 0; n < N; n++)
        for (int oh = 0; oh < oH; oh++)
            for (int ow = 0; ow < oW; ow++) {
                int row = n * oH * oW + oh * oW + ow;
                for (int k = 0; k < K; k++)
                    out[n*K*oH*oW + k*oH*oW + oh*oW + ow] = in[row * K + k];
            }
}

// ============================================================================
// Cube Conv2D
// ============================================================================
void CubeConv2D(half* input, half* weight, half* output,
                int N, int C, int H, int W,
                int K, int kH, int kW,
                int padH, int padW, int sH, int sW) {
    int oH = (H + 2*padH - kH) / sH + 1;
    int oW = (W + 2*padW - kW) / sW + 1;
    int M = N * oH * oW;
    int KK = C * kH * kW;
    int NN = K;

    half* im2colH = AllocHost(M * KK);
    Im2Col(input, im2colH, N, C, H, W, kH, kW, padH, padW, sH, sW);

    uint8_t *im2colD, *weightD, *outD;
    CHECK_ACL(aclrtMalloc((void**)&im2colD, M * KK * sizeof(half), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&weightD, NN * KK * sizeof(half), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&outD, M * NN * sizeof(half), ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(im2colD, M*KK*sizeof(half), im2colH, M*KK*sizeof(half), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(weightD, NN*KK*sizeof(half), weight, NN*KK*sizeof(half), ACL_MEMCPY_HOST_TO_DEVICE));

    CubeMatMul(im2colD, weightD, outD, M, NN, KK);

    half* outH = AllocHost(M * NN);
    CHECK_ACL(aclrtMemcpy(outH, M*NN*sizeof(half), outD, M*NN*sizeof(half), ACL_MEMCPY_DEVICE_TO_HOST));
    Col2Im(outH, output, N, K, oH, oW);

    aclrtFreeHost(im2colH);
    aclrtFreeHost(outH);
    aclrtFree(im2colD); aclrtFree(weightD); aclrtFree(outD);
}

// ============================================================================
// Cube FC Layer
// ============================================================================
void CubeFC(half* input, half* weight, half* output, int M, int N, int K) {
    uint8_t *inD, *wD, *outD;
    CHECK_ACL(aclrtMalloc((void**)&inD, M * K * sizeof(half), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&wD, K * N * sizeof(half), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&outD, M * N * sizeof(half), ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(inD, M*K*sizeof(half), input, M*K*sizeof(half), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(wD, K*N*sizeof(half), weight, K*N*sizeof(half), ACL_MEMCPY_HOST_TO_DEVICE));

    CubeMatMul(inD, wD, outD, M, N, K);

    CHECK_ACL(aclrtMemcpy(output, M*N*sizeof(half), outD, M*N*sizeof(half), ACL_MEMCPY_DEVICE_TO_HOST));
    aclrtFree(inD); aclrtFree(wD); aclrtFree(outD);
}

// ============================================================================
// Auxiliary ops (host-side)
// ============================================================================
void ReLU(half* d, int sz) {
    for (int i = 0; i < sz; i++) if ((float)d[i] < 0) d[i] = (half)0.0f;
}

void BatchNorm(half* d, half* gamma, half* beta, int N, int C, int H, int W) {
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++) {
            float g = (float)gamma[c], b = (float)beta[c];
            for (int hw = 0; hw < H*W; hw++) {
                int idx = n*C*H*W + c*H*W + hw;
                d[idx] = (half)((float)d[idx] * g + b);
            }
        }
}

void MaxPool2D(half* in, half* out, int N, int C, int H, int W, int k, int s) {
    int oH = (H - k) / s + 1, oW = (W - k) / s + 1;
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int oh = 0; oh < oH; oh++)
                for (int ow = 0; ow < oW; ow++) {
                    float mx = -1e10f;
                    for (int fh = 0; fh < k; fh++)
                        for (int fw = 0; fw < k; fw++) {
                            float v = (float)in[n*C*H*W + c*H*W + (oh*s+fh)*W + ow*s+fw];
                            if (v > mx) mx = v;
                        }
                    out[n*C*oH*oW + c*oH*oW + oh*oW + ow] = (half)mx;
                }
}

void GlobalAvgPool(half* in, half* out, int N, int C, int H, int W) {
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++) {
            float sum = 0;
            for (int hw = 0; hw < H*W; hw++) sum += (float)in[n*C*H*W + c*H*W + hw];
            out[n*C + c] = (half)(sum / (H * W));
        }
}

void Add(half* a, half* b, half* c, int sz) {
    for (int i = 0; i < sz; i++) c[i] = (half)((float)a[i] + (float)b[i]);
}

// ============================================================================
// BasicBlock
// ============================================================================
struct BasicBlock {
    half *c1w, *b1g, *b1b, *c2w, *b2g, *b2b;
    half *dsw, *dsg, *dsb;
    int inC, outC, stride;
    bool hasDS;

    BasicBlock(int in, int out, int s) : inC(in), outC(out), stride(s) {
        hasDS = (s != 1 || in != out);
        c1w = AllocHost(out*in*9); InitRandom(c1w, out*in*9);
        b1g = AllocHost(out); InitOnes(b1g, out);
        b1b = AllocHost(out); InitZeros(b1b, out);
        c2w = AllocHost(out*out*9); InitRandom(c2w, out*out*9);
        b2g = AllocHost(out); InitOnes(b2g, out);
        b2b = AllocHost(out); InitZeros(b2b, out);
        if (hasDS) {
            dsw = AllocHost(out*in); InitRandom(dsw, out*in);
            dsg = AllocHost(out); InitOnes(dsg, out);
            dsb = AllocHost(out); InitZeros(dsb, out);
        }
    }

    void Forward(half* in, half* out, int N, int H, int W) {
        int oH = (H + 2 - 3) / stride + 1;
        int oW = (W + 2 - 3) / stride + 1;
        int sz1 = N*outC*oH*oW;

        half* t1 = AllocHost(sz1);
        half* t2 = AllocHost(sz1);
        half* id = AllocHost(sz1);

        CubeConv2D(in, c1w, t1, N, inC, H, W, outC, 3, 3, 1, 1, stride, stride);
        BatchNorm(t1, b1g, b1b, N, outC, oH, oW);
        ReLU(t1, sz1);

        CubeConv2D(t1, c2w, t2, N, outC, oH, oW, outC, 3, 3, 1, 1, 1, 1);
        BatchNorm(t2, b2g, b2b, N, outC, oH, oW);

        if (hasDS) {
            CubeConv2D(in, dsw, id, N, inC, H, W, outC, 1, 1, 0, 0, stride, stride);
            BatchNorm(id, dsg, dsb, N, outC, oH, oW);
        } else {
            std::memcpy(id, in, sz1 * sizeof(half));
        }

        Add(t2, id, out, sz1);
        ReLU(out, sz1);

        aclrtFreeHost(t1); aclrtFreeHost(t2); aclrtFreeHost(id);
    }

    ~BasicBlock() {
        aclrtFreeHost(c1w); aclrtFreeHost(b1g); aclrtFreeHost(b1b);
        aclrtFreeHost(c2w); aclrtFreeHost(b2g); aclrtFreeHost(b2b);
        if (hasDS) { aclrtFreeHost(dsw); aclrtFreeHost(dsg); aclrtFreeHost(dsb); }
    }
};

// ============================================================================
// ResNet-18
// ============================================================================
class ResNet18 {
public:
    // Conv1
    half *conv1_w, *bn1_g, *bn1_b;

    // Layer1-4: 2 blocks each
    BasicBlock *layer1_0, *layer1_1;
    BasicBlock *layer2_0, *layer2_1;
    BasicBlock *layer3_0, *layer3_1;
    BasicBlock *layer4_0, *layer4_1;

    // FC
    half *fc_w;
    int numClasses;

    ResNet18(int nClasses = 10) : numClasses(nClasses) {
        conv1_w = AllocHost(64*3*7*7); InitRandom(conv1_w, 64*3*7*7);
        bn1_g = AllocHost(64); InitOnes(bn1_g, 64);
        bn1_b = AllocHost(64); InitZeros(bn1_b, 64);

        layer1_0 = new BasicBlock(64, 64, 1);
        layer1_1 = new BasicBlock(64, 64, 1);
        layer2_0 = new BasicBlock(64, 128, 2);
        layer2_1 = new BasicBlock(128, 128, 1);
        layer3_0 = new BasicBlock(128, 256, 2);
        layer3_1 = new BasicBlock(256, 256, 1);
        layer4_0 = new BasicBlock(256, 512, 2);
        layer4_1 = new BasicBlock(512, 512, 1);

        fc_w = AllocHost(512 * nClasses); InitRandom(fc_w, 512 * nClasses);
    }

    void Forward(half* input, half* output, int N) {
        int H = 224, W = 224;

        std::cout << "  [Conv1] 3x224x224 -> 64x112x112" << std::endl;
        int oH1 = (H + 2*3 - 7) / 2 + 1; // = 112
        int oW1 = (W + 2*3 - 7) / 2 + 1;
        half* c1 = AllocHost(N*64*oH1*oW1);
        CubeConv2D(input, conv1_w, c1, N, 3, H, W, 64, 7, 7, 3, 3, 2, 2);
        BatchNorm(c1, bn1_g, bn1_b, N, 64, oH1, oW1);
        ReLU(c1, N*64*oH1*oW1);

        std::cout << "  [MaxPool] 64x112x112 -> 64x56x56" << std::endl;
        int pH = (oH1 - 3) / 2 + 1; // = 56
        int pW = (oW1 - 3) / 2 + 1;
        half* p1 = AllocHost(N*64*pH*pW);
        MaxPool2D(c1, p1, N, 64, oH1, oW1, 3, 2);
        aclrtFreeHost(c1);

        std::cout << "  [Layer1] 64x56x56 -> 64x56x56" << std::endl;
        half* l1_0 = AllocHost(N*64*pH*pW);
        layer1_0->Forward(p1, l1_0, N, pH, pW);
        aclrtFreeHost(p1);
        half* l1_1 = AllocHost(N*64*pH*pW);
        layer1_1->Forward(l1_0, l1_1, N, pH, pW);
        aclrtFreeHost(l1_0);

        std::cout << "  [Layer2] 64x56x56 -> 128x28x28" << std::endl;
        int h2 = pH/2, w2 = pW/2; // 28
        half* l2_0 = AllocHost(N*128*h2*w2);
        layer2_0->Forward(l1_1, l2_0, N, pH, pW);
        aclrtFreeHost(l1_1);
        half* l2_1 = AllocHost(N*128*h2*w2);
        layer2_1->Forward(l2_0, l2_1, N, h2, w2);
        aclrtFreeHost(l2_0);

        std::cout << "  [Layer3] 128x28x28 -> 256x14x14" << std::endl;
        int h3 = h2/2, w3 = w2/2; // 14
        half* l3_0 = AllocHost(N*256*h3*w3);
        layer3_0->Forward(l2_1, l3_0, N, h2, w2);
        aclrtFreeHost(l2_1);
        half* l3_1 = AllocHost(N*256*h3*w3);
        layer3_1->Forward(l3_0, l3_1, N, h3, w3);
        aclrtFreeHost(l3_0);

        std::cout << "  [Layer4] 256x14x14 -> 512x7x7" << std::endl;
        int h4 = h3/2, w4 = w3/2; // 7
        half* l4_0 = AllocHost(N*512*h4*w4);
        layer4_0->Forward(l3_1, l4_0, N, h3, w3);
        aclrtFreeHost(l3_1);
        half* l4_1 = AllocHost(N*512*h4*w4);
        layer4_1->Forward(l4_0, l4_1, N, h4, w4);
        aclrtFreeHost(l4_0);

        std::cout << "  [AvgPool] 512x7x7 -> 512" << std::endl;
        half* gap = AllocHost(N*512);
        GlobalAvgPool(l4_1, gap, N, 512, h4, w4);
        aclrtFreeHost(l4_1);

        std::cout << "  [FC] 512 -> " << numClasses << " (Cube MatMul)" << std::endl;
        CubeFC(gap, fc_w, output, N, numClasses, 512);
        aclrtFreeHost(gap);
    }

    ~ResNet18() {
        aclrtFreeHost(conv1_w); aclrtFreeHost(bn1_g); aclrtFreeHost(bn1_b);
        delete layer1_0; delete layer1_1;
        delete layer2_0; delete layer2_1;
        delete layer3_0; delete layer3_1;
        delete layer4_0; delete layer4_1;
        aclrtFreeHost(fc_w);
    }
};

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "========================================================" << std::endl;
    std::cout << "  ResNet-18 on Ascend 910B using Cube Unit (ACT MatMul)  " << std::endl;
    std::cout << "========================================================" << std::endl;

    int64_t deviceId = 2;
    aclrtContext context;
    aclrtStream stream;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));
    std::cout << "ACL initialized on device " << deviceId << "\n" << std::endl;

    int N = 1, numClasses = 10;
    half* input = AllocHost(N * 3 * 224 * 224);
    half* output = AllocHost(N * numClasses);

    // Random input
    for (int i = 0; i < N * 3 * 224 * 224; i++) input[i] = (half)g_dist(g_gen);

    std::cout << "Input shape: " << N << "x3x224x224" << std::endl;
    std::cout << "\nForward pass:" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    ResNet18 resnet(numClasses);
    resnet.Forward(input, output, N);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    auto end = std::chrono::high_resolution_clock::now();

    double totalMs = std::chrono::duration<double, std::milli>(end - start).count();

    // Print logits
    std::cout << "\n========================================================" << std::endl;
    std::cout << "Output logits (" << numClasses << " classes):" << std::endl;
    float maxLogit = -1e10f;
    int maxIdx = 0;
    for (int i = 0; i < numClasses; i++) {
        float v = (float)output[i];
        std::cout << "  class " << i << ": " << v << std::endl;
        if (v > maxLogit) { maxLogit = v; maxIdx = i; }
    }
    std::cout << "\nPredicted class: " << maxIdx << " (logit=" << maxLogit << ")" << std::endl;
    std::cout << "Total time (incl. weight init): " << totalMs << " ms" << std::endl;
    std::cout << "========================================================" << std::endl;

    aclrtFreeHost(input);
    aclrtFreeHost(output);

    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "\nResNet-18 forward pass completed successfully!" << std::endl;
    return 0;
}
