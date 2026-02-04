#pragma once

#include <cstdint>
#include <cstring>
#include <acl/acl.h>
#include "cube_matmul.h"

namespace CubeOps {

// Conv2D using im2col + Cube MatMul
// Input: NCHW format, Weight: KCHW format
// This is a high-performance implementation using Cube unit

// im2col transformation on host side (for simplicity)
// In production, this should be done on device
inline void Im2ColHalf(const half* input, half* output,
                       int N, int C, int H, int W,
                       int K, int kH, int kW,
                       int padH, int padW, int strideH, int strideW) {
    int outH = (H + 2 * padH - kH) / strideH + 1;
    int outW = (W + 2 * padW - kW) / strideW + 1;
    int colH = N * outH * outW;  // M dimension for matmul
    int colW = C * kH * kW;      // K dimension for matmul
    
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                int row = n * outH * outW + oh * outW + ow;
                for (int c = 0; c < C; c++) {
                    for (int fh = 0; fh < kH; fh++) {
                        for (int fw = 0; fw < kW; fw++) {
                            int ih = oh * strideH - padH + fh;
                            int iw = ow * strideW - padW + fw;
                            int col = c * kH * kW + fh * kW + fw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                output[row * colW + col] = input[n * C * H * W + c * H * W + ih * W + iw];
                            } else {
                                output[row * colW + col] = (half)0.0f;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Reshape weight from KCHW to (K, C*kH*kW)
inline void ReshapeWeightHalf(const half* weight, half* output,
                              int K, int C, int kH, int kW) {
    // Weight is already in correct format for matmul: K x (C*kH*kW)
    std::memcpy(output, weight, K * C * kH * kW * sizeof(half));
}

// Cube-based Conv2D
// Input: NxCxHxW, Weight: KxCxkHxkW, Output: NxKxoHxoW
class CubeConv2D {
public:
    static void Forward(uint8_t* input, uint8_t* weight, uint8_t* output,
                       int N, int C, int H, int W,
                       int K, int kH, int kW,
                       int padH, int padW, int strideH, int strideW,
                       void* stream = nullptr) {
        int outH = (H + 2 * padH - kH) / strideH + 1;
        int outW = (W + 2 * padW - kW) / strideW + 1;
        
        // Matrix dimensions for matmul
        int M = N * outH * outW;  // batch * output spatial
        int KK = C * kH * kW;     // input channels * kernel size
        int NN = K;               // output channels
        
        // Allocate im2col buffer on device
        uint8_t* im2colDevice;
        size_t im2colSize = M * KK * sizeof(half);
        aclrtMalloc((void**)&im2colDevice, im2colSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        // Allocate weight buffer on device (reshaped)
        uint8_t* weightDevice;
        size_t weightSize = NN * KK * sizeof(half);
        aclrtMalloc((void**)&weightDevice, weightSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        // Perform im2col on host and copy to device
        // (In production, use device-side im2col kernel)
        half* im2colHost;
        half* weightHost;
        aclrtMallocHost((void**)&im2colHost, im2colSize);
        aclrtMallocHost((void**)&weightHost, weightSize);
        
        // Get input to host for im2col
        half* inputHost;
        aclrtMallocHost((void**)&inputHost, N * C * H * W * sizeof(half));
        aclrtMemcpy(inputHost, N * C * H * W * sizeof(half), input, 
                    N * C * H * W * sizeof(half), ACL_MEMCPY_DEVICE_TO_HOST);
        
        // Get weight to host
        aclrtMemcpy(weightHost, weightSize, weight, weightSize, ACL_MEMCPY_DEVICE_TO_HOST);
        
        // Perform im2col
        Im2ColHalf(inputHost, im2colHost, N, C, H, W, K, kH, kW, padH, padW, strideH, strideW);
        
        // Copy to device
        aclrtMemcpy(im2colDevice, im2colSize, im2colHost, im2colSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(weightDevice, weightSize, weightHost, weightSize, ACL_MEMCPY_HOST_TO_DEVICE);
        
        // Perform Cube MatMul: (M x KK) * (KK x NN)^T -> (M x NN)
        // Actually weight is (NN x KK), so we compute: im2col * weight^T
        // Or equivalently: (weight * im2col^T)^T
        // For simplicity, use: C = im2col * weight^T
        CubeMatMul(im2colDevice, weightDevice, output, M, NN, KK, stream);
        
        // Cleanup
        aclrtFreeHost(inputHost);
        aclrtFreeHost(im2colHost);
        aclrtFreeHost(weightHost);
        aclrtFree(im2colDevice);
        aclrtFree(weightDevice);
    }
};

} // namespace CubeOps
