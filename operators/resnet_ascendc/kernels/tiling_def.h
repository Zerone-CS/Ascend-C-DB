/**
 * Tiling 结构体定义 - 用于 Host 与 Kernel 之间传递参数
 */
#ifndef TILING_DEF_H
#define TILING_DEF_H

#include <cstdint>

// ReLU, Add 等简单算子的 tiling
struct SimpleTiling {
    uint32_t totalLength;
};

// BatchNorm tiling
struct BatchNormTiling {
    uint32_t N;
    uint32_t C;
    uint32_t HW;  // H * W
};

// MaxPool tiling
struct MaxPoolTiling {
    uint32_t N;
    uint32_t C;
    uint32_t H;
    uint32_t W;
    uint32_t K;       // kernel size
    uint32_t stride;
    uint32_t pad;
};

// Conv2D tiling
struct Conv2DTiling {
    uint32_t N;
    uint32_t inC;
    uint32_t H;
    uint32_t W;
    uint32_t outC;
    uint32_t kH;
    uint32_t kW;
    uint32_t stride;
    uint32_t pad;
};

// FC (MatMul) tiling
struct FCTiling {
    uint32_t M;
    uint32_t K;
    uint32_t N;
};

// GlobalAvgPool tiling
struct AvgPoolTiling {
    uint32_t N;
    uint32_t C;
    uint32_t HW;
};

#endif // TILING_DEF_H
