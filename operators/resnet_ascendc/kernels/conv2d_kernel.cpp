/**
 * AscendC Conv2D Kernel (Naive Implementation)
 * 
 * 这是一个基于 Vector 单元的朴素实现
 * 生产环境应使用 Cube 单元或 aclnnConvolution
 * 
 * 输入: x [N, inC, H, W], weight [outC, inC, kH, kW]
 * 输出: y [N, outC, outH, outW]
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

class KernelConv2D {
public:
    __aicore__ inline KernelConv2D() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                                 uint32_t N, uint32_t inC, uint32_t H, uint32_t W,
                                 uint32_t outC, uint32_t kH, uint32_t kW,
                                 uint32_t stride, uint32_t pad) {
        this->N = N;
        this->inC = inC;
        this->H = H;
        this->W = W;
        this->outC = outC;
        this->kH = kH;
        this->kW = kW;
        this->stride = stride;
        this->pad = pad;
        this->outH = (H + 2 * pad - kH) / stride + 1;
        this->outW = (W + 2 * pad - kW) / stride + 1;

        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        
        // 每个 block 处理若干个输出 channel
        uint32_t channelsPerBlock = (outC + blockNum - 1) / blockNum;
        uint32_t startOC = blockIdx * channelsPerBlock;
        uint32_t endOC = (startOC + channelsPerBlock > outC) ? outC : (startOC + channelsPerBlock);
        this->myStartOC = startOC;
        this->myEndOC = endOC;

        xGm.SetGlobalBuffer((__gm__ float*)x, N * inC * H * W);
        wGm.SetGlobalBuffer((__gm__ float*)weight, outC * inC * kH * kW);
        yGm.SetGlobalBuffer((__gm__ float*)y, N * outC * outH * outW);
    }

    __aicore__ inline void Process() {
        // 遍历 batch
        for (uint32_t n = 0; n < N; n++) {
            // 遍历输出 channel (本 block 负责的部分)
            for (uint32_t oc = myStartOC; oc < myEndOC; oc++) {
                ProcessOutputChannel(n, oc);
            }
        }
    }

private:
    __aicore__ inline void ProcessOutputChannel(uint32_t n, uint32_t oc) {
        // 遍历输出位置
        for (uint32_t oh = 0; oh < outH; oh++) {
            for (uint32_t ow = 0; ow < outW; ow++) {
                float sum = 0.0f;
                
                // 遍历输入 channel
                for (uint32_t ic = 0; ic < inC; ic++) {
                    // 遍历卷积核
                    for (uint32_t kh = 0; kh < this->kH; kh++) {
                        for (uint32_t kw = 0; kw < this->kW; kw++) {
                            int32_t ih = (int32_t)(oh * stride + kh) - (int32_t)pad;
                            int32_t iw = (int32_t)(ow * stride + kw) - (int32_t)pad;
                            
                            if (ih >= 0 && ih < (int32_t)H && iw >= 0 && iw < (int32_t)W) {
                                uint32_t xIdx = n * inC * H * W + ic * H * W + ih * W + iw;
                                uint32_t wIdx = oc * inC * this->kH * this->kW + ic * this->kH * this->kW + kh * this->kW + kw;
                                sum += xGm.GetValue(xIdx) * wGm.GetValue(wIdx);
                            }
                        }
                    }
                }
                
                uint32_t yIdx = n * outC * outH * outW + oc * outH * outW + oh * outW + ow;
                yGm.SetValue(yIdx, sum);
            }
        }
    }

private:
    GlobalTensor<float> xGm, wGm, yGm;
    uint32_t N, inC, H, W, outC, kH, kW, stride, pad;
    uint32_t outH, outW;
    uint32_t myStartOC, myEndOC;
};

extern "C" __global__ __aicore__ void conv2d_kernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelConv2D op;
    op.Init(x, weight, y, 
            tilingData.N, tilingData.inC, tilingData.H, tilingData.W,
            tilingData.outC, tilingData.kH, tilingData.kW,
            tilingData.stride, tilingData.pad);
    op.Process();
}
