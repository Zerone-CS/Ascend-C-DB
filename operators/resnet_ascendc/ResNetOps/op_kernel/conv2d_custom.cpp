/**
 * AscendC Conv2D Kernel - 直接卷积实现
 * 输入: x [N, inC, H, W], weight [outC, inC, kH, kW]
 * 输出: y [N, outC, outH, outW]
 * outH = (H + 2*pad - kH) / stride + 1
 */
#include "kernel_operator.h"
using namespace AscendC;

class KernelConv2d {
public:
    __aicore__ inline KernelConv2d() {}

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
        for (uint32_t n = 0; n < N; n++) {
            for (uint32_t oc = myStartOC; oc < myEndOC; oc++) {
                ProcessOutputChannel(n, oc);
            }
        }
    }

private:
    __aicore__ inline void ProcessOutputChannel(uint32_t n, uint32_t oc) {
        for (uint32_t oh = 0; oh < outH; oh++) {
            for (uint32_t ow = 0; ow < outW; ow++) {
                float sum = 0.0f;
                for (uint32_t ic = 0; ic < inC; ic++) {
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

extern "C" __global__ __aicore__ void conv2d_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelConv2d op;
    op.Init(x, weight, y,
            tilingData.N, tilingData.inC, tilingData.H, tilingData.W,
            tilingData.outC, tilingData.kH, tilingData.kW,
            tilingData.stride, tilingData.pad);
    op.Process();
}
