/**
 * AscendC MaxPool2D Kernel
 * 2D 滑动窗口池化，输出每个窗口的最大值
 * 
 * 输入: x [N, C, H, W]
 * 输出: y [N, C, outH, outW]
 * outH = (H + 2*pad - K) / stride + 1
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;  // 简化实现用单缓冲

class KernelMaxPool2D {
public:
    __aicore__ inline KernelMaxPool2D() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t N, uint32_t C, uint32_t H, uint32_t W,
                                 uint32_t K, uint32_t stride, uint32_t pad) {
        this->N = N;
        this->C = C;
        this->H = H;
        this->W = W;
        this->K = K;
        this->stride = stride;
        this->pad = pad;
        this->outH = (H + 2 * pad - K) / stride + 1;
        this->outW = (W + 2 * pad - K) / stride + 1;

        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        
        // 每个 block 处理若干个 channel
        uint32_t totalChannels = N * C;
        uint32_t channelsPerBlock = (totalChannels + blockNum - 1) / blockNum;
        uint32_t startChannel = blockIdx * channelsPerBlock;
        uint32_t endChannel = (startChannel + channelsPerBlock > totalChannels) 
                              ? totalChannels : (startChannel + channelsPerBlock);
        this->myStartChannel = startChannel;
        this->myEndChannel = endChannel;

        xGm.SetGlobalBuffer((__gm__ float*)x, N * C * H * W);
        yGm.SetGlobalBuffer((__gm__ float*)y, N * C * outH * outW);

        // 用于读取输入窗口
        pipe.InitBuffer(inQueue, BUFFER_NUM, K * W * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t ch = myStartChannel; ch < myEndChannel; ch++) {
            ProcessChannel(ch);
        }
    }

private:
    __aicore__ inline void ProcessChannel(uint32_t ch) {
        uint32_t inOffset = ch * H * W;
        uint32_t outOffset = ch * outH * outW;

        // 遍历每个输出位置
        for (uint32_t oh = 0; oh < outH; oh++) {
            for (uint32_t ow = 0; ow < outW; ow++) {
                float maxVal = -1e30f;
                
                // 遍历池化窗口
                for (uint32_t kh = 0; kh < K; kh++) {
                    for (uint32_t kw = 0; kw < K; kw++) {
                        int32_t ih = (int32_t)(oh * stride + kh) - (int32_t)pad;
                        int32_t iw = (int32_t)(ow * stride + kw) - (int32_t)pad;
                        
                        if (ih >= 0 && ih < (int32_t)H && iw >= 0 && iw < (int32_t)W) {
                            float val = xGm.GetValue(inOffset + ih * W + iw);
                            if (val > maxVal) maxVal = val;
                        }
                    }
                }
                
                yGm.SetValue(outOffset + oh * outW + ow, maxVal);
            }
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    GlobalTensor<float> xGm, yGm;
    uint32_t N, C, H, W, K, stride, pad;
    uint32_t outH, outW;
    uint32_t myStartChannel, myEndChannel;
};

extern "C" __global__ __aicore__ void maxpool_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelMaxPool2D op;
    op.Init(x, y, tilingData.N, tilingData.C, tilingData.H, tilingData.W,
            tilingData.K, tilingData.stride, tilingData.pad);
    op.Process();
}
