/**
 * MaxPool2D 最大池化算子 - AlexNet 池化层
 *
 * 输入: x [N, C, H, W] - NCHW 格式
 * 输出: y [N, C, oH, oW]
 *
 * oH = (H - kH) / strideH + 1
 * oW = (W - kW) / strideW + 1
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelMaxPool2D {
public:
    __aicore__ inline KernelMaxPool2D() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t batch, uint32_t channels,
                                uint32_t inH, uint32_t inW,
                                uint32_t kH, uint32_t kW,
                                uint32_t strideH, uint32_t strideW) {
        this->batch = batch;
        this->channels = channels;
        this->inH = inH;
        this->inW = inW;
        this->kH = kH;
        this->kW = kW;
        this->strideH = strideH;
        this->strideW = strideW;

        // 计算输出尺寸
        this->outH = (inH - kH) / strideH + 1;
        this->outW = (inW - kW) / strideW + 1;

        xGm.SetGlobalBuffer((__gm__ float*)x, batch * channels * inH * inW);
        yGm.SetGlobalBuffer((__gm__ float*)y, batch * channels * outH * outW);

        // 分配缓冲区
        uint32_t poolWindowSize = kH * kW;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, poolWindowSize * sizeof(float));
        pipe.InitBuffer(workBuffer, poolWindowSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        // 遍历所有 batch 和 channel
        for (uint32_t n = 0; n < batch; n++) {
            for (uint32_t c = 0; c < channels; c++) {
                ProcessChannel(n, c);
            }
        }
    }

private:
    __aicore__ inline void ProcessChannel(uint32_t n, uint32_t c) {
        uint32_t inputOffset = (n * channels + c) * inH * inW;
        uint32_t outputOffset = (n * channels + c) * outH * outW;

        for (uint32_t oh = 0; oh < outH; oh++) {
            for (uint32_t ow = 0; ow < outW; ow++) {
                // 收集池化窗口内的值
                LocalTensor<float> poolLocal = inQueueX.AllocTensor<float>();
                LocalTensor<float> workLocal = workBuffer.Get<float>();

                uint32_t ihStart = oh * strideH;
                uint32_t iwStart = ow * strideW;

                uint32_t idx = 0;
                for (uint32_t kh = 0; kh < kH; kh++) {
                    for (uint32_t kw = 0; kw < kW; kw++) {
                        uint32_t ih = ihStart + kh;
                        uint32_t iw = iwStart + kw;
                        uint32_t xIdx = inputOffset + ih * inW + iw;
                        float val = xGm.GetValue(xIdx);
                        poolLocal.SetValue(idx++, val);
                    }
                }

                // 计算最大值
                uint32_t poolSize = kH * kW;
                ReduceMax(workLocal, poolLocal, workLocal, poolSize);
                float maxVal = workLocal.GetValue(0);

                // 写入输出
                uint32_t yIdx = outputOffset + oh * outW + ow;
                yGm.SetValue(yIdx, maxVal);

                inQueueX.FreeTensor(poolLocal);
            }
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TBuf<QuePosition::VECCALC> workBuffer;
    GlobalTensor<float> xGm, yGm;

    uint32_t batch, channels, inH, inW;
    uint32_t kH, kW, strideH, strideW;
    uint32_t outH, outW;
};

extern "C" __global__ __aicore__ void maxpool2d_kernel(
    GM_ADDR x, GM_ADDR y,
    uint32_t batch, uint32_t channels,
    uint32_t inH, uint32_t inW,
    uint32_t kH, uint32_t kW,
    uint32_t strideH, uint32_t strideW) {
    KernelMaxPool2D op;
    op.Init(x, y, batch, channels, inH, inW, kH, kW, strideH, strideW);
    op.Process();
}
