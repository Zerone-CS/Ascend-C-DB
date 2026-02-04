/**
 * Conv2D 卷积算子 - AlexNet 核心算子
 *
 * 实现方式: im2col + MatMul
 * 将卷积操作转换为矩阵乘法，充分利用 Cube 单元
 *
 * 输入: x [N, C, H, W] - NCHW 格式
 * 权重: weight [out_channels, in_channels, kH, kW]
 * 输出: y [N, out_channels, oH, oW]
 *
 * oH = (H + 2*padH - kH) / strideH + 1
 * oW = (W + 2*padW - kW) / strideW + 1
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelConv2D {
public:
    __aicore__ inline KernelConv2D() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                                uint32_t batch, uint32_t inChannels, uint32_t inH, uint32_t inW,
                                uint32_t outChannels, uint32_t kH, uint32_t kW,
                                uint32_t strideH, uint32_t strideW,
                                uint32_t padH, uint32_t padW) {
        this->batch = batch;
        this->inChannels = inChannels;
        this->inH = inH;
        this->inW = inW;
        this->outChannels = outChannels;
        this->kH = kH;
        this->kW = kW;
        this->strideH = strideH;
        this->strideW = strideW;
        this->padH = padH;
        this->padW = padW;

        // 计算输出尺寸
        this->outH = (inH + 2 * padH - kH) / strideH + 1;
        this->outW = (inW + 2 * padW - kW) / strideW + 1;

        // im2col 后的矩阵大小
        // colMatrix: [outH * outW, inChannels * kH * kW]
        this->colRows = outH * outW;
        this->colCols = inChannels * kH * kW;

        // 设置全局内存
        xGm.SetGlobalBuffer((__gm__ float*)x, batch * inChannels * inH * inW);
        weightGm.SetGlobalBuffer((__gm__ float*)weight, outChannels * colCols);
        yGm.SetGlobalBuffer((__gm__ float*)y, batch * outChannels * outH * outW);

        // 初始化缓冲区
        uint32_t colMatrixSize = colRows * colCols;
        uint32_t outputSize = outChannels * colRows;
        
        // 根据可用 UB 大小调整 tile
        pipe.InitBuffer(colBuffer, colMatrixSize * sizeof(float));
        pipe.InitBuffer(weightBuffer, outChannels * colCols * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, outputSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        // 加载权重到本地内存
        LoadWeight();

        // 对每个 batch 处理
        for (uint32_t n = 0; n < batch; n++) {
            Im2Col(n);
            MatMul();
            CopyOut(n);
        }
    }

private:
    __aicore__ inline void LoadWeight() {
        LocalTensor<float> weightLocal = weightBuffer.Get<float>();
        DataCopy(weightLocal, weightGm, outChannels * colCols);
    }

    __aicore__ inline void Im2Col(uint32_t batchIdx) {
        LocalTensor<float> colLocal = colBuffer.Get<float>();
        uint32_t inputOffset = batchIdx * inChannels * inH * inW;

        // 对每个输出位置
        for (uint32_t oh = 0; oh < outH; oh++) {
            for (uint32_t ow = 0; ow < outW; ow++) {
                uint32_t colRowIdx = oh * outW + ow;

                // 计算输入起始位置
                int32_t ihStart = (int32_t)(oh * strideH) - (int32_t)padH;
                int32_t iwStart = (int32_t)(ow * strideW) - (int32_t)padW;

                // 对每个通道和卷积核位置
                for (uint32_t ic = 0; ic < inChannels; ic++) {
                    for (uint32_t kh = 0; kh < kH; kh++) {
                        for (uint32_t kw = 0; kw < kW; kw++) {
                            int32_t ih = ihStart + kh;
                            int32_t iw = iwStart + kw;

                            uint32_t colColIdx = ic * kH * kW + kh * kW + kw;
                            uint32_t colIdx = colRowIdx * colCols + colColIdx;

                            float val = 0.0f;
                            if (ih >= 0 && ih < (int32_t)inH && iw >= 0 && iw < (int32_t)inW) {
                                uint32_t xIdx = inputOffset + ic * inH * inW + ih * inW + iw;
                                val = xGm.GetValue(xIdx);
                            }
                            colLocal.SetValue(colIdx, val);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void MatMul() {
        // output[oc, oh*ow] = weight[oc, colCols] @ col[oh*ow, colCols]^T
        // 简化实现：使用向量点积
        LocalTensor<float> colLocal = colBuffer.Get<float>();
        LocalTensor<float> weightLocal = weightBuffer.Get<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        for (uint32_t oc = 0; oc < outChannels; oc++) {
            for (uint32_t pos = 0; pos < colRows; pos++) {
                float sum = 0.0f;
                // 点积计算
                for (uint32_t k = 0; k < colCols; k++) {
                    float w = weightLocal.GetValue(oc * colCols + k);
                    float x = colLocal.GetValue(pos * colCols + k);
                    sum += w * x;
                }
                yLocal.SetValue(oc * colRows + pos, sum);
            }
        }

        outQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t batchIdx) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        uint32_t outputOffset = batchIdx * outChannels * outH * outW;
        DataCopy(yGm[outputOffset], yLocal, outChannels * outH * outW);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> colBuffer, weightBuffer;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm, weightGm, yGm;

    uint32_t batch, inChannels, inH, inW;
    uint32_t outChannels, kH, kW;
    uint32_t strideH, strideW, padH, padW;
    uint32_t outH, outW;
    uint32_t colRows, colCols;
};

extern "C" __global__ __aicore__ void conv2d_kernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y,
    uint32_t batch, uint32_t inChannels, uint32_t inH, uint32_t inW,
    uint32_t outChannels, uint32_t kH, uint32_t kW,
    uint32_t strideH, uint32_t strideW,
    uint32_t padH, uint32_t padW) {
    KernelConv2D op;
    op.Init(x, weight, y, batch, inChannels, inH, inW,
            outChannels, kH, kW, strideH, strideW, padH, padW);
    op.Process();
}
