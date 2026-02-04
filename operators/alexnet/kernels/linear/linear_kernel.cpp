/**
 * Linear 全连接层 - AlexNet FC层
 *
 * y = x @ W^T + bias
 * x: [batch, in_features]
 * W: [out_features, in_features]
 * bias: [out_features]
 * y: [batch, out_features]
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelLinear {
public:
    __aicore__ inline KernelLinear() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                uint32_t batch, uint32_t inFeatures, uint32_t outFeatures,
                                uint32_t hasBias) {
        this->batch = batch;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        this->hasBias = hasBias;

        xGm.SetGlobalBuffer((__gm__ float*)x, batch * inFeatures);
        weightGm.SetGlobalBuffer((__gm__ float*)weight, outFeatures * inFeatures);
        if (hasBias) {
            biasGm.SetGlobalBuffer((__gm__ float*)bias, outFeatures);
        }
        yGm.SetGlobalBuffer((__gm__ float*)y, batch * outFeatures);

        // 初始化缓冲区
        pipe.InitBuffer(xBuffer, inFeatures * sizeof(float));
        pipe.InitBuffer(weightBuffer, outFeatures * inFeatures * sizeof(float));
        if (hasBias) {
            pipe.InitBuffer(biasBuffer, outFeatures * sizeof(float));
        }
        pipe.InitBuffer(outQueueY, BUFFER_NUM, outFeatures * sizeof(float));
    }

    __aicore__ inline void Process() {
        // 加载权重和偏置
        LoadWeightAndBias();

        // 处理每个 batch
        for (uint32_t n = 0; n < batch; n++) {
            ProcessBatch(n);
        }
    }

private:
    __aicore__ inline void LoadWeightAndBias() {
        LocalTensor<float> weightLocal = weightBuffer.Get<float>();
        DataCopy(weightLocal, weightGm, outFeatures * inFeatures);

        if (hasBias) {
            LocalTensor<float> biasLocal = biasBuffer.Get<float>();
            DataCopy(biasLocal, biasGm, outFeatures);
        }
    }

    __aicore__ inline void ProcessBatch(uint32_t n) {
        LocalTensor<float> xLocal = xBuffer.Get<float>();
        LocalTensor<float> weightLocal = weightBuffer.Get<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // 加载输入
        DataCopy(xLocal, xGm[n * inFeatures], inFeatures);

        // 矩阵向量乘法: y = x @ W^T
        for (uint32_t o = 0; o < outFeatures; o++) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < inFeatures; i++) {
                float xi = xLocal.GetValue(i);
                float wi = weightLocal.GetValue(o * inFeatures + i);
                sum += xi * wi;
            }

            // 加偏置
            if (hasBias) {
                LocalTensor<float> biasLocal = biasBuffer.Get<float>();
                sum += biasLocal.GetValue(o);
            }

            yLocal.SetValue(o, sum);
        }

        // 写出输出
        DataCopy(yGm[n * outFeatures], yLocal, outFeatures);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> xBuffer, weightBuffer, biasBuffer;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm, weightGm, biasGm, yGm;

    uint32_t batch, inFeatures, outFeatures, hasBias;
};

extern "C" __global__ __aicore__ void linear_kernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
    uint32_t batch, uint32_t inFeatures, uint32_t outFeatures,
    uint32_t hasBias) {
    KernelLinear op;
    op.Init(x, weight, bias, y, batch, inFeatures, outFeatures, hasBias);
    op.Process();
}
