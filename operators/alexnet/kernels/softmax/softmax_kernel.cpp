/**
 * Softmax 算子 - AlexNet 分类层
 *
 * softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 * 数值稳定性处理: 先减去最大值
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t batch, uint32_t numClasses) {
        this->batch = batch;
        this->numClasses = numClasses;

        xGm.SetGlobalBuffer((__gm__ float*)x, batch * numClasses);
        yGm.SetGlobalBuffer((__gm__ float*)y, batch * numClasses);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, numClasses * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, numClasses * sizeof(float));
        pipe.InitBuffer(tmpBuffer, numClasses * sizeof(float));
        pipe.InitBuffer(workBuffer, numClasses * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t n = 0; n < batch; n++) {
            CopyIn(n);
            Compute();
            CopyOut(n);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[idx * numClasses], numClasses);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
        LocalTensor<float> workLocal = workBuffer.Get<float>();

        // Step 1: 找最大值
        ReduceMax(workLocal, xLocal, workLocal, numClasses);
        float maxVal = workLocal.GetValue(0);

        // Step 2: x - max(数值稳定性)
        Adds(tmpLocal, xLocal, -maxVal, numClasses);

        // Step 3: exp(x - max)
        Exp(yLocal, tmpLocal, numClasses);

        // Step 4: sum(exp(...))
        ReduceSum(workLocal, yLocal, workLocal, numClasses);
        float sumVal = workLocal.GetValue(0);

        // Step 5: normalize
        Muls(yLocal, yLocal, 1.0f / sumVal, numClasses);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[idx * numClasses], yLocal, numClasses);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuffer, workBuffer;
    GlobalTensor<float> xGm, yGm;
    uint32_t batch, numClasses;
};

extern "C" __global__ __aicore__ void softmax_kernel(
    GM_ADDR x, GM_ADDR y,
    uint32_t batch, uint32_t numClasses) {
    KernelSoftmax op;
    op.Init(x, y, batch, numClasses);
    op.Process();
}
