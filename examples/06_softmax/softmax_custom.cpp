/**
 * 06_softmax - 复合算子示例 (Softmax)
 * 
 * 覆盖特性:
 * - 组合多种 API: ReduceMax, Adds, Exp, ReduceSum, Muls
 * - 数值稳定性处理 (x - max(x))
 * - 多步计算流程
 * 
 * 算法: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalRows, uint32_t rowSize) {
        this->totalRows = totalRows;
        this->rowSize = rowSize;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalRows * rowSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalRows * rowSize);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, rowSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, rowSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer, rowSize * sizeof(float));
        pipe.InitBuffer(workBuffer, rowSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < totalRows; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t row) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[row * rowSize], rowSize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
        LocalTensor<float> workLocal = workBuffer.Get<float>();
        
        // Step 1: 找最大值 max = ReduceMax(x)
        ReduceMax(workLocal, xLocal, workLocal, rowSize);
        float maxVal = workLocal.GetValue(0);
        
        // Step 2: y = x - max (数值稳定性)
        Adds(tmpLocal, xLocal, -maxVal, rowSize);
        
        // Step 3: y = exp(y)
        Exp(yLocal, tmpLocal, rowSize);
        
        // Step 4: sum = ReduceSum(y)
        ReduceSum(workLocal, yLocal, workLocal, rowSize);
        float sumVal = workLocal.GetValue(0);
        
        // Step 5: y = y / sum
        Muls(yLocal, yLocal, 1.0f / sumVal, rowSize);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t row) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[row * rowSize], yLocal, rowSize);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuffer, workBuffer;
    GlobalTensor<float> xGm, yGm;
    uint32_t totalRows, rowSize;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y, 
                                                      uint32_t totalRows, uint32_t rowSize) {
    KernelSoftmax op;
    op.Init(x, y, totalRows, rowSize);
    op.Process();
}
