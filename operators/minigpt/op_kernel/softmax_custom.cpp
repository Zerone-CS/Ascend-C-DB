/**
 * Softmax - AscendC 实现 (带数值稳定性)
 * softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
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
        uint32_t alignedRowSize = ((rowSize + 7) / 8) * 8;
        this->alignedRowSize = alignedRowSize;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalRows * rowSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalRows * rowSize);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, alignedRowSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, alignedRowSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer, alignedRowSize * sizeof(float));
        pipe.InitBuffer(workBuffer, 8 * sizeof(float));
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
        DataCopy(xLocal, xGm[row * rowSize], alignedRowSize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmp = tmpBuffer.Get<float>();
        LocalTensor<float> work = workBuffer.Get<float>();
        
        ReduceMax(work, xLocal, tmp, rowSize);
        float maxVal = work.GetValue(0);
        
        Adds(tmp, xLocal, -maxVal, alignedRowSize);
        Exp(yLocal, tmp, alignedRowSize);
        
        ReduceSum(work, yLocal, tmp, rowSize);
        float sumVal = work.GetValue(0);
        float invSum = 1.0f / sumVal;
        Muls(yLocal, yLocal, invSum, alignedRowSize);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t row) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[row * rowSize], yLocal, alignedRowSize);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuffer;
    TBuf<QuePosition::VECCALC> workBuffer;
    GlobalTensor<float> xGm, yGm;
    uint32_t totalRows, rowSize, alignedRowSize;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y,
                                                      GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmax op;
    op.Init(x, y, tiling_data.totalRows, tiling_data.rowSize);
    op.Process();
}
