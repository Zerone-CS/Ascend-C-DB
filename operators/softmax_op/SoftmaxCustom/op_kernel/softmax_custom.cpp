#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalRows, uint32_t rowSize)
    {
        this->totalRows = totalRows;
        this->rowSize = rowSize;
        this->blockIdx = GetBlockIdx();
        
        uint32_t blockNum = GetBlockNum();
        uint32_t rowsPerCore = (totalRows + blockNum - 1) / blockNum;
        uint32_t startRow = blockIdx * rowsPerCore;
        uint32_t endRow = (startRow + rowsPerCore > totalRows) ? totalRows : (startRow + rowsPerCore);
        this->processRows = (startRow >= totalRows) ? 0 : (endRow - startRow);
        
        uint32_t offset = startRow * rowSize;
        xGm.SetGlobalBuffer((__gm__ float*)x + offset, processRows * rowSize);
        yGm.SetGlobalBuffer((__gm__ float*)y + offset, processRows * rowSize);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, rowSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, rowSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer1, rowSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer2, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < processRows; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t row)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[row * rowSize], rowSize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t row)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmpLocal = tmpBuffer1.Get<float>();
        LocalTensor<float> maxSumLocal = tmpBuffer2.Get<float>();
        
        // Step 1: Find max value in the row
        ReduceMax(maxSumLocal, xLocal, tmpLocal, rowSize);
        float maxVal = maxSumLocal.GetValue(0);
        
        // Step 2: Subtract max and compute exp
        Adds(tmpLocal, xLocal, -maxVal, rowSize);
        Exp(yLocal, tmpLocal, rowSize);
        
        // Step 3: Sum of exp values
        ReduceSum(maxSumLocal, yLocal, tmpLocal, rowSize);
        float sumVal = maxSumLocal.GetValue(0);
        
        // Step 4: Divide by sum
        float invSum = 1.0f / sumVal;
        Muls(yLocal, yLocal, invSum, rowSize);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t row)
    {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[row * rowSize], yLocal, rowSize);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuffer1;
    TBuf<QuePosition::VECCALC> tmpBuffer2;
    GlobalTensor<float> xGm, yGm;
    uint32_t totalRows, rowSize, processRows, blockIdx;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmax op;
    op.Init(x, y, tiling_data.totalRows, tiling_data.rowSize);
    op.Process();
}
