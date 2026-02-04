/**
 * 05_reduce_sum - 规约操作示例 (ReduceSum)
 * 
 * 覆盖特性:
 * - ReduceSum 规约求和
 * - ReduceMax 规约求最大值
 * - 工作缓冲区使用
 * - GetValue 获取标量值
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelReduceSum {
public:
    __aicore__ inline KernelReduceSum() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalRows, uint32_t rowSize) {
        this->totalRows = totalRows;
        this->rowSize = rowSize;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalRows * rowSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalRows);  // 每行输出一个值
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, rowSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, sizeof(float));
        pipe.InitBuffer(workBuffer, rowSize * sizeof(float));  // Reduce工作缓冲区
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
        LocalTensor<float> workLocal = workBuffer.Get<float>();
        
        // 核心计算: 对每行求和
        // ReduceSum(dst, src, workBuffer, count)
        ReduceSum(workLocal, xLocal, workLocal, rowSize);
        
        // 获取规约结果 (第0个元素)
        float sumVal = workLocal.GetValue(0);
        yLocal.SetValue(0, sumVal);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t row) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[row], yLocal, 1);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> workBuffer;
    GlobalTensor<float> xGm, yGm;
    uint32_t totalRows, rowSize;
};

extern "C" __global__ __aicore__ void reduce_sum_custom(GM_ADDR x, GM_ADDR y, 
                                                         uint32_t totalRows, uint32_t rowSize) {
    KernelReduceSum op;
    op.Init(x, y, totalRows, rowSize);
    op.Process();
}
