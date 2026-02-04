# Softmax 算子模板

## 适用算子
softmax, log_softmax, softmin

## 算法公式
```
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

## 所需API
`DataCopy`, `ReduceMax`, `Adds`, `Exp`, `ReduceSum`, `Muls`

## 完整模板

```cpp
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
        pipe.InitBuffer(tmpBuffer, 1, rowSize * sizeof(float));
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
        
        // Step 1: max = ReduceMax(x)
        float maxVal = 0.0f;
        ReduceMax(tmpLocal, xLocal, tmpLocal, rowSize);
        maxVal = tmpLocal.GetValue(0);
        
        // Step 2: y = x - max
        Adds(yLocal, xLocal, -maxVal, rowSize);
        
        // Step 3: y = exp(y)
        Exp(yLocal, yLocal, rowSize);
        
        // Step 4: sum = ReduceSum(y)
        float sumVal = 0.0f;
        ReduceSum(tmpLocal, yLocal, tmpLocal, rowSize);
        sumVal = tmpLocal.GetValue(0);
        
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
    TBuf<QuePosition::VECCALC> tmpBuffer;
    GlobalTensor<float> xGm, yGm;
    uint32_t totalRows, rowSize;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y, 
                                                      uint32_t totalRows, uint32_t rowSize) {
    KernelSoftmax op;
    op.Init(x, y, totalRows, rowSize);
    op.Process();
}
```

## 关键点

1. **必须先减max** - 数值稳定性关键
2. **TBuf用于临时空间** - Reduce操作需要workspace
3. **按行处理** - 外层循环rows，内层处理单行

## 扩展: log_softmax

在Step 5后添加:
```cpp
// Step 6: y = log(y)
Log(yLocal, yLocal, rowSize);
```
