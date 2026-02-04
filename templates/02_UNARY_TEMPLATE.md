# 一元算子模板

## 适用算子
exp, log, sqrt, rsqrt, abs, neg, sin, cos, tanh, erf, reciprocal,
relu, gelu, silu, sigmoid, swish

## 所需API
`DataCopy` + 对应操作 API

## 完整模板

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class Kernel__OP_NAME__ {
public:
    __aicore__ inline Kernel__OP_NAME__() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileLength) {
        this->totalLength = totalLength;
        this->tileLength = tileLength;
        this->tileNum = totalLength / tileLength;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[idx * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        // ====== 核心计算：替换为对应API ======
        __OP_API__(yLocal, xLocal, tileLength);
        // ============================================
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[idx * tileLength], yLocal, tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm, yGm;
    uint32_t totalLength, tileLength, tileNum;
};

extern "C" __global__ __aicore__ void __OP_NAME_LOWER__(GM_ADDR x, GM_ADDR y, 
                                                         uint32_t totalLength, uint32_t tileLength) {
    Kernel__OP_NAME__ op;
    op.Init(x, y, totalLength, tileLength);
    op.Process();
}
```

## 替换表

| 算子 | `__OP_NAME__` | `__OP_NAME_LOWER__` | `__OP_API__` |
|------|---------------|---------------------|---------------|
| exp | Exp | exp | Exp |
| log | Log | log | Log |
| sqrt | Sqrt | sqrt | Sqrt |
| abs | Abs | abs | Abs |
| relu | Relu | relu | Relu |
| gelu | Gelu | gelu | Gelu |
| silu | Silu | silu | Swish |
| sigmoid | Sigmoid | sigmoid | Sigmoid |
| tanh | Tanh | tanh | Tanh |

## GELU 示例

将Compute中的核心计算替换为:
```cpp
Gelu(yLocal, xLocal, tileLength);
```

类名替换为 `KernelGelu`，函数名替换为 `gelu_custom`
