# 二元算子模板

## 适用算子
add, sub, mul, div, pow, maximum, minimum

## 所需API
`DataCopy` + 对应操作 API (Add/Sub/Mul/Div/Max/Min)

## 完整模板

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class Kernel__OP_NAME__ {
public:
    __aicore__ inline Kernel__OP_NAME__() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, 
                                uint32_t totalLength, uint32_t tileLength) {
        this->totalLength = totalLength;
        this->tileLength = tileLength;
        this->tileNum = totalLength / tileLength;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        zGm.SetGlobalBuffer((__gm__ float*)z, totalLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(float));
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
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        DataCopy(xLocal, xGm[idx * tileLength], tileLength);
        DataCopy(yLocal, yGm[idx * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        
        // ====== 核心计算：替换为对应API ======
        __OP_API__(zLocal, xLocal, yLocal, tileLength);
        // ============================================
        
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx) {
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        DataCopy(zGm[idx * tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<float> xGm, yGm, zGm;
    uint32_t totalLength, tileLength, tileNum;
};

extern "C" __global__ __aicore__ void __OP_NAME_LOWER__(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                         uint32_t totalLength, uint32_t tileLength) {
    Kernel__OP_NAME__ op;
    op.Init(x, y, z, totalLength, tileLength);
    op.Process();
}
```

## 替换表

| 算子 | `__OP_NAME__` | `__OP_API__` |
|------|---------------|---------------|
| add | Add | Add |
| sub | Sub | Sub |
| mul | Mul | Mul |
| div | Div | Div |
| maximum | Maximum | Max |
| minimum | Minimum | Min |

## 注意事项

1. **两个输入Queue** - inQueueX 和 inQueueY
2. **两次DataCopy** - 分别加载两个输入
3. **两次FreeTensor** - 释放两个输入tensor
