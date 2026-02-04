# Reduce 规约算子模板

## 适用算子
reduce_sum, reduce_mean, reduce_max, reduce_min, sum, mean

## 所需API
`DataCopy`, `ReduceSum` / `ReduceMax` / `ReduceMin`, (`Muls` for mean)

## 完整模板

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelReduce__OP__ {
public:
    __aicore__ inline KernelReduce__OP__() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t outerSize, uint32_t reduceSize) {
        this->outerSize = outerSize;
        this->reduceSize = reduceSize;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, outerSize * reduceSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, outerSize);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, reduceSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer, 1, reduceSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < outerSize; i++) {
            CopyIn(i);
            float result = Compute();
            yGm.SetValue(i, result);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[idx * reduceSize], reduceSize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline float Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
        
        // ====== 核心计算：替换为对应Reduce操作 ======
        Reduce__OP__(tmpLocal, xLocal, tmpLocal, reduceSize);
        float result = tmpLocal.GetValue(0);
        // =============================================
        
        inQueueX.FreeTensor(xLocal);
        return result;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TBuf<QuePosition::VECCALC> tmpBuffer;
    GlobalTensor<float> xGm, yGm;
    uint32_t outerSize, reduceSize;
};

extern "C" __global__ __aicore__ void reduce___OP_LOWER__(GM_ADDR x, GM_ADDR y,
                                                           uint32_t outerSize, uint32_t reduceSize) {
    KernelReduce__OP__ op;
    op.Init(x, y, outerSize, reduceSize);
    op.Process();
}
```

## 替换表

| 算子 | `__OP__` | `__OP_LOWER__` | 核心API |
|------|----------|----------------|----------|
| reduce_sum | Sum | sum | ReduceSum |
| reduce_max | Max | max | ReduceMax |
| reduce_min | Min | min | ReduceMin |
| sum | Sum | sum | ReduceSum |

## reduce_mean 变体

```cpp
// 在Compute中，ReduceSum之后添加:
result = result / reduceSize;
```

## 注意事项

1. **输出用SetValue** - 规约结果是单个值，直接写入GM
2. **TBuf用于workspace** - Reduce操作需要临时空间
3. **按外层维度循环** - 写入yGm[i]
