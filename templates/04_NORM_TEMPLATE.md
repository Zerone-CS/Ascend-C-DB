# 归一化算子模板

## 适用算子
layernorm, rmsnorm, batchnorm, groupnorm

---

## LayerNorm

### 算法公式
```
y = (x - mean) / sqrt(var + eps) * gamma + beta
mean = sum(x) / n
var = sum((x - mean)^2) / n
```

### 所需API
`DataCopy`, `ReduceSum`, `Muls`, `Adds`, `Mul`, `Add`, `Sqrt`

### 完整模板

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelLayerNorm {
public:
    __aicore__ inline KernelLayerNorm() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                uint32_t totalRows, uint32_t normSize, float eps) {
        this->totalRows = totalRows;
        this->normSize = normSize;
        this->eps = eps;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalRows * normSize);
        gammaGm.SetGlobalBuffer((__gm__ float*)gamma, normSize);
        betaGm.SetGlobalBuffer((__gm__ float*)beta, normSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalRows * normSize);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, normSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, normSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer, 1, normSize * sizeof(float));
        pipe.InitBuffer(gammaBuffer, 1, normSize * sizeof(float));
        pipe.InitBuffer(betaBuffer, 1, normSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        LocalTensor<float> gammaLocal = gammaBuffer.Get<float>();
        LocalTensor<float> betaLocal = betaBuffer.Get<float>();
        DataCopy(gammaLocal, gammaGm, normSize);
        DataCopy(betaLocal, betaGm, normSize);
        
        for (uint32_t i = 0; i < totalRows; i++) {
            CopyIn(i);
            Compute(gammaLocal, betaLocal);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t row) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[row * normSize], normSize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(LocalTensor<float>& gamma, LocalTensor<float>& beta) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
        
        // Step 1: mean = sum(x) / n
        ReduceSum(tmpLocal, xLocal, tmpLocal, normSize);
        float mean = tmpLocal.GetValue(0) / normSize;
        
        // Step 2: y = x - mean
        Adds(yLocal, xLocal, -mean, normSize);
        
        // Step 3: var = sum((x-mean)^2) / n
        Mul(tmpLocal, yLocal, yLocal, normSize);
        ReduceSum(tmpLocal, tmpLocal, tmpLocal, normSize);
        float var = tmpLocal.GetValue(0) / normSize;
        
        // Step 4: y = (x - mean) / sqrt(var + eps)
        float invStd = 1.0f / sqrtf(var + eps);
        Muls(yLocal, yLocal, invStd, normSize);
        
        // Step 5: y = y * gamma + beta
        Mul(yLocal, yLocal, gamma, normSize);
        Add(yLocal, yLocal, beta, normSize);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t row) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[row * normSize], yLocal, normSize);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuffer;
    TBuf<QuePosition::VECCALC> gammaBuffer;
    TBuf<QuePosition::VECCALC> betaBuffer;
    GlobalTensor<float> xGm, gammaGm, betaGm, yGm;
    uint32_t totalRows, normSize;
    float eps;
};

extern "C" __global__ __aicore__ void layernorm_custom(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
    uint32_t totalRows, uint32_t normSize, float eps) {
    KernelLayerNorm op;
    op.Init(x, gamma, beta, y, totalRows, normSize, eps);
    op.Process();
}
```

---

## RMSNorm 变体

### 算法公式
```
y = x / sqrt(mean(x^2) + eps) * gamma
```

### 与 LayerNorm 的区别

| | LayerNorm | RMSNorm |
|---|-----------|----------|
| 减均值 | 是 | 否 |
| beta参数 | 是 | 否 |
| 计算量 | 较大 | 较小 |

### RMSNorm核心Compute（替换LayerNorm模板中的Compute）

```cpp
__aicore__ inline void Compute(LocalTensor<float>& gamma) {
    LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
    
    // Step 1: rms = sqrt(mean(x^2) + eps)
    Mul(tmpLocal, xLocal, xLocal, normSize);  // x^2
    ReduceSum(tmpLocal, tmpLocal, tmpLocal, normSize);
    float rms = sqrtf(tmpLocal.GetValue(0) / normSize + eps);
    
    // Step 2: y = x / rms * gamma
    Muls(yLocal, xLocal, 1.0f / rms, normSize);
    Mul(yLocal, yLocal, gamma, normSize);
    
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}
```
