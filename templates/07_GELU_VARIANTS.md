# GELU 变体模板

## 适用算子
gelu, gelu_tanh, gelu_erf, quick_gelu, geglu, swiglu

## 算法公式

### 标准 GELU (erf)
```
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

### GELU Tanh 近似
```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
```

### Quick GELU
```
QuickGELU(x) = x * sigmoid(1.702 * x)
```

### GeGLU (Gated)
```
GeGLU(x, gate) = GELU(gate) * x
```

### SwiGLU
```
SwiGLU(x, gate) = Swish(gate) * x
```

---

## 模板1: 标准GELU (直接用API)

```cpp
__aicore__ inline void Compute() {
    LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    
    // 直接调用Ascend C的Gelu API
    Gelu(yLocal, xLocal, tileLength);
    
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}
```

---

## 模板2: GELU Tanh 近似 (手动实现)

```cpp
__aicore__ inline void Compute() {
    LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
    
    // sqrt(2/pi) = 0.7978845608
    // 0.044715
    const float sqrt2pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    // tmp = x^3
    Mul(tmpLocal, xLocal, xLocal, tileLength);   // x^2
    Mul(tmpLocal, tmpLocal, xLocal, tileLength); // x^3
    
    // tmp = 0.044715 * x^3
    Muls(tmpLocal, tmpLocal, coeff, tileLength);
    
    // tmp = x + 0.044715 * x^3
    Add(tmpLocal, xLocal, tmpLocal, tileLength);
    
    // tmp = sqrt(2/pi) * (x + 0.044715 * x^3)
    Muls(tmpLocal, tmpLocal, sqrt2pi, tileLength);
    
    // tmp = tanh(...)
    Tanh(tmpLocal, tmpLocal, tileLength);
    
    // tmp = 1 + tanh(...)
    Adds(tmpLocal, tmpLocal, 1.0f, tileLength);
    
    // y = 0.5 * x * (1 + tanh(...))
    Mul(yLocal, xLocal, tmpLocal, tileLength);
    Muls(yLocal, yLocal, 0.5f, tileLength);
    
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}
```

**所需API**: `DataCopy`, `Mul`, `Muls`, `Add`, `Adds`, `Tanh`

---

## 模板3: Quick GELU

```cpp
__aicore__ inline void Compute() {
    LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
    
    // QuickGELU(x) = x * sigmoid(1.702 * x)
    Muls(tmpLocal, xLocal, 1.702f, tileLength);  // 1.702 * x
    Sigmoid(tmpLocal, tmpLocal, tileLength);      // sigmoid(1.702 * x)
    Mul(yLocal, xLocal, tmpLocal, tileLength);    // x * sigmoid(...)
    
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}
```

**所需API**: `DataCopy`, `Muls`, `Sigmoid`, `Mul`

---

## 模板4: GeGLU / SwiGLU (门控激活)

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelGeGLU {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gate, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileLength) {
        this->totalLength = totalLength;
        this->tileLength = tileLength;
        this->tileNum = totalLength / tileLength;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        gateGm.SetGlobalBuffer((__gm__ float*)gate, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(inQueueGate, BUFFER_NUM, tileLength * sizeof(float));
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
        LocalTensor<float> gateLocal = inQueueGate.AllocTensor<float>();
        DataCopy(xLocal, xGm[idx * tileLength], tileLength);
        DataCopy(gateLocal, gateGm[idx * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueGate.EnQue(gateLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> gateLocal = inQueueGate.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        // GeGLU: y = x * GELU(gate)
        // SwiGLU: y = x * Swish(gate)
        Gelu(yLocal, gateLocal, tileLength);  // 或 Swish() for SwiGLU
        Mul(yLocal, xLocal, yLocal, tileLength);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueGate.FreeTensor(gateLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[idx * tileLength], yLocal, tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueGate;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm, gateGm, yGm;
    uint32_t totalLength, tileLength, tileNum;
};
```

**所需API**: `DataCopy`, `Gelu`/`Swish`, `Mul`

---

## 选择指南

| 变体 | 何时使用 | API |
|------|----------|-----|
| GELU (erf) | 精度优先 | `Gelu` |
| GELU (tanh) | 速度优先 | 手动实现 |
| Quick GELU | 最快 | `Sigmoid`, `Mul` |
| GeGLU | LLaMA/GLM | `Gelu`, `Mul` |
| SwiGLU | LLaMA2 | `Swish`, `Mul` |
