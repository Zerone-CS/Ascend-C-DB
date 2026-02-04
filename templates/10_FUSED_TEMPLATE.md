# 融合算子模板

## 适用算子
add_relu, mul_add, fused_bias_gelu, residual_add, add_layernorm

## 融合的价值

| 指标 | 分离实现 | 融合实现 |
|------|----------|----------|
| 内存访问 | 2N (or more) | N |
| Kernel启动 | 多次 | 1次 |
| 中间结果 | 写回GM | 保留在UB |

## 融合原则

1. **可融合**: 算子间无依赖，可流水线执行
2. **限制**: UB内存足够容纳中间结果
3. **模式**: 多个元素级操作 → 一个Kernel

---

## 模板1: Add + ReLU

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelAddRelu {
public:
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
        
        // 融合: z = ReLU(x + y)
        Add(zLocal, xLocal, yLocal, tileLength);  // z = x + y
        Relu(zLocal, zLocal, tileLength);          // z = relu(z)
        
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

extern "C" __global__ __aicore__ void add_relu(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    uint32_t totalLength, uint32_t tileLength) {
    KernelAddRelu op;
    op.Init(x, y, z, totalLength, tileLength);
    op.Process();
}
```

**所需API**: `DataCopy`, `Add`, `Relu`

---

## 模板2: Bias + GELU

```cpp
__aicore__ inline void Compute(LocalTensor<float>& biasLocal) {
    LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    
    // 融合: y = GELU(x + bias)
    Add(yLocal, xLocal, biasLocal, cols);   // y = x + bias
    Gelu(yLocal, yLocal, cols);              // y = gelu(y)
    
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}
```

**所需API**: `DataCopy`, `Add`, `Gelu`

---

## 模板3: Residual + LayerNorm

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelResidualLayerNorm {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR residual, 
                                GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                uint32_t rows, uint32_t cols, float eps) {
        this->rows = rows;
        this->cols = cols;
        this->eps = eps;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, rows * cols);
        residualGm.SetGlobalBuffer((__gm__ float*)residual, rows * cols);
        gammaGm.SetGlobalBuffer((__gm__ float*)gamma, cols);
        betaGm.SetGlobalBuffer((__gm__ float*)beta, cols);
        yGm.SetGlobalBuffer((__gm__ float*)y, rows * cols);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, cols * sizeof(float));
        pipe.InitBuffer(inQueueRes, BUFFER_NUM, cols * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, cols * sizeof(float));
        pipe.InitBuffer(tmpBuffer, 1, cols * sizeof(float));
        pipe.InitBuffer(gammaBuffer, 1, cols * sizeof(float));
        pipe.InitBuffer(betaBuffer, 1, cols * sizeof(float));
    }

    __aicore__ inline void Process() {
        // 预加载gamma和beta
        LocalTensor<float> gammaLocal = gammaBuffer.Get<float>();
        LocalTensor<float> betaLocal = betaBuffer.Get<float>();
        DataCopy(gammaLocal, gammaGm, cols);
        DataCopy(betaLocal, betaGm, cols);
        
        for (uint32_t i = 0; i < rows; i++) {
            CopyIn(i);
            Compute(gammaLocal, betaLocal);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t row) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> resLocal = inQueueRes.AllocTensor<float>();
        DataCopy(xLocal, xGm[row * cols], cols);
        DataCopy(resLocal, residualGm[row * cols], cols);
        inQueueX.EnQue(xLocal);
        inQueueRes.EnQue(resLocal);
    }

    __aicore__ inline void Compute(LocalTensor<float>& gamma, LocalTensor<float>& beta) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> resLocal = inQueueRes.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
        
        // Step 1: hidden = x + residual
        Add(yLocal, xLocal, resLocal, cols);
        
        // Step 2: LayerNorm
        // mean
        ReduceSum(tmpLocal, yLocal, tmpLocal, cols);
        float mean = tmpLocal.GetValue(0) / cols;
        
        // y = hidden - mean
        Adds(yLocal, yLocal, -mean, cols);
        
        // var
        Mul(tmpLocal, yLocal, yLocal, cols);
        ReduceSum(tmpLocal, tmpLocal, tmpLocal, cols);
        float var = tmpLocal.GetValue(0) / cols;
        
        // normalize
        float invStd = 1.0f / sqrtf(var + eps);
        Muls(yLocal, yLocal, invStd, cols);
        
        // scale and shift
        Mul(yLocal, yLocal, gamma, cols);
        Add(yLocal, yLocal, beta, cols);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueRes.FreeTensor(resLocal);
    }

    __aicore__ inline void CopyOut(uint32_t row) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[row * cols], yLocal, cols);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueRes;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuffer, gammaBuffer, betaBuffer;
    GlobalTensor<float> xGm, residualGm, gammaGm, betaGm, yGm;
    uint32_t rows, cols;
    float eps;
};
```

**所需API**: `DataCopy`, `Add`, `ReduceSum`, `Adds`, `Mul`, `Muls`, `Sqrt`

---

## 模板4: Mul + Add (FMA)

```cpp
__aicore__ inline void Compute() {
    LocalTensor<float> aLocal = inQueueA.DeQue<float>();
    LocalTensor<float> bLocal = inQueueB.DeQue<float>();
    LocalTensor<float> cLocal = inQueueC.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    
    // y = a * b + c
    Mul(yLocal, aLocal, bLocal, tileLength);
    Add(yLocal, yLocal, cLocal, tileLength);
    
    outQueueY.EnQue(yLocal);
    // ... FreeTensor
}
```

---

## 常见融合模式速查

| 融合模式 | Compute内容 |
|----------|-------------|
| Add+ReLU | `Add` → `Relu` |
| Add+GELU | `Add` → `Gelu` |
| Mul+Add | `Mul` → `Add` |
| Bias+Activation | `Add(bias)` → `Activation` |
| Residual+Norm | `Add(residual)` → `LayerNorm` |
| Scale+Softmax | `Muls(scale)` → `Softmax` |

---

## 融合实现要点

1. **不要中间写回GM** - 中间结果保持在LocalTensor
2. **复用buffer** - `yLocal`可以作为中间结果继续使用
3. **注意UB大小** - 确保所有buffer不超出UB容量
