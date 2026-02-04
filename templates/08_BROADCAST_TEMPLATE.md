# Broadcast 广播算子模板

## 适用算子
add_scalar, mul_scalar, bias_add, scale, add_bias, broadcast_add, broadcast_mul

## 场景说明

| 场景 | 输入A | 输入B | 操作 |
|------|-------|-------|------|
| 加标量 | [M, N] | scalar | A + scalar |
| 加偏置 | [M, N] | [N] | A + B (broadcast) |
| 乘缩放 | [M, N] | [N] | A * B (broadcast) |
| 分层缩放 | [M, N] | [M, 1] | A * B (broadcast) |

---

## 模板1: Tensor + Scalar

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelAddScalar {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, float scalar,
                                uint32_t totalLength, uint32_t tileLength) {
        this->scalar = scalar;
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
        
        // y = x + scalar
        Adds(yLocal, xLocal, scalar, tileLength);
        // 或: Muls(yLocal, xLocal, scalar, tileLength);  // y = x * scalar
        
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
    float scalar;
    uint32_t totalLength, tileLength, tileNum;
};

extern "C" __global__ __aicore__ void add_scalar(GM_ADDR x, GM_ADDR y, float scalar,
                                                  uint32_t totalLength, uint32_t tileLength) {
    KernelAddScalar op;
    op.Init(x, y, scalar, totalLength, tileLength);
    op.Process();
}
```

**所需API**: `DataCopy`, `Adds` / `Muls`

---

## 模板2: BiasAdd [M,N] + [N]

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelBiasAdd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y,
                                uint32_t rows, uint32_t cols) {
        this->rows = rows;
        this->cols = cols;  // bias的长度
        
        xGm.SetGlobalBuffer((__gm__ float*)x, rows * cols);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, cols);
        yGm.SetGlobalBuffer((__gm__ float*)y, rows * cols);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, cols * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, cols * sizeof(float));
        pipe.InitBuffer(biasBuffer, 1, cols * sizeof(float));
    }

    __aicore__ inline void Process() {
        // 预加载bias（只加载一次）
        LocalTensor<float> biasLocal = biasBuffer.Get<float>();
        DataCopy(biasLocal, biasGm, cols);
        
        // 按行处理
        for (uint32_t i = 0; i < rows; i++) {
            CopyIn(i);
            Compute(biasLocal);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t row) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[row * cols], cols);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(LocalTensor<float>& biasLocal) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        // y = x + bias (bias广播到每一行)
        Add(yLocal, xLocal, biasLocal, cols);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t row) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[row * cols], yLocal, cols);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> biasBuffer;
    GlobalTensor<float> xGm, biasGm, yGm;
    uint32_t rows, cols;
};

extern "C" __global__ __aicore__ void bias_add(GM_ADDR x, GM_ADDR bias, GM_ADDR y,
                                                uint32_t rows, uint32_t cols) {
    KernelBiasAdd op;
    op.Init(x, bias, y, rows, cols);
    op.Process();
}
```

**所需API**: `DataCopy`, `Add`

---

## 模板3: Scale [M,N] * [N]

```cpp
// 与BiasAdd结构相同，只改Compute里的操作:

__aicore__ inline void Compute(LocalTensor<float>& scaleLocal) {
    LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    
    // y = x * scale (逐元素乘)
    Mul(yLocal, xLocal, scaleLocal, cols);
    
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}
```

**所需API**: `DataCopy`, `Mul`

---

## 模板4: 通用 Broadcast Binary

处理两个不同shape的tensor运算:

```cpp
// 当前限制：较小的tensor必须是较大tensor的最后几维
// 例如: [M,N,K] op [K] 或 [M,N,K] op [N,K]

// 核心思路:
// 1. 小 tensor只加载一次到UB
// 2. 大 tensor分块载入
// 3. 每块与小 tensor计算
```

---

## 注意事项

1. **小 tensor预加载** - bias/scale等只加载一次，避免重复访存
2. **TBuf vs TQue** - 反复使用的数据用TBuf，流式数据用TQue
3. **广播维度** - 确保广播维度是最后维度
