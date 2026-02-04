# Attention 算子模板

## 适用算子
scaled_dot_product_attention, self_attention, cross_attention, multi_head_attention

## 算法公式
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

**分解步骤**:
1. `scores = Q @ K^T` (MatMul)
2. `scores = scores / sqrt(d_k)` (Scale)
3. `scores = scores + mask` (Optional: Mask)
4. `attn = softmax(scores)` (Softmax)
5. `output = attn @ V` (MatMul)

## 复杂度说明

| 实现方式 | 复杂度 | 说明 |
|----------|--------|------|
| 拆分实现 | 中 | 分别实现MatMul+Softmax |
| 融合实现 | 高 | 需要复杂的tiling |
| Flash Attention | 非常高 | 建议用高阶API |

---

## 方案一：拆分实现（推荐新手）

将Attention拆分为多个简单算子，分别实现：

```python
# Host端组装
# Step 1: scores = Q @ K^T
matmul_op(Q, K_transpose, scores)  # 用 06_MATMUL_TEMPLATE

# Step 2: scores = scores / sqrt(d_k)
scale_op(scores, 1.0/sqrt(d_k), scores)  # 用 08_BROADCAST (Muls)

# Step 3: attn = softmax(scores)
softmax_op(scores, attn)  # 用 01_SOFTMAX_TEMPLATE

# Step 4: output = attn @ V  
matmul_op(attn, V, output)  # 用 06_MATMUL_TEMPLATE
```

---

## 方案二：简化融合实现

小序列长度（seq_len < 2048）可以融合 Scale + Softmax:

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelScaledSoftmax {
public:
    __aicore__ inline void Init(GM_ADDR scores, GM_ADDR attn,
                                uint32_t batchSize, uint32_t seqLen, float scale) {
        this->batchSize = batchSize;
        this->seqLen = seqLen;
        this->scale = scale;
        this->totalRows = batchSize * seqLen;  // 每个query对应一行
        
        scoresGm.SetGlobalBuffer((__gm__ float*)scores, totalRows * seqLen);
        attnGm.SetGlobalBuffer((__gm__ float*)attn, totalRows * seqLen);
        
        pipe.InitBuffer(inQueue, BUFFER_NUM, seqLen * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, seqLen * sizeof(float));
        pipe.InitBuffer(tmpBuffer, 1, seqLen * sizeof(float));
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
        LocalTensor<float> local = inQueue.AllocTensor<float>();
        DataCopy(local, scoresGm[row * seqLen], seqLen);
        inQueue.EnQue(local);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> scoresLocal = inQueue.DeQue<float>();
        LocalTensor<float> attnLocal = outQueue.AllocTensor<float>();
        LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
        
        // Step 1: scores = scores * scale
        Muls(attnLocal, scoresLocal, scale, seqLen);
        
        // Step 2-5: softmax
        // max
        ReduceMax(tmpLocal, attnLocal, tmpLocal, seqLen);
        float maxVal = tmpLocal.GetValue(0);
        
        // exp(x - max)
        Adds(attnLocal, attnLocal, -maxVal, seqLen);
        Exp(attnLocal, attnLocal, seqLen);
        
        // sum
        ReduceSum(tmpLocal, attnLocal, tmpLocal, seqLen);
        float sumVal = tmpLocal.GetValue(0);
        
        // normalize
        Muls(attnLocal, attnLocal, 1.0f / sumVal, seqLen);
        
        outQueue.EnQue(attnLocal);
        inQueue.FreeTensor(scoresLocal);
    }

    __aicore__ inline void CopyOut(uint32_t row) {
        LocalTensor<float> local = outQueue.DeQue<float>();
        DataCopy(attnGm[row * seqLen], local, seqLen);
        outQueue.FreeTensor(local);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> tmpBuffer;
    GlobalTensor<float> scoresGm, attnGm;
    uint32_t batchSize, seqLen, totalRows;
    float scale;
};

extern "C" __global__ __aicore__ void scaled_softmax(
    GM_ADDR scores, GM_ADDR attn,
    uint32_t batchSize, uint32_t seqLen, float scale) {
    KernelScaledSoftmax op;
    op.Init(scores, attn, batchSize, seqLen, scale);
    op.Process();
}
```

**所需API**: `DataCopy`, `Muls`, `ReduceMax`, `Adds`, `Exp`, `ReduceSum`

---

## 方案三：Flash Attention（高阶）

**强烈建议使用华为提供的Flash Attention实现**

查询数据库:
```bash
python query_db.py search "FlashAttention"
python query_db.py search "融合注意力"
```

原因:
1. Flash Attention需要复杂的分块策略
2. 需要处理online softmax
3. 需要管理多级内存层次

---

## Attention Mask 处理

### Causal Mask (下三角)

```cpp
// 在softmax之前应用mask
// mask[i][j] = 0 if j <= i else -inf

__aicore__ inline void ApplyCausalMask(LocalTensor<float>& scores, uint32_t row) {
    // 将未来位置设为-inf
    for (uint32_t j = row + 1; j < seqLen; j++) {
        scores.SetValue(j, -INFINITY);
    }
}
```

### Padding Mask

```cpp
// 从Mask tensor读取，0表示有效，1表示padding
__aicore__ inline void ApplyPaddingMask(LocalTensor<float>& scores, 
                                         LocalTensor<int>& mask) {
    // scores = scores + mask * (-10000)
    // 或使用Select API
}
```

---

## Multi-Head Attention

```cpp
// MHA = Concat(head_1, head_2, ..., head_h) @ W_o
// 其中 head_i = Attention(Q @ W_q_i, K @ W_k_i, V @ W_v_i)

// 实现要点:
// 1. 先对Q/K/V做Linear投影
// 2. Reshape为 [batch, heads, seq, head_dim]
// 3. 每个head独立计算Attention
// 4. Concat + 输出Linear
```

---

## 推荐实现路径

| 场景 | 推荐方案 |
|------|----------|
| 学习/原型 | 拆分实现 |
| seq_len < 512 | 简化融合 |
| seq_len > 512 | Flash Attention API |
| 生产部署 | Flash Attention API |
