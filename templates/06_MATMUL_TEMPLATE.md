# MatMul 矩阵乘法算子模板

## 适用算子
matmul, gemm, linear, bmm

## 算法公式
```
C = A @ B
C[m,n] = sum(A[m,k] * B[k,n])
```

## 所需API
`DataCopy`, `Mmad` (Matrix Multiply Add)

## 重要说明

**MatMul使用Cube单元，与Vector算子结构不同：**

| 项目 | Vector算子 | Cube算子(MatMul) |
|------|-----------|------------------|
| 计算单元 | Vector Core | Cube Core |
| 数据布局 | 连续内存 | 分形(Fractal)格式 |
| 内存层级 | GM→UB | GM→L1→L0A/L0B→L0C→UB→GM |
| Tiling | 简单分块 | M/N/K三维分块 |

---

## 基础模板（单核简化版）

```cpp
#include "kernel_operator.h"
using namespace AscendC;

class KernelMatMul {
public:
    __aicore__ inline KernelMatMul() {}
    
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                uint32_t M, uint32_t N, uint32_t K) {
        this->M = M;
        this->N = N;
        this->K = K;
        
        aGm.SetGlobalBuffer((__gm__ half*)a, M * K);
        bGm.SetGlobalBuffer((__gm__ half*)b, K * N);
        cGm.SetGlobalBuffer((__gm__ float*)c, M * N);
        
        // Cube计算需要的buffer
        pipe.InitBuffer(aL1, M * K * sizeof(half));
        pipe.InitBuffer(bL1, K * N * sizeof(half));
        pipe.InitBuffer(cUB, M * N * sizeof(float));
    }

    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<half> aLocal = aL1.Get<half>();
        LocalTensor<half> bLocal = bL1.Get<half>();
        
        DataCopy(aLocal, aGm, M * K);
        DataCopy(bLocal, bGm, K * N);
    }

    __aicore__ inline void Compute() {
        LocalTensor<half> aLocal = aL1.Get<half>();
        LocalTensor<half> bLocal = bL1.Get<half>();
        LocalTensor<float> cLocal = cUB.Get<float>();
        
        // Cube矩阵乘法
        // 注意：实际使用需要处理数据格式转换(ND→NZ)
        Mmad(cLocal, aLocal, bLocal, M, N, K);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> cLocal = cUB.Get<float>();
        DataCopy(cGm, cLocal, M * N);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::A1> aL1;
    TBuf<QuePosition::B1> bL1;
    TBuf<QuePosition::CO1> cUB;
    GlobalTensor<half> aGm, bGm;
    GlobalTensor<float> cGm;
    uint32_t M, N, K;
};

extern "C" __global__ __aicore__ void matmul_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c,
    uint32_t M, uint32_t N, uint32_t K) {
    KernelMatMul op;
    op.Init(a, b, c, M, N, K);
    op.Process();
}
```

---

## 关键差异说明

### 1. Buffer位置不同

```cpp
// Vector算子
TQue<QuePosition::VECIN, 2> inQueue;
TQue<QuePosition::VECOUT, 2> outQueue;

// Cube算子
TBuf<QuePosition::A1> aL1;    // A矩阵放L1
TBuf<QuePosition::B1> bL1;    // B矩阵放L1  
TBuf<QuePosition::CO1> cUB;   // C结果放UB
```

### 2. 数据类型要求

| 输入A | 输入B | 输出C |
|-------|-------|-------|
| half | half | float |
| int8 | int8 | int32 |

### 3. Shape对齐要求

- M: 需要16对齐
- N: 需要16对齐  
- K: 需要16对齐

---

## 推荐：使用高阶API

对于MatMul，**强烈建议使用高阶API `matmul_intf`** 而非手写：

```cpp
#include "kernel_operator.h"
using namespace AscendC;
using namespace matmul;

// 使用Matmul高阶API
MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType;
MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType;
MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType;

Matmul<decltype(aType), decltype(bType), decltype(cType)> mm;
mm.Init(...);
mm.Compute(...);
```

**原因**：
1. 高阶API自动处理分形格式转换
2. 自动处理多核并行
3. 自动优化Tiling

---

## 注意事项

1. **Cube与Vector不混用** - MatMul计算完成后，如需后处理(如BiasAdd)，需要先将数据从L0C搬到UB
2. **数据格式** - Cube单元要求NZ(分形)格式，需要格式转换
3. **复杂度高** - MatMul是最复杂的算子类型，建议先掌握Vector算子

## 参考

查询数据库获取更多MatMul细节：
```bash
python query_db.py search "Mmad"
python query_db.py search "矩阵乘法"
python query_db.py show 6.3  # 矩阵编程章节
```
