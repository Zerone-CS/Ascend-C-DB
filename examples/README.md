# Ascend C 算子示例集

本目录包含 13 个示例算子，覆盖 Ascend C 的大部分语法和核心机制。

## 示例列表

| 序号 | 示例 | 核心特性 | 复杂度 |
|------|------|---------|--------|
| 01 | [unary_exp](01_unary_exp/) | 一元运算, TQue, Double Buffer | ★☆☆ |
| 02 | [binary_add](02_binary_add/) | 二元运算, 多输入队列, 多核并行 | ★☆☆ |
| 03 | [activation_relu](03_activation_relu/) | 激活函数, 数据对齐 | ★☆☆ |
| 04 | [scalar_ops](04_scalar_ops/) | 标量运算 (Adds/Muls), TBuf | ★☆☆ |
| 05 | [reduce_sum](05_reduce_sum/) | 规约运算, GetValue, 工作缓冲区 | ★★☆ |
| 06 | [softmax](06_softmax/) | 复合运算, 数值稳定性 | ★★☆ |
| 07 | [gated_swiglu](07_gated_swiglu/) | 双输入, 多中TBuf, Div | ★★☆ |
| 08 | [multicore](08_multicore/) | GetBlockIdx/Num, 负载均衡 | ★★☆ |
| 09 | [cast_dtype](09_cast_dtype/) | 类型转换 (Cast), half, RoundMode | ★★☆ |
| 10 | [broadcast](10_broadcast/) | 广播加法, Duplicate | ★★☆ |
| 11 | [layernorm](11_layernorm/) | 多步规约, 仿射参数 | ★★★ |
| 12 | [rms_norm](12_rms_norm/) | RMSNorm (LLaMA), Rsqrt | ★★★ |
| 13 | [tiling_params](13_tiling_params/) | Tiling结构体, 动态分块 | ★★★ |

---

## 特性覆盖矩阵

### 数据类型
| 特性 | 示例 |
|------|------|
| float32 | 所有示例 |
| float16 (half) | 09_cast_dtype |
| 类型转换 Cast | 09_cast_dtype |

### 内存管理
| 特性 | 示例 |
|------|------|
| GlobalTensor | 所有示例 |
| LocalTensor | 所有示例 |
| TQue (VECIN/VECOUT) | 所有示例 |
| TBuf (VECCALC) | 04, 05, 06, 07, 10, 11, 12 |
| TPipe | 所有示例 |
| Double Buffer | 所有示例 |

### 数据搬运
| 特性 | 示例 |
|------|------|
| DataCopy | 所有示例 |
| GM -> Local | 所有示例 |
| Local -> GM | 所有示例 |

### 计算 API
| 类别 | API | 示例 |
|------|-----|------|
| 一元 | Exp | 01 |
| 一元 | Relu | 03 |
| 二元 | Add | 02, 10, 11 |
| 二元 | Mul | 06, 07, 11, 12 |
| 二元 | Div | 07 |
| 标量 | Adds | 04, 05, 06, 07 |
| 标量 | Muls | 04, 06, 07, 08, 11, 12, 13 |
| 规约 | ReduceSum | 05, 06, 11, 12 |
| 规约 | ReduceMax | 06 |
| 转换 | Cast | 09 |
| 填充 | Duplicate | 10 |

### 多核并行
| 特性 | 示例 |
|------|------|
| GetBlockIdx() | 02, 03, 08, 13 |
| GetBlockNum() | 02, 03, 08, 13 |
| 负载均衡 | 02, 08, 13 |

### Tiling 机制
| 特性 | 示例 |
|------|------|
| 静态tileSize | 01-12 |
| Tiling结构体 | 13 |
| 动态分块 | 13 |

---

## 代码结构模板

所有示例遵循统一结构:

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;  // Double Buffer

class KernelXxx {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, ...) {
        // 1. SetGlobalBuffer
        // 2. pipe.InitBuffer
    }
    
    __aicore__ inline void Process() {
        for (...) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }
    
private:
    __aicore__ inline void CopyIn(...)  { /* GM -> Local */ }
    __aicore__ inline void Compute()    { /* 核心计算 */ }
    __aicore__ inline void CopyOut(...) { /* Local -> GM */ }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<T> xGm, yGm;
};

extern "C" __global__ __aicore__ void xxx_kernel(...) {
    KernelXxx op;
    op.Init(...);
    op.Process();
}
```

---

## Queue 操作配对规则

```
AllocTensor  <->  FreeTensor
EnQue        <->  DeQue
```

正确顺序:
1. `Alloc` -> `EnQue` (写入方)
2. `DeQue` -> `Free` (读取方)

---

## 学习路径建议

**初学者**: 01 -> 02 -> 03 -> 04 -> 05

**进阶者**: 06 -> 07 -> 08 -> 09 -> 10

**高级用户**: 11 -> 12 -> 13

---

## 编译示例

```bash
# 设置环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 编译单个文件 (仅生成.o)
ccec --cce-aicore-arch=dav-c220 -O2 -c xxx_custom.cpp -o xxx_custom.o
```

## 注意事项

1. **对齐要求**: DataCopy 的 count 必须 32 字节对齐
2. **BUFFER_NUM**: 必须为 2 (实现 Double Buffer)
3. **Queue配对**: Alloc/Free, EnQue/DeQue 必须严格配对
4. **数值稳定性**: Softmax 等算子需先减去最大值
