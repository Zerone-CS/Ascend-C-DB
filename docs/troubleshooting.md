# 常见问题排查

## 编译错误

### 1. undefined reference to `xxx`

**原因**: 缺少头文件或命名空间

**解决**:
```cpp
#include "kernel_operator.h"
using namespace AscendC;
```

### 2. ccec: command not found

**原因**: CANN 环境未加载

**解决**:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 3. error: expected unqualified-id

**原因**: 关键字拼写错误

**检查**:
- `__aicore__` 不是 `__aicore`
- `__global__` 不是 `__global`
- `BUFFER_NUM` 必须为常量

---

## 运行时错误

### 1. aclrtSynchronizeStream failed (507015)

**原因**: 内核执行失败

**排查步骤**:
1. 检查 NPU 状态: `npu-smi info`
2. 检查设备是否被占用
3. 确认 SOC 版本匹配

### 2. 内存越界 / Segfault

**常见原因**:
- DataCopy count 未对齐
- GlobalBuffer size 不足
- 索引超出范围

**解决**:
```cpp
// 确保 count 32字节对齐 (float=4B, 所以8个元素)
uint32_t alignedCount = ((count + 7) / 8) * 8;
DataCopy(dst, src, alignedCount);
```

### 3. Queue 死锁

**原因**: Alloc/Free 或 EnQue/DeQue 不配对

**正确模式**:
```cpp
// 写入方
LocalTensor<T> x = queue.AllocTensor<T>();
// ... 填充数据 ...
queue.EnQue(x);

// 读取方
LocalTensor<T> x = queue.DeQue<T>();
// ... 使用数据 ...
queue.FreeTensor(x);
```

---

## 精度问题

### 1. Softmax 输出全为 NaN

**原因**: 数值溢出

**解决**: 先减去最大值
```cpp
// 错误: 直接 exp(x)
Exp(y, x, count);  // 大值x会溢出!

// 正确: exp(x - max(x))
ReduceMax(work, x, work, count);
float maxVal = work.GetValue(0);
Adds(y, x, -maxVal, count);
Exp(y, y, count);
```

### 2. LayerNorm 误差大

**原因**: 方差计算方式

**解决**: 使用数值稳定的计算方式
```cpp
// 计算 mean
ReduceSum(work, x, work, count);
float mean = work.GetValue(0) / count;

// 计算 (x - mean)^2的均值
Adds(tmp, x, -mean, count);
Mul(tmp, tmp, tmp, count);
ReduceSum(work, tmp, work, count);
float var = work.GetValue(0) / count;
```

---

## 性能问题

### 1. 速度比预期慢

**检查点**:
1. tileSize 是否套合理 (通常 256 或 512)
2. 是否启用多核
3. 数据搬运是否对齐

### 2. 内存不足

**优化方案**:
1. 减小 tileSize
2. 减少 TBuf 数量
3. 复用缓冲区

---

## 调试技巧

### 1. 打印中间结果

```cpp
// 在 Compute() 中保存中间结果到 GM
DataCopy(debugGm, tmpLocal, count);
```

### 2. 单步验证

```cpp
// 先只做 DataCopy 确认数据通路
DataCopy(yLocal, xLocal, count);  // 简单复制

// 然后逐步添加计算逻辑
```

### 3. 对比参考实现

```python
# Python 参考实现
import numpy as np
golden = np.exp(input_data)  # 生成golden数据
golden.tofile('golden.bin')
```
