# Ascend C API 参考

## 数据类型

| 类型 | 说明 | 大小 |
|------|------|------|
| `float` | 32位浮点 | 4 bytes |
| `half` | 16位浮点 | 2 bytes |
| `int32_t` | 32位整数 | 4 bytes |
| `int8_t` | 8位整数 | 1 byte |

## 内存类型

### GlobalTensor
全局内存张量，指向 GM (Global Memory)。

```cpp
GlobalTensor<float> xGm;
xGm.SetGlobalBuffer((__gm__ float*)addr, size);
```

### LocalTensor
本地内存张量，位于 UB (Unified Buffer)。

```cpp
LocalTensor<float> xLocal = queue.AllocTensor<float>();
float val = xLocal.GetValue(0);  // 读取元素
xLocal.SetValue(0, 1.0f);        // 设置元素
```

## 队列管理

### TQue
用于输入/输出的队列缓冲区。

```cpp
// 定义
TQue<QuePosition::VECIN, 2> inQueue;   // 输入队列
TQue<QuePosition::VECOUT, 2> outQueue; // 输出队列

// 初始化
pipe.InitBuffer(inQueue, BUFFER_NUM, size);

// 操作
LocalTensor<T> x = inQueue.AllocTensor<T>();  // 分配
inQueue.EnQue(x);                              // 入队
LocalTensor<T> x = inQueue.DeQue<T>();        // 出队
inQueue.FreeTensor(x);                         // 释放
```

### TBuf
用于中间计算的缓冲区 (不需要队列语义)。

```cpp
TBuf<QuePosition::VECCALC> tmpBuffer;
pipe.InitBuffer(tmpBuffer, size);
LocalTensor<T> tmp = tmpBuffer.Get<T>();
```

## 数据搬运

### DataCopy
在 GM 和 UB 之间拷贝数据。

```cpp
// GM -> UB
DataCopy(localTensor, globalTensor[offset], count);

// UB -> GM
DataCopy(globalTensor[offset], localTensor, count);
```

> 注意: count 必须 32 字节对齐

## 计算 API

### 一元运算

```cpp
Exp(dst, src, count);        // dst = exp(src)
Log(dst, src, count);        // dst = log(src)
Sqrt(dst, src, count);       // dst = sqrt(src)
Rsqrt(dst, src, count);      // dst = 1/sqrt(src)
Abs(dst, src, count);        // dst = |src|
Neg(dst, src, count);        // dst = -src
Relu(dst, src, count);       // dst = max(0, src)
Sigmoid(dst, src, count);    // dst = 1/(1+exp(-src))
Tanh(dst, src, count);       // dst = tanh(src)
Gelu(dst, src, count);       // dst = gelu(src)
Swish(dst, src, count);      // dst = src * sigmoid(src)
```

### 二元运算

```cpp
Add(dst, src0, src1, count); // dst = src0 + src1
Sub(dst, src0, src1, count); // dst = src0 - src1
Mul(dst, src0, src1, count); // dst = src0 * src1
Div(dst, src0, src1, count); // dst = src0 / src1
Max(dst, src0, src1, count); // dst = max(src0, src1)
Min(dst, src0, src1, count); // dst = min(src0, src1)
```

### 标量运算

```cpp
Adds(dst, src, scalar, count);  // dst = src + scalar
Muls(dst, src, scalar, count);  // dst = src * scalar
```

### 规约运算

```cpp
ReduceSum(dst, src, work, count);  // dst[0] = sum(src)
ReduceMax(dst, src, work, count);  // dst[0] = max(src)
ReduceMin(dst, src, work, count);  // dst[0] = min(src)
```

> work 为工作缓冲区，大小 >= src

### 类型转换

```cpp
// Cast(dst, src, roundMode, count)
Cast(dstF16, srcF32, RoundMode::CAST_ROUND, count);  // f32 -> f16
Cast(dstF32, srcF16, RoundMode::CAST_NONE, count);   // f16 -> f32
```

RoundMode:
- `CAST_NONE` - 无舍入
- `CAST_ROUND` - 四舍五入
- `CAST_FLOOR` - 向下取整
- `CAST_CEIL` - 向上取整

### 填充操作

```cpp
Duplicate(dst, scalar, count);  // 用scalar填充dst
```

## 多核并行

```cpp
uint32_t blockIdx = GetBlockIdx();   // 当前核索引 [0, blockNum)
uint32_t blockNum = GetBlockNum();   // 总核数
```

## 关键字

| 关键字 | 说明 |
|--------|------|
| `__aicore__` | AI Core上执行的函数 |
| `__global__` | 全局入口函数 |
| `__gm__` | 全局内存指针 |
| `GM_ADDR` | 全局内存地址类型 |
