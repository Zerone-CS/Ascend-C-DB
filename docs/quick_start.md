# 快速开始

## 环境准备

### 1. 检查 CANN 安装

```bash
# 加载环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查编译器
ccec --version

# 检查 NPU
npu-smi info
```

### 2. 加载项目环境

```bash
cd Ascend-C-DB
source scripts/setup_env.sh
```

## 第一个算子

### 示例: 实现 Exp 算子

#### 1. 查询算子信息

```bash
python tools/op_lookup.py exp
```

输出:
```
算子: exp
类别: unary
复杂度: simple
模板: 02_UNARY_TEMPLATE.md
所需API: DataCopy, Exp
```

#### 2. 查看模板

```bash
cat templates/02_UNARY_TEMPLATE.md
```

#### 3. 参考示例

```bash
cat examples/01_unary_exp/exp_custom.cpp
```

#### 4. 创建算子工程

```bash
mkdir my_exp_op
cp examples/01_unary_exp/exp_custom.cpp my_exp_op/
```

#### 5. 编译

```bash
bash scripts/build.sh my_exp_op
```

## 开发流程

```
1. 查表     →  python tools/op_lookup.py <op>
2. 看模板   →  cat templates/<category>_TEMPLATE.md
3. 看示例   →  cat examples/<example>/*.cpp
4. 改代码   →  只修改 Compute() 函数
5. 编译     →  bash scripts/build.sh <op_name>
6. 测试     →  bash scripts/run_test.sh <op_name>
```

## 代码结构

所有算子必须遵循:

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;  // 必须为2

class KernelXxx {
public:
    __aicore__ inline void Init(...)  { /* 初始化 */ }
    __aicore__ inline void Process() {
        for (...) {
            CopyIn();   // GM -> Local
            Compute();  // 核心计算
            CopyOut();  // Local -> GM
        }
    }
};

extern "C" __global__ __aicore__ void xxx_kernel(...) {
    KernelXxx op;
    op.Init(...);
    op.Process();
}
```

## 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| undefined reference | 缺少头文件 | 添加 `#include "kernel_operator.h"` |
| Queue死锁 | Alloc/Free不配对 | 检查每个Alloc有对应Free |
| 越界 | 未对齐 | DataCopy count需32字节对齐 |
| 精度不达标 | 数值不稳定 | Softmax先减max |
