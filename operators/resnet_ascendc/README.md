# AscendC 算子开发 vs aclnn API 调用

## 核心区别

| 方式 | 说明 | 类比 |
|------|------|------|
| **aclnn API** | 调用 CANN 内置的预编译算子 | 类似调用 cuDNN |
| **AscendC** | 自己写 kernel 代码，在 AI Core 上执行 | 类似写 CUDA kernel |

---

## AscendC 开发完整流程

```
╒══════════════════════════════════════════════════╕
│              1. 编写 AscendC Kernel                  │
│      op_kernel/relu_custom.cpp                       │
├──────────────────────────────────────────────────┤
│  class KernelRelu {                                   │
│      Init()    → 绑定 GM + 初始化缓冲区               │
│      Process() → 主循环                               │
│      CopyIn()  → GM -> UB                            │
│      Compute() → Relu(y, x, len)  ← AscendC API     │
│      CopyOut() → UB -> GM                            │
│  }                                                    │
╘══════════════════════════════════════════════════╛
                        ↓
╒══════════════════════════════════════════════════╕
│              2. 编写 Host 端代码                       │
│      op_host/relu_custom.cpp + _tiling.h             │
├──────────────────────────────────────────────────┤
│  TilingFunc()  → 计算分块策略                       │
│  InferShape()  → 推导输出 shape                      │
│  OP_ADD()      → 注册算子                             │
╘══════════════════════════════════════════════════╛
                        ↓
╒══════════════════════════════════════════════════╕
│              3. 编译                                  │
├──────────────────────────────────────────────────┤
│  CCEC 编译 kernel → ReluCustom_*.o (AI Core 机器码)  │
│  GCC  编译 host   → libcust_opmaster_rt2.0.so       │
│  自动生成         → aclnn_relu_custom.h + .so        │
╘══════════════════════════════════════════════════╛
                        ↓
╒══════════════════════════════════════════════════╕
│              4. 安装到 CANN                           │
├──────────────────────────────────────────────────┤
│  bash install.sh → 安装到 opp/vendors/customize/     │
╘══════════════════════════════════════════════════╛
                        ↓
╒══════════════════════════════════════════════════╕
│              5. 调用自定义算子                        │
├──────────────────────────────────────────────────┤
│  aclnnReluCustomGetWorkspaceSize(x, y, &ws, &exec);  │
│  aclnnReluCustom(workspace, ws, exec, stream);       │
╘══════════════════════════════════════════════════╛
```

---

## 本目录文件结构

```
resnet_ascendc/
├── kernels/                      # AscendC kernel 源码
│   ├── relu_kernel.cpp           # ReLU: y = max(0, x)
│   ├── add_kernel.cpp            # Add: y = x1 + x2 (残差连接)
│   ├── batchnorm_kernel.cpp      # BatchNorm: y = gamma*(x-mean)/sqrt(var+eps) + beta
│   ├── maxpool_kernel.cpp        # MaxPool2D: 2D 滑动窗口池化
│   ├── avgpool_global_kernel.cpp # GlobalAvgPool: 全局平均池化
│   ├── conv2d_kernel.cpp         # Conv2D (朴素实现, 生产用 Cube 单元)
│   └── fc_kernel.cpp             # FC/MatMul (朴素实现)
│
├── ReluCustom/                   # 完整算子工程示例
│   ├── op_kernel/relu_custom.cpp # Device 端 kernel
│   ├── op_host/relu_custom.cpp   # Host 端 tiling + 注册
│   ├── op_host/relu_custom_tiling.h
│   ├── CMakeLists.txt
│   ├── CMakePresets.json
│   └── build.sh
│
├── test_relu.cpp                 # 测试程序
└── CMakeLists.txt
```

---

## AscendC Kernel 核心概念

```cpp
// 1. 流水线管理器 - 管理缓冲区和队列
TPipe pipe;

// 2. 双缓冲队列 - 实现计算/搬运重叠
TQue<QuePosition::VECIN, 2> inQueue;   // 输入队列
TQue<QuePosition::VECOUT, 2> outQueue; // 输出队列

// 3. 全局内存 (GM) 绑定
GlobalTensor<float> xGm, yGm;
xGm.SetGlobalBuffer((__gm__ float*)x, length);

// 4. 缓冲区初始化
pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_SIZE * sizeof(float));

// 5. 队列操作 (必须配对)
LocalTensor<float> local = inQueue.AllocTensor<float>();  // 分配
inQueue.EnQue(local);                                      // 入队
local = inQueue.DeQue<float>();                           // 出队
inQueue.FreeTensor(local);                                // 释放

// 6. 数据搬运 (32字节对齐)
DataCopy(localTensor, gmTensor[offset], alignedCount);

// 7. 向量计算 API
Relu(yLocal, xLocal, count);      // y = max(0, x)
Add(yLocal, x1, x2, count);       // y = x1 + x2
Muls(yLocal, xLocal, scalar, count);  // y = x * scalar
Adds(yLocal, xLocal, scalar, count);  // y = x + scalar
```

---

## 运行测试

```bash
# 1. 编译自定义算子
cd ReluCustom && bash build.sh

# 2. 安装算子
cd build_out/_CPack_Packages/Linux/External/custom_opp_*.run
bash install.sh

# 3. 编译测试
cd .. && mkdir build && cd build
cmake .. && make

# 4. 运行测试
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
./test_relu
```

---

## 实际中的 ResNet 实现策略

| 算子 | 推荐实现 | 原因 |
|------|----------|------|
| ReLU | AscendC 自定义 | 简单，展示核心概念 |
| Add | AscendC 自定义 | 残差连接的核心 |
| BatchNorm | AscendC 自定义 | 中等复杂度，向量操作组合 |
| Conv2D | **aclnnConvolution** | 极其复杂，需要 Cube 单元 + im2col |
| MaxPool | AscendC 或 aclnn | 中等复杂度 |
| AvgPool | AscendC 自定义 | Reduce 操作 |
| MatMul/FC | **aclnnMatmul** | 需要 Cube 单元优化 |

**建议**: 生产环境中，Conv2D 和 MatMul 使用 CANN 内置算子（经过深度优化），其他算子可根据需求自定义。

---

## 测试结果

```
=== AscendC 自定义 ReLU 算子测试 ===

[第1阶段] aclnnReluCustomGetWorkspaceSize...
  workspaceSize = 0 bytes
[第2阶段] aclnnReluCustom 执行计算...

=== 结果验证 ===
输入 (10 个): -20.0 -19.0 -18.0 -17.0 -16.0 -15.0 -14.0 -13.0 -12.0 -11.0
输出 (10 个): 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
期望 (10 个): 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

最大误差: 0.000000
错误数量 (>1e-5): 0/2048

测试: ✅ PASSED
```
