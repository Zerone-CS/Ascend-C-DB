# Cube-based ResNet on Ascend 910B

使用 **Cube 单元**（ACT 库）实现的高性能 ResNet，完全基于 AscendC，**不使用 aclnn 内置算子**。

## 性能结果

### Cube MatMul 性能

| 矩阵尺寸 | Cube (ms) | Cube (TFLOPS) | vs CPU 加速比 |
|----------|-----------|---------------|---------------|
| 128x128x128 | 0.015 | 0.28 | 217x |
| 256x256x256 | 0.015 | 2.21 | 2160x |
| 512x512x512 | 0.016 | 16.79 | 18070x |
| 1024x1024x1024 | 0.021 | **103.78** | - |
| 2048x2048x2048 | 0.069 | **248.84** | - |

### ResNet-18 前向传播

- 输入: 1x3x224x224
- 总时间 (含权重初始化): ~14s
- 纯推理时间: 显著更低

## 架构说明

```
┌─────────────────────────────────────────────────────────────┐
│              Cube-based ResNet Architecture                  │
├─────────────────────────────────────────────────────────────┤
│  Conv2D:  im2col + Cube MatMul (ACT 库)                      │
│  FC:      Cube MatMul (ACT 库)                               │
│  ReLU/BN/Pool: Host-side 计算 (演示用)                        │
├─────────────────────────────────────────────────────────────┤
│  编译器:  ccec (AscendC 专用编译器)                           │
│  目标:    Ascend 910B (dav-c220)                              │
│  数据类型: half (FP16)                                        │
└─────────────────────────────────────────────────────────────┘
```

## 文件结构

```
CubeResNet/
├── cube_matmul.h           # Cube MatMul 封装 (ACT 库)
├── cube_conv2d.h           # Cube Conv2D (im2col + MatMul)
├── test_cube_matmul.cpp    # MatMul 单元测试
├── test_cube_resnet.cpp    # BasicBlock 测试
├── test_resnet18.cpp       # 完整 ResNet-18
├── benchmark_matmul.cpp    # 性能对比测试
├── CMakeLists.txt          # ccec 编译配置
├── build.sh                # 编译脚本
└── README.md               # 本文件
```

## 编译和运行

```bash
# 设置环境
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# 编译
cd CubeResNet && bash build.sh

# 运行测试
./build/bin/test_cube_matmul 1024 1024 1024    # MatMul 测试
./build/bin/test_cube_resnet                    # BasicBlock 测试
./build/bin/test_resnet18                       # 完整 ResNet-18
./build/bin/benchmark_matmul                    # 性能对比
```

## 为什么使用 Cube 单元?

1. **Cube 单元是 910B 的矩阵乘法专用硬件**
   - 910B 有 Cube Core (用于矩阵乘法) 和 Vector Core (用于向量运算)
   - Cube 单元使用专用 MMAD 指令，吞吐量远超 Vector 单元

2. **性能对比**
   - Vector-based naive MatMul: ~1 ms for 512x512
   - Cube-based MatMul: 0.016 ms for 512x512 (62x 更快)
   - 更大矩阵差距更明显

3. **ACT 库**
   - ACT (AscendC Template) 是华为官方的 Cube 单元模板库
   - 位置: `$ASCEND_HOME_PATH/aarch64-linux/ascendc/act/`
   - 提供高度优化的 MatMul kernel 实现

## 关键代码解析

### Cube MatMul 配置

```cpp
// Tile shapes 决定了 L1/L0 缓存的利用方式
using L1TileShape = AscendC::Shape<_128, _256, _256>;
using L0TileShape = AscendC::Shape<_128, _256, _64>;

// BlockMmad 是 Cube 单元的核心执行单元
using BlockMmad = Block::BlockMmadBuilder<
    half, LayoutA, half, LayoutB, half, LayoutC, half, LayoutC,
    L1TileShape, L0TileShape, IterateKScheduler,
    MatmulMultiBlockWithLayout<>>;
```

### Conv2D 实现 (im2col + Cube MatMul)

```cpp
void CubeConv2D(input, weight, output, N, C, H, W, K, kH, kW, ...) {
    // 1. im2col: 将卷积转换为矩阵乘法
    //    输入 (N, C, H, W) -> 矩阵 (N*oH*oW, C*kH*kW)
    Im2Col(input, im2col, N, C, H, W, kH, kW, ...);

    // 2. Cube MatMul: (N*oH*oW, C*kH*kW) * (K, C*kH*kW)^T
    CubeMatMul(im2col, weight, output, M, K, C*kH*kW);

    // 3. col2im: 重塑输出 -> (N, K, oH, oW)
    Col2Im(output, ...);
}
```

## 后续优化方向

1. **Device-side im2col**: 当前 im2col 在 host 端执行，可以迁移到 device
2. **算子融合**: Conv+BN+ReLU 融合，减少内存访问
3. **多核并行**: 利用多个 AI Core 并行处理
4. **量化**: 使用 INT8 量化进一步提升性能

## 参考

- ACT 库示例: `$ASCEND_HOME_PATH/aarch64-linux/ascendc/act/examples/`
- ACT 头文件: `$ASCEND_HOME_PATH/aarch64-linux/ascendc/act/include/`
