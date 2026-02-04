# ResNet-18 on Ascend NPU

使用 aclnn API 在昇腾 910B NPU 上实现完整的 ResNet-18 推理。

## 网络结构

```
输入: [1, 3, 224, 224]
  │
  ├─ conv1: 7×7, 64 filters, stride=2, pad=3
  ├─ bn1 + relu
  ├─ maxpool: 3×3, stride=2, pad=1
  │
  ├─ layer1: 2× BasicBlock (64→64)
  ├─ layer2: 2× BasicBlock (64→128, first stride=2)
  ├─ layer3: 2× BasicBlock (128→256, first stride=2)
  ├─ layer4: 2× BasicBlock (256→512, first stride=2)
  │
  ├─ avgpool: global (1×1)
  └─ fc: 512→1000

输出: [1, 1000]
```

## 使用的 aclnn API

| 算子 | API |
|------|-----|
| Conv2D | `aclnnConvolution` |
| BatchNorm | `aclnnBatchNorm` |
| ReLU | `aclnnRelu` |
| Add | `aclnnAdd` |
| MaxPool | `aclnnMaxPool` |
| AdaptiveAvgPool2D | `aclnnAdaptiveAvgPool2d` |
| MatMul | `aclnnMatmul` |

## 编译

```bash
# 设置环境
export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash

# 编译
bash build.sh
```

## 运行

```bash
# 默认使用 device 2
./build/resnet18_npu

# 指定 device
./build/resnet18_npu <device_id>
```

## 性能

在 Ascend 910B 上的测试结果：

| 指标 | 值 |
|------|----|
| 首次推理 | ~323 ms (含 JIT 编译) |
| 后续推理 | ~1.9-2.5 ms |
| 平均延迟 | ~2.0 ms |

## 文件结构

```
resnet/
├── resnet18_npu.cpp    # ResNet-18 主程序
├── CMakeLists.txt      # CMake 配置
├── build.sh            # 编译脚本
└── README.md           # 本文件
```

## 扩展

- 修改 `numClasses` 参数可适配不同分类任务
- 可加载预训练权重替代随机初始化
- 可修改为 ResNet-34/50/101/152
