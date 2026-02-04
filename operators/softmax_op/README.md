# Ascend C Softmax 算子工程

## 概述
本工程实现了基于Ascend C的Softmax算子，支持在昇腾NPU (910B2)上运行。

## 工程结构
```
softmax_op/
├── softmax_custom.cpp      # Kernel侧Ascend C实现
├── main.cpp                # Host侧调用程序
├── data_utils.h            # 数据读写工具
├── CMakeLists.txt          # 编译配置
├── run.sh                  # 编译运行脚本
├── SoftmaxCustom/          # msopgen生成的标准算子工程
│   ├── op_kernel/          # Kernel代码
│   └── op_host/            # Host侧tiling代码
├── scripts/
│   ├── gen_data.py         # 生成测试数据
│   └── verify_result.py    # 验证结果
└── test_softmax_torch.py   # PyTorch NPU测试
```

## Softmax算法实现
核心算法流程:
1. ReduceMax: 对每行求最大值 max
2. Adds+Exp: 计算 exp(x - max)
3. ReduceSum: 对exp结果求和 sum
4. Muls: 计算 exp(x-max) / sum

计算公式:
```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
```

## 使用方法

### 1. PyTorch NPU测试（推荐）
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 test_softmax_torch.py
```

### 2. 标准算子工程编译
```bash
cd SoftmaxCustom
bash build.sh
```

## 测试结果
- 测试shape: [8, 256]
- 数据类型: float32
- 最大绝对误差: < 1e-8
- 测试状态: **PASSED**

## 依赖环境
- CANN 8.3.RC1
- NPU: Ascend 910B2
- torch_npu 2.1.0.post8
