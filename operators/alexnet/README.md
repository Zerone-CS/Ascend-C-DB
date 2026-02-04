# AlexNet AscendC Operator Project

基于 Ascend 910B2 NPU 的自定义算子开发示例，包含完整的 AscendC kernel 实现、编译部署和 Python 调用流程。

## 目录结构

```
alexnet/
├── kernels/                 # AscendC kernel 源码
│   ├── relu/               # ReLU 算子
│   ├── conv2d/             # Conv2D 算子
│   ├── maxpool2d/          # MaxPool2D 算子
│   ├── linear/             # Linear (全连接) 算子
│   └── softmax/            # Softmax 算子
│
├── op_projects/             # 完整可编译的算子工程
│   └── relu/               # ReLU 算子工程 (含 tiling + 注册)
│       ├── op_kernel/      # Device 端 kernel
│       ├── op_host/        # Host 端 tiling + 注册
│       └── build.sh        # 编译脚本
│
├── bindings/                # Python 绑定
│   └── pybind/             # PyBind11 封装
│       ├── relu_custom_pybind.cpp
│       └── setup.py
│
├── examples/                # 示例代码
│   ├── cpp/                # C++ 示例
│   │   └── alexnet_full_npu.cpp
│   └── python/             # Python 示例
│       └── demo_ascendc_relu.py
│
├── tests/                   # 测试代码
│   ├── test_relu_*.cpp
│   └── test_relu_custom.py
│
├── scripts/                 # 构建和运行脚本
│   ├── build_and_run.sh
│   └── run_real_npu.sh
│
├── tools/                   # 分析工具
│   ├── visualize_pipeline.py
│   └── generate_timeline.py
│
├── profiling/               # 性能分析数据
│
└── data/                    # 输入输出数据
    ├── input/
    └── output/
```

## 快速开始

```bash
# 1. 环境设置
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH

# 2. 编译算子
cd op_projects/relu && bash build.sh

# 3. 安装算子
cd build_out/_CPack_Packages/Linux/External/custom_opp_*.run && bash install.sh

# 4. 编译 PyBind11 (可选)
cd ../../../../bindings/pybind && python3 setup.py build_ext --inplace

# 5. 运行测试
python3 examples/python/demo_ascendc_relu.py
```

## 开发流程

```
kernels/*.cpp          编写 AscendC kernel
       ↓
op_projects/           添加 tiling + 注册
       ↓
bash build.sh          CCEC 编译 → .o + .so
       ↓
bash install.sh        部署到 CANN
       ↓
bindings/pybind/       PyBind11 封装 (可选)
       ↓
examples/              运行测试
       ↓
tools/                 msprof 性能分析
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `kernels/relu/relu_kernel.cpp` | ReLU AscendC kernel 实现 |
| `op_projects/relu/op_kernel/relu_custom.cpp` | 完整 ReLU 工程 kernel |
| `op_projects/relu/op_host/relu_custom.cpp` | Tiling + Op 注册 |
| `bindings/pybind/relu_custom_pybind.cpp` | PyBind11 封装 |
| `examples/cpp/alexnet_full_npu.cpp` | 完整 AlexNet 示例 |
| `examples/python/demo_ascendc_relu.py` | Python 调用示例 |

## 性能分析

```bash
# 采集数据
msprof --output=./profiling/new_prof \
       --task-time=on --ai-core=on \
       --aic-metrics=PipeUtilization \
       --application="python3 examples/python/demo_ascendc_relu.py"

# 导出报告
msprof --export=on --output=./profiling/new_prof/PROF_*

# 可视化流水线
python3 tools/visualize_pipeline.py
```

## 环境要求

| 项目 | 版本 |
|------|------|
| Device | Ascend 910B2 |
| CANN | 8.3.RC1 |
| Python | 3.9+ |
| PyBind11 | 2.10+ |

## 添加新算子

1. 在 `kernels/<op_name>/` 下编写 kernel
2. 复制 `op_projects/relu/` 为模板，修改为新算子
3. 更新 `op_host/*.cpp` 中的 tiling 逻辑和注册
4. 编译安装测试

示例模板参见 `kernels/` 下的各算子实现。

## 参考文档

- [AscendC 开发指南](../../AGENTS.md)
- [CANN 算子开发文档](https://www.hiascend.com/document)
