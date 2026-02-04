<div align="center">

# Ascend C Operator Cookbook

**🚀 基于模板的昇腾 AscendC 算子开发知识库**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CANN](https://img.shields.io/badge/CANN-8.0+-orange.svg)](https://www.hiascend.com/software/cann)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[English](README_EN.md) | 简体中文

</div>

---

## ✨ 特性

- 📚 **10+ 算子模板** — 覆盖 Softmax、Norm、Reduce、MatMul 等常见类型
- 📖 **13 个渐进示例** — 从一元运算到多核并行，分级学习
- 🛠️ **查询工具** — 一行命令获取算子分类、API、模板建议
- 🏭 **完整工程案例** — 包含 Tiling、编译、PyBind 封装的实战项目
- 🤖 **AI Agent 友好** — 结构化文档，适合 LLM 辅助开发

## 📦 项目结构

```
ascend-c-cookbook/
├── templates/              # 📋 算子模板 (10类)
│   ├── 01_SOFTMAX_TEMPLATE.md
│   ├── 02_UNARY_TEMPLATE.md
│   └── ...
├── examples/               # 📖 渐进示例 (13个)
│   ├── 01_unary_exp/
│   ├── 06_softmax/
│   └── ...
├── operators/              # 🏭 完整算子工程
│   ├── softmax_op/
│   ├── alexnet/
│   └── ...
├── tools/                  # 🛠️ 辅助工具
│   ├── op_lookup.py        # 算子查询
│   └── gen_data.py         # 测试数据生成
├── docs/                   # 📚 文档
├── scripts/                # 🚀 构建脚本
└── AGENTS.md               # 🤖 AI Agent 开发指南
```

## 🚀 快速开始

### 环境要求

- CANN 8.0+
- Ascend NPU (910A/910B/310P)
- Python 3.8+

### 安装

```bash
git clone https://github.com/your-org/ascend-c-cookbook.git
cd ascend-c-cookbook

# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 查询算子信息

```bash
python tools/op_lookup.py softmax
```

输出:
```
算子: softmax
类别: softmax
复杂度: medium
模板: 01_SOFTMAX_TEMPLATE.md
所需API: DataCopy, ReduceMax, Adds, Exp, ReduceSum, Muls
```

### 编译示例

```bash
bash scripts/build.sh examples
```

## 📖 学习路径

| 阶段 | 示例 | 核心概念 |
|------|------|----------|
| **入门** | 01-05 | TQue, DataCopy, Double Buffer, 多核 |
| **进阶** | 06-10 | Softmax, TBuf, Cast, Broadcast |
| **高级** | 11-13 | LayerNorm, RMSNorm, 动态 Tiling |

详见 [examples/README.md](examples/README.md)

## 📋 模板索引

| 模板 | 适用算子 | 复杂度 |
|--------|---------|--------|
| [SOFTMAX](templates/01_SOFTMAX_TEMPLATE.md) | softmax, log_softmax | ⭐⭐ |
| [UNARY](templates/02_UNARY_TEMPLATE.md) | exp, relu, sigmoid, gelu | ⭐ |
| [BINARY](templates/03_BINARY_TEMPLATE.md) | add, mul, div | ⭐ |
| [NORM](templates/04_NORM_TEMPLATE.md) | layernorm, rmsnorm | ⭐⭐⭐ |
| [REDUCE](templates/05_REDUCE_TEMPLATE.md) | sum, max, mean | ⭐⭐ |
| [MATMUL](templates/06_MATMUL_TEMPLATE.md) | matmul, linear | ⭐⭐⭐ |

## 🛠️ 工具集

```bash
# 算子查询 - 获取分类、API、模板
python tools/op_lookup.py <op_name>
python tools/op_lookup.py list          # 查看所有支持的算子

# 知识库搜索
python tools/query_db.py "ReduceMax"

# 测试数据生成
python tools/gen_data.py --shape 1024 --dtype float32
```

## 🏭 完整工程案例

| 项目 | 说明 | 特性 |
|--------|------|------|
| [softmax_op](operators/softmax_op/) | Softmax 算子 | 完整 Tiling + PyBind |
| [alexnet](operators/alexnet/) | AlexNet 网络 | 多算子组合 |
| [swiglu_op](operators/swiglu_op/) | SwiGLU 激活 | LLM 常用 |

## 🤖 AI Agent 集成

本项目包含 [AGENTS.md](AGENTS.md)，为 LLM 提供结构化的开发指南：

- 算子分类决策树
- 标准 Kernel 骨架
- API 速查表
- 常见错误排查
- 性能分析指南

## 📚 相关资源

- [昇腾社区](https://www.hiascend.com/) - 官方文档和工具
- [CANN 开发指南](https://www.hiascend.com/document) - 完整 API 参考
- [Ascend 样例仓库](https://gitee.com/ascend/samples) - 更多官方示例

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)。

- 🐛 报告问题
- 💡 提出建议
- 📝 改进文档
- 🚀 提交新算子示例

## 📄 许可证

[Apache License 2.0](LICENSE)

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

</div>