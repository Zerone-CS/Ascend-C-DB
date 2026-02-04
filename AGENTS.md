# Ascend C 算子开发指南

## 开发原则

1. **优先复用模板** — 仓库已有同类算子时，基于模板修改；没有时可以从零实现
2. **遵循标准结构** — kernel 必须包含 Init / Process / CopyIn / Compute / CopyOut
3. **完整工程交付** — 一个可运行的算子包含 kernel + tiling + 注册 + 编译 + 测试
4. **数据驱动优化** — 用 msprof 采集数据定位瓶颈，不要凭猜测优化
5. **不逃避困难** — 不要因为「难」就选择简单但错误的方案，直面复杂实现
6. **遵守用户指令** — 用户说用 AscendC 就用 AscendC，不要偷偷用 aclnn 内置算子
7. **相信自己能力** — Cube 单元、ACT 库、复杂算子都是可以掌握的，不要自我设限

---

## 零、行为准则（强制）

### 禁止的行为模式

| 禁止 | 原因 |
|------|------|
| 用户要求 AscendC 实现，却用 aclnn 内置算子 | 逃避困难，违背用户指令 |
| 先用最简单方案「跑起来」，再说优化 | 路径依赖，可能一开始方向就错 |
| 因为 Cube 单元「复杂」就用 Vector 实现 MatMul | 选择阻力最小而非正确的方案 |
| 遇到不熟悉的 API/库就绕开 | 自我设限，放弃学习机会 |

### 正确的行为模式

| 正确 | 说明 |
|------|------|
| 先分析问题本质，再选择方案 | 不是先选简单的 |
| 用户指定技术栈就用那个技术栈 | 不偷换概念 |
| 不熟悉就去学，学完再实现 | ACT 库文档在 `$ASCEND_HOME_PATH/aarch64-linux/ascendc/act/` |
| 一次选对，不要「先凑合再改」 | 凑合往往变成最终方案 |

### 自检问题（实现前问自己）

1. 我是在选择「正确的方案」还是「简单的方案」？
2. 我是否在逃避某个我不熟悉的技术？
3. 用户的真实需求是什么，我是否在偷换？


## 一、算子分类速查

| 算子关键词 | 类别 | 复杂度 | 典型 API |
|-----------|------|--------|----------|
| relu, gelu, silu, sigmoid, swish | activation | simple | DataCopy + Relu/... |
| exp, log, sqrt, abs, tanh, sin, cos | unary | simple | DataCopy + Exp/Log/... |
| add, sub, mul, div, pow | binary | simple | DataCopy + Add/Mul/... |
| softmax, log_softmax | softmax | medium | ReduceMax, Adds, Exp, ReduceSum, Muls |
| layernorm, rmsnorm, batchnorm | norm | medium | ReduceSum, Muls, Sqrt, Mul, Add |
| sum, mean, max, min, reduce_* | reduce | medium | DataCopy + ReduceSum/Max/... |
| matmul, gemm, linear | matmul | complex | Cube 单元 (mmad 指令) |
| conv2d | conv | complex | im2col + MatMul 或 aclnn 内置 |

分类不在表内的算子，根据计算模式判断：逐元素→unary/binary，跨元素聚合→reduce，矩阵乘→matmul。

---

## 二、Kernel 实现（Device 端）

### 2.1 标准代码结构

所有算子 kernel 遵循同一骨架，**按需修改**各部分：

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;  // double buffer，保证流水线重叠

class KernelXxx {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        // 多核分块
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = this->blockLength / TILE_SIZE;

        // 绑定全局内存
        xGm.SetGlobalBuffer((__gm__ float*)x + GetBlockIdx() * blockLength, blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + GetBlockIdx() * blockLength, blockLength);

        // 初始化 UB 缓冲区
        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_SIZE * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t idx) {
        LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        DataCopy(xLocal, xGm[idx * TILE_SIZE], TILE_SIZE);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t idx) {
        LocalTensor<float> xLocal = inQueue.DeQue<float>();
        LocalTensor<float> yLocal = outQueue.AllocTensor<float>();

        // ======== 核心计算逻辑 ========
        Relu(yLocal, xLocal, TILE_SIZE);
        // ==============================

        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t idx) {
        LocalTensor<float> yLocal = outQueue.DeQue<float>();
        DataCopy(yGm[idx * TILE_SIZE], yLocal, TILE_SIZE);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<float> xGm, yGm;
    uint32_t blockLength, tileNum;
    static constexpr uint32_t TILE_SIZE = 256;  // 每 tile 处理的元素数
};

extern "C" __global__ __aicore__ void xxx_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelXxx op;
    op.Init(x, y, tiling_data.size);
    op.Process();
}
```

### 2.2 各部分可修改范围

| 部分 | 可修改内容 | 注意事项 |
|------|-----------|----------|
| **Init** | GlobalBuffer 数量、InitBuffer 大小、成员变量 | 多输入算子需要多个 GlobalTensor |
| **CopyIn** | 搬运逻辑、搬运数量 | 复杂算子 (conv2d) 可能需要 im2col 等预处理 |
| **Compute** | 核心计算逻辑 | 可使用任意 AscendC API 组合 |
| **CopyOut** | 搬出逻辑 | 与 CopyIn 对称 |
| **Process** | 循环结构 | 多阶段算子 (softmax) 可能需要多轮循环 |
| **成员变量** | 增加 TBuf/TQue、GlobalTensor | 按需添加临时缓冲区 |

### 2.3 必须保持的约定

- Queue 操作必须配对：`AllocTensor` ↔ `FreeTensor`，`EnQue` ↔ `DeQue`
- `BUFFER_NUM = 2`（double buffer 是流水线重叠的基础，除非明确不需要流水线）
- `DataCopy` 的 count 必须 32 字节对齐（float32 至少 8 个元素）
- 入口函数签名必须是 `extern "C" __global__ __aicore__ void`

### 2.4 Softmax 示例（多 API 组合）

```cpp
__aicore__ inline void Compute() {
    LocalTensor<float> xLocal = inQueue.DeQue<float>();
    LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
    LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
    LocalTensor<float> workLocal = workBuffer.Get<float>();

    ReduceMax(workLocal, xLocal, workLocal, numClasses);     // max(x)
    float maxVal = workLocal.GetValue(0);
    Adds(tmpLocal, xLocal, -maxVal, numClasses);             // x - max
    Exp(yLocal, tmpLocal, numClasses);                       // exp(x - max)
    ReduceSum(workLocal, yLocal, workLocal, numClasses);     // sum(exp(...))
    float sumVal = workLocal.GetValue(0);
    Muls(yLocal, yLocal, 1.0f / sumVal, numClasses);         // normalize

    outQueue.EnQue(yLocal);
    inQueue.FreeTensor(xLocal);
}
```

---

## 三、Tiling + 算子注册（Host 端）

### 3.1 Tiling 数据结构

文件：`op_host/xxx_custom_tiling.h`

```cpp
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(XxxCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, size);       // 总元素数
    // 按需添加更多字段：tileNum, blockDim, ...
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(XxxCustom, XxxCustomTilingData)
}
```

### 3.2 Tiling 函数 + 算子注册

文件：`op_host/xxx_custom.cpp`

```cpp
#include "xxx_custom_tiling.h"
#include "register/op_def_registry.h"

// Tiling：运行在 CPU，根据输入 shape 动态计算分块策略
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    XxxCustomTilingData tiling;
    const auto* shape = context->GetInputShape(0);
    int32_t totalSize = 1;
    for (int i = 0; i < shape->GetStorageShape().GetDimNum(); i++)
        totalSize *= shape->GetStorageShape().GetDim(i);

    tiling.set_size(totalSize);
    context->SetBlockDim(8);  // 使用的 AI Core 数量
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

// Shape 推导
namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

// 算子注册
namespace ops {
class XxxCustom : public OpDef {
public:
    explicit XxxCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape)
             .SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(XxxCustom);
}
```

---

## 四、编译与安装

### 4.1 工程目录结构

```
xxx_op_project/
├── op_kernel/xxx_custom.cpp         # AscendC kernel（CCEC 编译 → .o）
├── op_host/xxx_custom.cpp           # Tiling + 注册（GCC 编译 → .so）
├── op_host/xxx_custom_tiling.h      # Tiling 数据结构
├── CMakeLists.txt                   # 顶层 CMake
├── CMakePresets.json                # 编译预设（SOC 型号、路径）
└── build.sh                         # 一键编译脚本
```

### 4.2 CMakePresets.json 关键配置

```json
{
    "ASCEND_COMPUTE_UNIT": "ascend910b",
    "ASCEND_CANN_PACKAGE_PATH": "/usr/local/Ascend/ascend-toolkit/latest",
    "ENABLE_BINARY_PACKAGE": true,
    "vendor_name": "customize"
}
```

### 4.3 编译

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
cd xxx_op_project && bash build.sh
```

编译工具链：

| 组件 | 编译器 | 产物 |
|------|--------|------|
| op_kernel (Device) | CCEC | `XxxCustom_*.o` (ELF, arch=0x1029, AI Core 指令集) |
| op_host (CPU) | GCC | `libcust_opmaster_rt2.0.so` (Tiling 库) |
| autogen (CPU) | GCC | `libcust_opapi.so` + `aclnn_xxx_custom.h` (自动生成) |
| 打包 | CPack | `custom_opp_*.run` (安装包) |

### 4.4 安装到 CANN

```bash
cd build_out/_CPack_Packages/Linux/External/custom_opp_*.run
bash install.sh
```

安装后的文件布局：
```
/usr/local/Ascend/.../opp/vendors/customize/
├── op_api/include/aclnn_xxx_custom.h     # 应用层引用的头文件
├── op_api/lib/libcust_opapi.so           # aclnn API 库
├── op_impl/.../kernel/ascend910b/
│   ├── XxxCustom_*.o                     # kernel 二进制
│   └── XxxCustom_*.json                  # kernel 元信息
└── op_impl/.../op_tiling/
    └── libcust_opmaster_rt2.0.so         # Tiling 库
```

CANN Runtime 启动时扫描 `opp/vendors/` 目录，自动注册所有自定义算子。

---

## 五、调用算子

### 5.1 C++ 调用（aclnn 两阶段 API）

```cpp
#include "acl/acl.h"
#include "aclnn_xxx_custom.h"  // 编译自动生成

// 阶段1：计算 workspace 大小 + 编译执行计划
aclnnXxxCustomGetWorkspaceSize(xTensor, yTensor, &workspaceSize, &executor);

// 阶段2：执行
aclnnXxxCustom(workspace, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```

### 5.2 PyBind11 封装

```cpp
// ascendc_pybind/xxx_custom_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "acl/acl.h"
#include "aclnn_xxx_custom.h"

py::array_t<float> forward(py::array_t<float> input) {
    // 1. aclrtMalloc + aclrtMemcpy (H2D)
    // 2. aclCreateTensor
    // 3. aclnnXxxCustomGetWorkspaceSize + aclnnXxxCustom
    // 4. aclrtSynchronizeStream
    // 5. aclrtMemcpy (D2H)
    return result;
}

PYBIND11_MODULE(xxx_custom_npu, m) {
    m.def("forward", &forward);
}
```

编译：
```bash
# setup.py 关键链接库
libraries=["ascendcl", "nnopbase", "acl_op_compiler", "cust_opapi"]

python3 setup.py build_ext --inplace
```

### 5.3 运行时调用链

```
Python: xxx_custom_npu.forward(x)
  │
  ▼ PyBind11
C++: aclnnXxxCustomGetWorkspaceSize()
  ├─ 加载 kernel 元信息 (JSON)
  ├─ 调用 TilingFunc() → 计算 blockDim, tileNum
  └─ 返回 workspace 大小
  │
  ▼
C++: aclnnXxxCustom()
  ├─ 加载 XxxCustom_*.o → AI Core 机器码
  └─ 下发到 NPU 的 N 个 AI Core 并行执行
  │
  ▼ NPU Hardware
8 个 AI Core 各自执行 Init → Process (CopyIn/Compute/CopyOut 循环)
```

---

## 六、性能分析（msprof）

### 6.1 Profiling 整体流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Ascend C 算子 Profiling 流程                      │
├─────────────────────────────────────────────────────────────────────┤
│  Step 1: 环境准备                                                    │
│    source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash   │
│    export LD_LIBRARY_PATH=<custom_op_api_lib>:$LD_LIBRARY_PATH      │
│                                                                     │
│  Step 2: 选择采集模式                                                │
│    ├─ msprof (通用模式)     → 采集整体性能 + 流水线利用率            │
│    ├─ msprof --instr-profiling → 采集指令级流水数据                 │
│    └─ msprof op (算子模式)  → 单算子详细分析 + 性能建议              │
│                                                                     │
│  Step 3: 分析数据                                                    │
│    ├─ CSV 文件 → 流水线利用率、带宽、耗时统计                       │
│    ├─ JSON 文件 → Chrome Tracing 可视化                             │
│    └─ SQLite DB → 详细指标查询                                       │
│                                                                     │
│  Step 4: 定位瓶颈 + 优化                                             │
│    根据 Scalar/Vector/MTE 占比判断瓶颈类型                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 采集命令详解

#### 方式一：msprof 通用采集（推荐首选）

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH

# 采集流水线利用率
msprof --output=./profiling \
       --aic-metrics=PipeUtilization \
       --ai-core=on \
       --application="./your_test_binary"
```

**关键参数说明：**
| 参数 | 说明 |
|------|------|
| `--aic-metrics=PipeUtilization` | 采集流水线占比（Scalar/Vector/MTE2/MTE3）|
| `--aic-metrics=Memory` | 采集内存带宽 |
| `--aic-metrics=MemoryUB` | 采集 UB 读写带宽 |
| `--aic-metrics=ArithmeticUtilization` | 采集算术单元利用率 |
| `--ai-core=on` | 开启 AI Core 采集 |
| `--instr-profiling=on` | 开启指令级流水采集（不能与 aic-metrics 同时使用）|

#### 方式二：msprof op 单算子分析（详细性能建议）

```bash
msprof op --output=./profiling_op \
          --aic-metrics=PipeUtilization \
          --application="./your_test_binary"
```

**msprof op 优势：**
- 自动生成性能瓶颈摘要
- 输出每个 Block 的详细流水线数据
- 提供内存带宽活跃值

#### 方式三：指令级流水采集

```bash
msprof --output=./profiling_instr \
       --instr-profiling=on \
       --application="./your_test_binary"
```

> **注意**：`--instr-profiling=on` 不能与 `--aic-metrics` 同时使用

### 6.3 产出文件说明

| 文件 | 内容 |
|------|------|
| `op_summary_*.csv` | 每次算子调用的耗时、shape、流水线占比 |
| `op_statistic_*.csv` | 按算子类型汇总：总耗时、平均、占比 |
| `task_time_*.csv` | 任务级耗时统计 |
| `api_statistic_*.csv` | ACL API 调用耗时 |
| `msprof_*.json` | Chrome Tracing 格式（`chrome://tracing` 打开） |
| `PipeUtilization.csv` | 每个 AI Core 的 Vector/Scalar/MTE2/MTE3 占比 |
| `OpBasicInfo.csv` | 算子基本信息（msprof op 产出）|
| `visualize_data.bin` | MindStudio 可视化数据（含指令级流水图） |
| `sqlite/*.db` | 详细性能数据（可用 sqlite3 查询）|

### 6.4 流水线利用率解读

**AI Vector Core 流水线单元：**
| 单元 | 功能 | 对应指标 |
|------|------|----------|
| **Scalar** | 标量计算、循环控制、地址计算 | `aiv_scalar_ratio` |
| **Vector** | 向量计算（Exp, Relu, Add 等）| `aiv_vec_ratio` |
| **MTE2** | 数据搬入（GM → UB）| `aiv_mte2_ratio` |
| **MTE3** | 数据搬出（UB → GM）| `aiv_mte3_ratio` |

**AI Core (Cube) 流水线单元：**
| 单元 | 功能 | 对应指标 |
|------|------|----------|
| **Cube** | 矩阵乘法 | `aic_cube_ratio` |
| **Scalar** | 标量计算 | `aic_scalar_ratio` |
| **MTE1** | L1 缓存搬运 | `aic_mte1_ratio` |
| **MTE2** | GM → L1/UB | `aic_mte2_ratio` |
| **FixPipe** | 后处理流水线 | `aic_fixpipe_ratio` |

### 6.5 性能瓶颈判断与优化

| 现象 | 瓶颈 | 优化方向 |
|------|------|----------|
| Scalar 占比 > 50% | 循环控制开销大 | 增大 TILE_SIZE，减少循环次数 |
| Vector 占比 < 10% | 计算单元空闲 | 增大每次计算的数据量，提高计算密度 |
| MTE2/MTE3 带宽 < 80% | 数据搬运未打满 | 增大搬运块、检查 32B 对齐 |
| TransData 占比高 | 格式转换开销 | 使用 NC1HWC0 格式贯穿网络 |
| GetWorkspaceSize 首次慢 | JIT 编译 | 预热 (warm-up) 或持久化编译结果 |
| icache_miss_rate > 0 | 指令缓存未命中 | 减少代码体积，优化分支 |

### 6.6 SQLite 数据库查询示例

```bash
# 查看表结构
sqlite3 profiling/PROF_*/device_*/sqlite/ai_core_op_summary.db '.tables'

# 查询流水线指标
sqlite3 -header -column profiling/PROF_*/device_*/sqlite/ai_core_op_summary.db \
  'SELECT * FROM ai_core_metrics;'

# 查询算子信息
sqlite3 -header -column profiling/PROF_*/device_*/sqlite/ai_core_op_summary.db \
  'SELECT op_name, op_type, block_dim, task_type FROM ge_summary;'
```

### 6.7 完整 Profiling 示例

```bash
# 1. 环境准备
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH

# 2. 采集流水线利用率
rm -rf ./prof_output
msprof --output=./prof_output \
       --aic-metrics=PipeUtilization \
       --ai-core=on \
       --application="./build/your_op_test"

# 3. 查看结果
cat ./prof_output/PROF_*/mindstudio_profiler_output/op_summary_*.csv

# 4. 单算子详细分析
rm -rf ./prof_op
msprof op --output=./prof_op \
          --aic-metrics=PipeUtilization \
          --application="./build/your_op_test"

# 5. 查看详细流水线数据
cat ./prof_op/OPPROF_*/PipeUtilization.csv
cat ./prof_op/OPPROF_*/OpBasicInfo.csv
```

### 6.8 典型分析报告示例

```
======================================================================
Sigmoid 算子指令流水分析报告
======================================================================

【算子基本信息】
  算子名称: SigmoidCustom
  算子类型: vector (AI Vector Core)
  执行时间: 2.62 us
  Block数量: 8
  运行频率: 1800 MHz

【各Block流水线利用率】
----------------------------------------------------------------------
Block  Time(us)   Cycles   Vec%     Scalar%    MTE2%    MTE3%
----------------------------------------------------------------------
0      1.804      3248     5.08     79.86      23.98    7.94
1      1.968      3543     4.54     82.92      21.54    8.04
...
----------------------------------------------------------------------
平均                        4.48     81.48      21.39    7.69

【性能瓶颈分析】
  1. Scalar单元占比最高(~80%), 说明控制流开销较大
  2. Vector单元利用率低(~4.5%), 计算密度可优化
  3. MTE2/MTE3带宽利用率低, 数据搬运未成为瓶颈

【优化建议】
  1. 增大TILE_SIZE减少循环迭代次数, 降低Scalar开销
  2. 使用流水线双缓冲技术隐藏数据搬运延迟
  3. 对于更大的输入, 增加每个Block处理的数据量
======================================================================
```

### 6.9 msprof 能分析自定义算子的原因

1. 算子通过 **aclnn API** 调用 → msprof 钩住这些 API 记录时间戳
2. kernel 安装时带有**元信息 JSON** → profiler 识别算子类型和属性
3. NPU 硬件内置 **PMU 计数器** → 采集 cycles、指令数、带宽等底层数据

---

## 七、API 速查

```cpp
// 数据搬运
DataCopy(dst, src, count);           // count 必须 32 字节对齐

// 一元运算
Exp(dst, src, count);
Log(dst, src, count);
Sqrt(dst, src, count);
Rsqrt(dst, src, count);
Abs(dst, src, count);
Relu(dst, src, count);
Not(dst, src, count);
Reciprocal(dst, src, count);

// 二元运算
Add(dst, src0, src1, count);
Sub(dst, src0, src1, count);
Mul(dst, src0, src1, count);
Div(dst, src0, src1, count);
Max(dst, src0, src1, count);
Min(dst, src0, src1, count);

// 标量运算
Adds(dst, src, scalar, count);       // dst = src + scalar
Muls(dst, src, scalar, count);       // dst = src * scalar

// 规约运算
ReduceSum(dst, src, workBuffer, count);
ReduceMax(dst, src, workBuffer, count);
ReduceMin(dst, src, workBuffer, count);

// 数据类型
Cast(dst, src, dstType, count);      // 类型转换

// 全局内存
GlobalTensor<T> gm;
gm.SetGlobalBuffer((__gm__ T*)addr, length);

// 队列操作（必须配对）
LocalTensor<T> t = queue.AllocTensor<T>();  // 分配
queue.EnQue(t);                             // 入队
LocalTensor<T> t = queue.DeQue<T>();        // 出队
queue.FreeTensor(t);                        // 释放

// 临时缓冲区（不走队列）
TBuf<QuePosition::VECCALC> tmpBuf;
pipe.InitBuffer(tmpBuf, size);
LocalTensor<T> t = tmpBuf.Get<T>();
```

---

## 八、编译前自检清单

- [ ] `#include "kernel_operator.h"` + `using namespace AscendC`
- [ ] `BUFFER_NUM = 2`
- [ ] 每个输入/输出都有 `SetGlobalBuffer`
- [ ] 每个 Queue 都有 `pipe.InitBuffer`
- [ ] Queue 操作配对：`Alloc ↔ Free`、`EnQue ↔ DeQue`
- [ ] `DataCopy` 的 count 是 8 的倍数（float32 下 32 字节对齐）
- [ ] 无硬编码 shape 值，通过 tiling 参数传入
- [ ] 入口函数有 `GM_ADDR workspace, GM_ADDR tiling` 参数
- [ ] Host 端有 `OP_ADD(XxxCustom)` 注册

---

## 九、常见错误速查

| 错误现象 | 原因 | 修复 |
|---------|------|------|
| undefined reference to kernel | 入口函数缺 `extern "C"` | 加 `extern "C" __global__ __aicore__` |
| tiling data size mismatch | Tiling 结构体字段与 kernel 不一致 | 检查 `.h` 和两端的字段定义 |
| 精度不达标 (softmax) | 未做数值稳定性处理 | 先 ReduceMax 减最大值再 Exp |
| 死锁/挂起 | Queue Alloc/Free 不配对 | 逐行检查 EnQue/DeQue/Free 顺序 |
| 结果全零 | DataCopy count 为 0 或 offset 错误 | 检查 tileNum/blockLength 计算 |
| 越界 | 搬运 count 未对齐 | 确保 count 是 8 的倍数 (float32) |
| install.sh 后找不到算子 | LD_LIBRARY_PATH 未设置 | `export LD_LIBRARY_PATH=.../customize/op_api/lib/:$LD_LIBRARY_PATH` |
| msprof 无数据 | kernel-name 不匹配 | 用 `--kernel-name=XxxCustom`（首字母大写，不带 _custom 后缀） |
| ACL_ERROR 507015 | Cube 单元访问非法地址（tiling 错误或 Matmul 库未初始化） | 见第十三章详细排查 |

---

## 十、完整流程一图总结

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  op_kernel/xxx_custom.cpp          AscendC kernel              │
  │  op_host/xxx_custom.cpp            Tiling + 注册               │
  │  op_host/xxx_custom_tiling.h       Tiling 数据结构              │
  └──────────────┬──────────────────────────────────────────────────┘
                 │ bash build.sh
                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  CCEC 编译 kernel → XxxCustom_*.o      (AI Core 机器码)         │
  │  GCC  编译 host   → libcust_opmaster_rt2.0.so (Tiling)         │
  │  自动生成          → aclnn_xxx_custom.h + libcust_opapi.so     │
  │  CPack 打包        → custom_opp_*.run                          │
  └──────────────┬──────────────────────────────────────────────────┘
                 │ bash install.sh
                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  CANN Runtime: opp/vendors/customize/                          │
  │    kernel/*.o + *.json  │  libcust_opapi.so  │  Tiling .so     │
  └──────────────┬──────────────────────────────────────────────────┘
                 │ python3 setup.py build_ext --inplace (可选)
                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  调用: XxxCuaclnnstom() 或 PyBind11 封装                        │
  └──────────────┬──────────────────────────────────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  NPU 执行: CANN 加载 kernel → N 个 AI Core 并行                 │
  └──────────────┬──────────────────────────────────────────────────┘
                 │ msprof
                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  性能分析: op_summary.csv / PipeUtilization.csv / timeline.json │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 十一、快速复现命令

```bash
# 0. 环境
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH

# 1. 编译
cd xxx_op_project && bash build.sh

# 2. 安装
cd build_out/_CPack_Packages/Linux/External/custom_opp_*.run && bash install.sh

# 3. 编译 PyBind11（可选）
cd ascendc_pybind && python3 setup.py build_ext --inplace

# 4. 运行
python3 demo.py

# 5. 性能分析
msprof --output=./prof --task-time=on --ai-core=on \
       --aic-metrics=PipeUtilization \
       --application="python3 demo.py"
msprof --export=on --output=./prof/PROF_*
```

---

## 十二、实战参考

完整可运行案例见 `operators/alexnet/`：

| 路径 | 内容 |
|------|------|
| `relu_op_project/` | 自定义 ReLU 算子工程（kernel + tiling + 注册 + 编译） |
| `ascendc_pybind/` | PyBind11 封装，Python 调用自定义算子 |
| `real_npu/alexnet_full_npu.cpp` | 用 aclnn 内置算子组合完整 AlexNet |
| `demo_ascendc_relu.py` | 自定义 ReLU 的 Python 演示 |
| `profiling_*/` | msprof 采集的性能数据 |
| `softmax_kernel.cpp` | Softmax 算子实现（ReduceMax + Exp + ReduceSum 组合） |
| `conv2d_kernel.cpp` | Conv2D 算子实现（im2col + MatMul） |

---

## 十三、910B MatMul 算子开发（LLM 工作流）

### 触发条件

当用户请求涉及以下关键词时，使用本章指导：
- `matmul`、`gemm`、`矩阵乘法`、`linear`
- `910B` + `Cube`
- `ACL_ERROR 507015`
- `CCU instruction address check error`

---

### 决策树：选择实现方式

```
用户需求
    │
    ├─→ "直接调用/快速实现/生产环境"
    │       → 推荐 aclnnMatMul（内置算子）
    │       → 无需写 kernel
    │
    ├─→ "学习 AscendC/自定义实现/调试"
    │       → 推荐 Vector 指令实现
    │       → 使用模板 [Template A]
    │
    └─→ "高性能/算子融合/FlashAttention"
            → 推荐 Matmul 库 + KFC
            → 使用模板 [Template B]
```

---

### Template A：Vector 指令实现（推荐用于学习/调试）

#### 文件结构

```
MatmulCustom/
├── op_kernel/matmul_custom.cpp      # [复制模板 A1]
├── op_host/matmul_custom.cpp        # [复制模板 A2]
├── op_host/matmul_custom_tiling.h   # [复制模板 A3]
├── CMakeLists.txt                   # 从现有算子复制
├── CMakePresets.json                # 从现有算子复制
└── build.sh                         # 从现有算子复制
```

#### 模板 A1：Kernel 实现

```cpp
// op_kernel/matmul_custom.cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelMatmul {
public:
    __aicore__ inline KernelMatmul() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                 int32_t M, int32_t N, int32_t K) {
        this->M = M; this->N = N; this->K = K;
        aGm.SetGlobalBuffer((__gm__ half*)a, M * K);
        bGm.SetGlobalBuffer((__gm__ half*)b, K * N);
        cGm.SetGlobalBuffer((__gm__ float*)c, M * N);
        
        alignedK = ((K + 15) / 16) * 16;
        alignedN = ((N + 7) / 8) * 8;
        pipe.InitBuffer(inQueueA, BUFFER_NUM, alignedK * sizeof(half));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, alignedK * sizeof(half));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, alignedN * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < M; i++) {
            ComputeRow(i);
        }
    }

private:
    __aicore__ inline void ComputeRow(int32_t row) {
        LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();
        LocalTensor<half> aRow = inQueueA.AllocTensor<half>();
        DataCopy(aRow, aGm[row * K], alignedK);
        inQueueA.EnQue(aRow);
        aRow = inQueueA.DeQue<half>();
        
        for (int32_t j = 0; j < N; j++) {
            LocalTensor<half> bCol = inQueueB.AllocTensor<half>();
            for (int32_t k = 0; k < K; k++) {
                bCol.SetValue(k, bGm.GetValue(k * N + j));
            }
            for (int32_t k = K; k < alignedK; k++) {
                bCol.SetValue(k, (half)0.0f);
            }
            inQueueB.EnQue(bCol);
            bCol = inQueueB.DeQue<half>();
            
            float sum = 0.0f;
            for (int32_t k = 0; k < K; k++) {
                sum += (float)aRow.GetValue(k) * (float)bCol.GetValue(k);
            }
            cLocal.SetValue(j, sum);
            inQueueB.FreeTensor(bCol);
        }
        
        for (int32_t j = N; j < alignedN; j++) cLocal.SetValue(j, 0.0f);
        inQueueA.FreeTensor(aRow);
        outQueueC.EnQue(cLocal);
        cLocal = outQueueC.DeQue<float>();
        DataCopy(cGm[row * N], cLocal, alignedN);
        outQueueC.FreeTensor(cLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA, inQueueB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueC;
    GlobalTensor<half> aGm, bGm;
    GlobalTensor<float> cGm;
    int32_t M, N, K, alignedK, alignedN;
};

extern "C" __global__ __aicore__ void matmul_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelMatmul op;
    op.Init(a, b, c, tilingData.M, tilingData.N, tilingData.K);
    op.Process();
}
```

#### 模板 A2：Host 实现

```cpp
// op_host/matmul_custom.cpp
#include "matmul_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    auto aShape = context->GetInputShape(0);
    auto bShape = context->GetInputShape(1);
    int32_t M = aShape->GetStorageShape().GetDim(0);
    int32_t K = aShape->GetStorageShape().GetDim(1);
    int32_t N = bShape->GetStorageShape().GetDim(1);
    
    MatmulCustomTilingData tiling;
    tiling.set_M(M);
    tiling.set_N(N);
    tiling.set_K(K);
    tiling.set_usedCoreNum(1);
    context->SetBlockDim(1);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* ctx) {
    auto cShape = ctx->GetOutputShape(0);
    cShape->SetDimNum(2);
    cShape->SetDim(0, ctx->GetInputShape(0)->GetDim(0));
    cShape->SetDim(1, ctx->GetInputShape(1)->GetDim(1));
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* ctx) {
    ctx->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class MatmulCustom : public OpDef {
public:
    explicit MatmulCustom(const char* name) : OpDef(name) {
        this->Input("a").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("b").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("c").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(MatmulCustom);
}
```

#### 模板 A3：Tiling 定义

```cpp
// op_host/matmul_custom_tiling.h
#ifndef MATMUL_CUSTOM_TILING_H
#define MATMUL_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulCustomTilingData)
    TILING_DATA_FIELD_DEF(int32_t, M);
    TILING_DATA_FIELD_DEF(int32_t, N);
    TILING_DATA_FIELD_DEF(int32_t, K);
    TILING_DATA_FIELD_DEF(int32_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulCustom, MatmulCustomTilingData)
}
#endif
```

---

### Template B：Matmul 库 + KFC（高性能场景）

```cpp
// op_kernel/matmul_custom.cpp
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;

class MyMatmulOp {
public:
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, half>,
           MatmulType<TPosition::GM, CubeFormat::ND, half>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>> mm;
           
    __aicore__ inline void Process(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                    const TCubeTiling* tiling) {
        GlobalTensor<half> aGm, bGm;
        GlobalTensor<float> cGm;
        aGm.SetGlobalBuffer((__gm__ half*)a, tiling->M * tiling->Ka);
        bGm.SetGlobalBuffer((__gm__ half*)b, tiling->Ka * tiling->N);
        cGm.SetGlobalBuffer((__gm__ float*)c, tiling->M * tiling->N);
        
        mm.SetOrgShape(tiling->M, tiling->N, tiling->Ka);
        mm.SetSingleShape(tiling->singleCoreM, tiling->singleCoreN, tiling->singleCoreK);
        mm.SetTensorA(aGm, false);
        mm.SetTensorB(bGm, false);
        mm.IterateAll(cGm);
    }
};

extern "C" __global__ __aicore__ void matmul_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(TCubeTiling, tilingData, tiling);
    
    MyMatmulOp op;
    // 关键：必须用此宏设置 KFC 分核通信
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm, &tilingData);
    op.Process(a, b, c, &tilingData);
}
```

---

### 错误处理：ACL_ERROR 507015

#### 识别特征

```
Synchronize ret: 507015
CCU instruction address check error
Aicore kernel execute failed
```

#### 排查步骤

```
Step 1: 确认是否使用 Matmul 库
        ├─ 是 → 检查是否有 REGIST_MATMUL_OBJ 宏调用
        │       ├─ 没有 → 添加该宏，或改用 Template A
        │       └─ 有   → 检查 tiling 参数是否正确
        └─ 否 → 进入 Step 2

Step 2: 验证 tiling 数据传递
        ├─ Host 端是否调用了 SaveToBuffer？
        ├─ REGISTER_TILING_DATA_CLASS 名称是否与算子名一致？
        └─ Kernel 端是否用 GET_TILING_DATA 解析？

Step 3: 检查内存对齐
        ├─ DataCopy count 是否 32 字节对齐？
        │   - half: 需要 16 的倍数
        │   - float: 需要 8 的倍数
        └─ GlobalTensor 长度是否计算正确？

Step 4: 使用 Template A 隔离问题
        └─ 如果 Vector 版本能运行 → 问题在 Matmul 库配置
           如果 Vector 版本也失败 → 问题在 tiling 或内存
```

#### 快速修复方案

| 现象 | 原因 | 修复 |
|-----|------|------|
| 所有 Matmul API 无效果 | 910B 默认 enableMixDualMaster=false | 添加 REGIST_MATMUL_OBJ 或改用 Vector 实现 |
| 编译报 "do not register tiling struct" | REGISTER 名称不匹配 | 确保名称与算子类名完全一致 |
| 运行时数据错乱 | Host/Device tiling 字段不一致 | 两端使用同一个 .h 文件 |

---

### 验证命令

```bash
# 编译
cd MatmulCustom && bash build.sh

# 安装
cd build_out/_CPack_Packages/Linux/External/custom_opp_*.run
bash install.sh

# 设置环境
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH

# 运行测试
./verify_test
# 期望输出: Test: PASSED
```

---

### 参考实现

| 路径 | 用途 |
|------|------|
| `operators/matmul_op/MatmulCustom/` | 完整 Vector 实现工程 |
| `operators/matmul_op/verify_test.cpp` | 数值正确性测试 |
| `operators/softmax_op/SoftmaxCustom/` | 可参考的工程结构模板 |


---

## 十四、端到端模型开发（Transformer/LLM 工作流）

### 触发条件

当用户请求涉及以下关键词时，使用本章指导：
- `GPT`、`Transformer`、`LLM`、`大模型`
- `多算子组合`、`端到端`、`完整模型`
- `Attention`、`FFN`、`Embedding`

---

### 决策树：开发策略选择

```
用户需求
    │
    ├─→ "快速验证/原型开发"
    │       → 阶段1: CPU NumPy 参考实现
    │       → 先跑通逻辑，再迁移 NPU
    │
    ├─→ "单算子开发/调试"
    │       → 阶段2: AscendC 单算子实现
    │       → 对比 CPU 结果验证精度
    │
    ├─→ "端到端推理"
    │       → 阶段3: ACL API 组装算子
    │       → 关注内存管理和数据搬运
    │
    └─→ "生产环境/性能优化"
            → 阶段4: msprof 分析 + 算子融合
            → 参考第六章性能分析
```

**核心原则**：先跑通，再优化。每个阶段验证通过后再进入下一阶段。

---

### Transformer 算子划分

| 模块 | 所需算子 | 复杂度 | 实现建议 |
|------|---------|--------|----------|
| Embedding | `embedding_lookup` | simple | Vector 内存索引 |
| LayerNorm | `layernorm_custom` | medium | ReduceSum + Muls + Sqrt |
| Attention QKV | `matmul` | complex | aclnn 内置或 Vector 实现 |
| Attention Softmax | `softmax_custom` | medium | ReduceMax + Exp + ReduceSum |
| FFN | `matmul` + `gelu` | complex | GELU 用快速近似 |
| Residual | `add_custom` | simple | 逐元素加法 |

---

### 关键代码模板

#### GELU 快速近似（避免 erf）

```cpp
// GELU(x) ≈ x * sigmoid(1.702 * x)
Muls(tmp, xLocal, 1.702f, len);
Muls(tmp, tmp, -1.0f, len);
Exp(tmp, tmp, len);
Adds(tmp, tmp, 1.0f, len);
Reciprocal(tmp, tmp, len);
Mul(yLocal, xLocal, tmp, len);
```

#### ACL Python 环境配置

```python
import os, sys
ASCEND_PATH = "/usr/local/Ascend/ascend-toolkit/latest"
os.environ["LD_LIBRARY_PATH"] = f"{ASCEND_PATH}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, f"{ASCEND_PATH}/pyACL/python/site-packages")
import acl  # 必须在设置环境变量后导入
```

#### NPU 设备选择

```bash
npu-smi info  # 查看 HBM-Usage，选择占用低的设备
```

---

### 32 字节对齐速查

| 数据类型 | 最小元素数 | 对齐公式 |
|---------|-----------|----------|
| float32 | 8 | `((len + 7) / 8) * 8` |
| float16/half | 16 | `((len + 15) / 16) * 16` |
| int8 | 32 | `((len + 31) / 32) * 32` |

**必须在 Init 中计算对齐大小**，否则 `DataCopy` 会报错或结果异常。

---

### 端到端开发自检清单

- [ ] CPU 参考实现已验证算法正确性
- [ ] 每个算子单独测试通过，精度对齐 CPU
- [ ] 所有 `DataCopy` 的 count 已 32 字节对齐
- [ ] Softmax/LayerNorm 已做数值稳定性处理（减最大值）
- [ ] ACL 资源有 `try-finally` 保证释放
- [ ] NPU 设备可用（`npu-smi info` 检查）
- [ ] `LD_LIBRARY_PATH` 已正确设置

---

### 常见错误速查（端到端场景）

| 错误现象 | 原因 | 修复 |
|---------|------|------|
| `import acl` 失败 | LD_LIBRARY_PATH 未设置 | 先设置环境变量再 import |
| 结果全零 | DataCopy offset/count 计算错误 | 检查 `row * rowSize` 等索引 |
| 结果 NaN/Inf | Exp 输入过大或除零 | Softmax 先减最大值 |
| 精度不达标 | 未做数值稳定性处理 | LayerNorm/Softmax 参考模板 |
| 程序挂起 | Queue Alloc/Free 不配对 | 检查 EnQue↔DeQue 顺序 |
| ACL 返回非零 | 设备占用或内存不足 | 换设备或释放旧资源 |
| 生成乱码 | 模型权重未训练 | 随机权重预期行为，非错误 |

---

### 参考实现

| 路径 | 内容 |
|------|------|
| `operators/minigpt/` | 完整 MiniGPT AscendC 实现 |
| `operators/minigpt/op_kernel/` | 6 个核心算子: gelu, layernorm, softmax, matmul, add, embedding |
| `operators/minigpt/tests/minigpt_inference.py` | CPU NumPy 参考实现 |
| `operators/minigpt/tests/minigpt_npu.py` | ACL Python NPU 推理 |

```bash
# 运行测试
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
cd operators/minigpt/tests && python3 test_minigpt.py
```
