# SwiGLU 算子实现与运行指南

## 实现概述

已完成 SwiGLU 算子的 Ascend C 实现，代码位于 `swiglu_custom.cpp`。

###算法公式
```
SwiGLU(x, gate) = x * gate * sigmoid(gate)
sigmoid(gate) = 1 / (1 + exp(-gate))
```

### 实现步骤
1. `tmpLocal1 = -gate`  (Muls)
2. `tmpLocal1 = exp(-gate)`  (Exp)
3. `tmpLocal1 = 1 + exp(-gate)`  (Adds)
4. `tmpLocal2 = gate / (1 + exp(-gate))`  (Div) → gate × sigmoid(gate)
5. `yLocal = x * tmpLocal2`  (Mul) → 最终结果

### 代码特点
- 遵循 AGENTS.md 指南，基于模板修改
- 使用 double buffer (BUFFER_NUM = 2)
- Tile-based 处理 (tileSize = 256)
- 遵循标准流程：CopyIn → Compute → CopyOut

## 当前状态

✅ **编译成功**：内核编译通过，生成 `swiglu_custom_kernel.o` (72KB)
❌ **运行时错误**：执行时报错 `aclrtSynchronizeStream failed, ret=507015`

## 问题诊断

### 已测试的解决方案
1. ✅ 切换到不同的 NPU 设备 (0, 1, 2, 7) → 所有设备同样错误
2. ✅ 简化内核实现（仅数据拷贝）→ 仍然失败
3. ✅ 验证 ACL 基本功能 → 正常工作
4. ✅ 测试其他算子 (add_op) → 同样失败

### 错误分析

**错误码 507015** 表示内核执行失败（EZ0044）。由于：
- ACL 初始化、设备设置、内存分配均成功
- 内核二进制加载和函数句柄获取成功
- 只在 `aclrtSynchronizeStream` 时失败
- 所有算子（包括已有的 add_op）都有同样问题

**结论**：这是**系统级问题**，而非代码实现问题。

### 可能原因
1. **驱动/固件问题**：NPU 驱动与 CANN 8.3.RC1 版本不匹配
2. **权限问题**：内核执行需要特殊权限
3. **资源冲突**：设备 0 被其他进程占用（PID 1098717）
4. **编译工具链问题**：ccec 编译器配置不正确

## 建议解决步骤

###方案 1：联系系统管理员
```bash
# 检查驱动版本
npu-smi info

# 检查内核驱动日志
dmesg | tail -50

# 检查 NPU 进程
npu-smi info | grep "Process"
```

### 方案 2：使用 ACLNN 接口
尝试使用已编译的 ACLNN 接口（位于 `SwiGLUCustom/build_out`）：
```bash
bash run_aclnn.sh
```

### 方案 3：使用 PyTorch 接口
修复 torch_npu 依赖后测试：
```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
python3 test_swiglu_torch.py
```

##文件清单

- `swiglu_custom.cpp` - 内核实现
- `main.cpp` - host 端测试程序
- `CMakeLists.txt` - 构建配置
- `run.sh` - 一键运行脚本
- `scripts/gen_data.py` - 测试数据生成
- `scripts/verify_result.py` - 结果验证

## 系统信息

- CANN 版本：8.3.RC1
- SOC 版本：Ascend910B2  
- NPU 数量：8
- 编译器：ccec (Ascend C Compiler)

## 后续步骤

一旦系统问题解决，执行：
```bash
cd swiglu_op
bash run.sh
```

预期输出：
```
✅ SwiGLU TEST PASSED!
Max error: < 1e-5
```
