# 贡献指南

感谢您对 Ascend-C-DB 的关注！欢迎提交贡献。

## 如何贡献

### 报告问题

- 使用 GitHub Issues 报告 bug 或提出功能建议
- 请提供：CANN 版本、NPU 型号、复现步骤、错误日志

### 提交代码

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m "feat: add your feature"`
4. 推送分支：`git push origin feature/your-feature`
5. 创建 Pull Request

### Commit 规范

使用 [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: 新功能
fix: 修复 bug
docs: 文档更新
examples: 示例更新
refactor: 代码重构
test: 测试相关
chore: 构建/工具变更
```

## 贡献类型

### 添加新算子示例

1. 在 `examples/` 创建目录，命名格式：`XX_算子名/`
2. 包含完整的 `*_custom.cpp` 文件
3. 更新 `examples/README.md` 索引
4. 如有新模板，添加到 `templates/`

### 完善文档

- 修正文档错误
- 补充 API 说明
- 添加使用案例

### 改进工具

- 扩展 `op_lookup.py` 支持的算子
- 优化构建脚本

## 代码规范

### C++ (Kernel)

- 使用 `__aicore__ inline` 修饰成员函数
- 类名格式：`KernelXxx`
- 入口函数：`extern "C" __global__ __aicore__ void xxx_custom(...)`
- `BUFFER_NUM = 2`（Double Buffer）

### Python (工具)

- 使用 Python 3.8+
- 添加类型提示
- 使用 docstring 说明函数用途

## 测试

提交前请确保：

```bash
# 语法检查 (如有 NPU 环境)
bash scripts/build.sh examples

# 工具测试
python tools/op_lookup.py softmax
```

## 问题讨论

- 大型改动请先开 Issue 讨论
- 欢迎在 Discussions 区交流 AscendC 开发经验

感谢您的贡献！🎉
