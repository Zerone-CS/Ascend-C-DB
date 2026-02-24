# Contributing / 贡献指南

感谢你对 AscendMind 的关注！欢迎任何形式的贡献。

## 如何贡献

### 报告 Bug

请通过 [GitHub Issues](../../issues) 提交，并包含：
- 环境信息 (Python 版本、操作系统)
- 复现步骤
- 预期行为 vs 实际行为

### 功能建议

请通过 Issue 提交 Feature Request，说明场景和预期效果。

### 提交代码

1. Fork 本仓库
2. 创建特性分支: `git checkout -b feat/your-feature`
3. 提交更改: `git commit -m 'feat: add your feature'`
4. 推送分支: `git push origin feat/your-feature`
5. 创建 Pull Request

### Commit 规范

使用 [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` 新功能
- `fix:` Bug 修复
- `docs:` 文档更新
- `refactor:` 重构
- `test:` 测试
- `chore:` 构建/工具链

### 代码规范

- 使用 [Ruff](https://github.com/astral-sh/ruff) 进行代码检查: `ruff check .`
- Python >= 3.9
- 新增函数请添加 docstring 和类型注解
- 测试放在 `tests/` 目录

## 开发设置

```bash
git clone https://github.com/your-org/ascend-mind.git
cd ascend-mind
pip install -e ".[dev,agent]"
python -m pytest tests/
```
