#!/bin/bash
set -e

echo "🧹 开始清理项目..."
echo ""

# 1. 清理构建产物
echo "📦 清理构建产物..."
find . -type f \( -name "*.o" -o -name "*.so" -o -name "*.a" -o -name "*.pyc" \) -delete 2>/dev/null || true
find . -type d \( -name "build" -o -name "build_out" -o -name "__pycache__" \) -exec rm -rf {} + 2>/dev/null || true
echo "   ✓ 已删除编译文件和构建目录"

# 2. 清理性能分析数据
echo "📊 清理性能分析数据..."
find . -type d \( -name "prof_output*" -o -name "prof_instr*" -o -name "profiling*" \) -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "PROF_*" -exec rm -rf {} + 2>/dev/null || true
echo "   ✓ 已删除性能分析目录"

# 3. 清理 CMake 缓存
echo "🔧 清理 CMake 缓存..."
find . -name "CMakeCache.txt" -delete 2>/dev/null || true
find . -name "cmake_install.cmake" -delete 2>/dev/null || true
find . -name "install_manifest.txt" -delete 2>/dev/null || true
find . -type d -name "CMakeFiles" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "_CPack_Packages" -exec rm -rf {} + 2>/dev/null || true
echo "   ✓ 已删除 CMake 缓存文件"

# 4. 清理二进制测试数据
echo "🗂️  清理测试数据..."
find ./operators -name "*.bin" -delete 2>/dev/null || true
find ./operators -name "*_npu" -type f -delete 2>/dev/null || true
find ./operators -name "*_aclnn" -type f -delete 2>/dev/null || true
echo "   ✓ 已删除测试二进制文件"

# 5. 清理打包产物
echo "📦 清理打包产物..."
find . -name "*.run" -type f -delete 2>/dev/null || true
echo "   ✓ 已删除打包文件"

# 6. 统计清理结果
echo ""
echo "✅ 清理完成！"
echo ""
echo "📊 清理统计:"
echo "   当前项目大小: $(du -sh . | cut -f1)"
echo "   剩余 .o 文件: $(find . -name "*.o" 2>/dev/null | wc -l | tr -d ' ')"
echo "   剩余 .so 文件: $(find . -name "*.so" 2>/dev/null | wc -l | tr -d ' ')"
echo "   剩余 build 目录: $(find . -type d -name "build*" 2>/dev/null | wc -l | tr -d ' ')"
echo ""
echo "⚠️  请继续执行以下步骤:"
echo "   1. 修复脚本中的硬编码路径"
echo "   2. 增强 .gitignore 配置"
echo "   3. 初始化 Git 仓库"
