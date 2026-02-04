#!/bin/bash
# 一键运行 NPU 算子测试

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "====================================================="
echo "NPU Operator Test Runner"
echo "====================================================="

# 加载环境
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    echo "❌ CANN not found"
    exit 1
fi

# 加载 Conda 环境
CONDA_FOUND=false

if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PREFIX/etc/profile.d/conda.sh"
    CONDA_FOUND=true
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    CONDA_FOUND=true
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=true
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
    CONDA_FOUND=true
fi

if [ "$CONDA_FOUND" = false ]; then
    echo "❌ Conda not found"
    exit 1
fi

conda activate torch_npu

echo "✅ Environment loaded"
echo ""

# 运行测试
cd "$PROJECT_ROOT"
python3 scripts/test_all_ops.py
