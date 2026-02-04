#!/bin/bash
# Ascend C 环境配置脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ASCEND_C_DB_ROOT="$(dirname "$SCRIPT_DIR")"

echo "====================================================="
echo "Ascend C Development Environment Setup"
echo "====================================================="

# 1. CANN 环境
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "✅ CANN environment loaded"
else
    echo "⚠️  CANN toolkit not found"
fi

# 2. 检查编译器
if command -v ccec &> /dev/null; then
    echo "✅ ccec compiler: $(which ccec)"
else
    echo "❌ ccec compiler not found"
fi

# 3. 检查NPU
if command -v npu-smi &> /dev/null; then
    NPU_COUNT=$(npu-smi info -l 2>/dev/null | grep "Total Count" | awk '{print $4}')
    echo "✅ NPU devices: ${NPU_COUNT:-unknown}"
else
    echo "⚠️  npu-smi not found"
fi

# 4. Conda 环境
CONDA_FOUND=false

# 尝试多种方式查找 conda
if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
    # 如果已经在 conda 环境中
    source "$CONDA_PREFIX/etc/profile.d/conda.sh"
    CONDA_FOUND=true
elif command -v conda &> /dev/null; then
    # 如果 conda 在 PATH 中
    eval "$(conda shell.bash hook)"
    CONDA_FOUND=true
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # 用户目录下的 anaconda
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # 用户目录下的 miniconda
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=true
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    # 系统级 conda
    source /opt/conda/etc/profile.d/conda.sh
    CONDA_FOUND=true
fi

if [ "$CONDA_FOUND" = true ]; then
    if conda env list | grep -q "torch_npu"; then
        echo "✅ Conda torch_npu environment available"
        echo ""
        echo "💡 To activate PyTorch NPU environment, run:"
        echo "   conda activate torch_npu"
    else
        echo "⚠️  torch_npu conda environment not found"
    fi
else
    echo "⚠️  Conda not found (optional)"
fi

# 5. 导出路径
export PATH="$ASCEND_C_DB_ROOT/tools:$PATH"

echo ""
echo "====================================================="
echo "Environment setup complete!"
echo "  ASCEND_C_DB_ROOT=$ASCEND_C_DB_ROOT"
echo "====================================================="
