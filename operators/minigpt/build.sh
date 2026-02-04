#!/bin/bash
# MiniGPT AscendC 编译脚本

set -e

echo "========================================"
echo "MiniGPT AscendC Build"
echo "========================================"

# 环境设置
export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "编译完成! 运行测试:"
echo "  cd ${SCRIPT_DIR}/tests && python3 test_minigpt.py"
