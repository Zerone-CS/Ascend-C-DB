#!/bin/bash
set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR

echo "========================================"
echo " SwiGLU 算子测试"
echo "========================================"

# 设置环境
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# 1. 生成测试数据
echo ""
echo ">>> Step 1: 生成测试数据"
cd scripts && python3 gen_data.py && cd ..

# 2. 编译
echo ""
echo ">>> Step 2: 编译"
mkdir -p build && cd build
cmake .. -DSOC_VERSION=Ascend910B2
make -j4
cd ..

# 3. 拷贝必要文件
cp build/swiglu_custom_kernel.o .
cp build/swiglu_npu .

# 4. 运行
echo ""
echo ">>> Step 3: 运行NPU测试"
./swiglu_npu

# 5. 验证结果
echo ""
echo ">>> Step 4: Python验证"
cd scripts && python3 verify_result.py && cd ..

echo ""
echo "========================================"
echo " 测试完成"
echo "========================================"
