#!/bin/bash
set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR

echo "========================================"
echo " SwiGLU 算子测试 (ACLNN 接口)"
echo "========================================"

# 设置环境
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
VENDOR_PATH=$ASCEND_HOME/opp/vendors/customize

# 编译主程序
echo ""
echo ">>> Step 1: 编译测试程序"
g++ -std=c++17 -O2 \
    -I$ASCEND_HOME/include \
    -I$VENDOR_PATH/op_api/include \
    -I$SCRIPT_DIR \
    -L$ASCEND_HOME/lib64 \
    -L$VENDOR_PATH/op_api/lib \
    main_aclnn.cpp \
    -o swiglu_aclnn \
    -lascendcl -lnnopbase -lcust_opapi

echo "编译成功!"

# 设置运行时库路径
export LD_LIBRARY_PATH=$VENDOR_PATH/op_api/lib:$ASCEND_HOME/lib64:$LD_LIBRARY_PATH

# 运行
echo ""
echo ">>> Step 2: 运行NPU测试"
./swiglu_aclnn

echo ""
echo "========================================"
echo " 测试完成"
echo "========================================"
