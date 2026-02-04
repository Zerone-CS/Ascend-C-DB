#!/bin/bash
set -e

BASE=$(dirname $(realpath $0))
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

OPS=("ReluCustom" "AddCustom" "Conv2dCustom" "BnCustom" "MaxPoolCustom" "AvgPoolCustom" "FcCustom")

echo "=== 编译所有 AscendC 自定义算子 ==="

for op in "${OPS[@]}"; do
    echo ""
    echo ">>> 编译 $op ..."
    cd $BASE/$op
    rm -rf build_out
    bash build.sh 2>&1 | tail -20
    
    # 安装
    echo ">>> 安装 $op ..."
    cd build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run
    bash install.sh 2>&1 | tail -5
done

echo ""
echo "=== 所有算子编译安装完成 ==="
