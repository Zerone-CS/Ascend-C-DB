#!/bin/bash
# AlexNet on NPU - True NPU Execution
set -e

source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
BUILD_DIR="$PROJECT_DIR/build"

# Build if needed
if [ ! -f "$BUILD_DIR/alexnet_full" ]; then
    echo "Building..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" && cmake "$PROJECT_DIR" && make -j4
fi

echo ""
echo "Select which to run:"
echo "  1) test_relu_aclnn   - ReLU via aclnn API"
echo "  2) alexnet_full      - Complete AlexNet (Conv+Pool+FC)"
echo "  3) demo_relu_python  - Python ReLU demo"
echo ""

if [ -n "$1" ]; then
    choice=$1
else
    read -p "Enter choice [1-3, default=2]: " choice
    choice=${choice:-2}
fi

case $choice in
    1) "$BUILD_DIR/test_relu_aclnn" ;;
    2) "$BUILD_DIR/alexnet_full" ;;
    3) cd "$PROJECT_DIR" && python3 examples/python/demo_ascendc_relu.py ;;
    *) echo "Invalid choice"; exit 1 ;;
esac
