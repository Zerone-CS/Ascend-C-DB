#!/bin/bash
set -e

# Setup CANN environment
export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash

echo "=== Building ResNet-18 for Ascend NPU ==="
echo "ASCEND_PATH: ${ASCEND_INSTALL_PATH}"

# Create build directory
mkdir -p build
cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

echo ""
echo "Build complete! Run with:"
echo "  ./build/resnet18_npu [device_id]"
echo "  Default device_id is 2"
