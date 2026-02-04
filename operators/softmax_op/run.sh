#!/bin/bash
set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR

# Source environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

RUN_MODE=${1:-npu}
echo "Run mode: $RUN_MODE"

# Generate input data
echo "=== Generating test data ==="
cd scripts && python3 gen_data.py && cd ..

# Build
echo "=== Building ==="
rm -rf build && mkdir build && cd build
cmake .. -DRUN_MODE=$RUN_MODE -DSOC_VERSION=Ascend910B2
make -j4

# Run
echo "=== Running ==="
if [ "$RUN_MODE" == "cpu" ]; then
    ./softmax_cpu
else
    ./softmax_npu
fi

# Verify
echo "=== Verifying ==="
cd ../scripts && python3 verify_result.py
