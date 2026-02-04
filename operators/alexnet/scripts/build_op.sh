#!/bin/bash
# Build and install custom operators
set -e

source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
OP_PROJECTS="$PROJECT_DIR/op_projects"

OP_NAME=${1:-relu}

if [ ! -d "$OP_PROJECTS/$OP_NAME" ]; then
    echo "Error: Operator '$OP_NAME' not found in $OP_PROJECTS"
    echo "Available operators:"
    ls -1 "$OP_PROJECTS"
    exit 1
fi

echo "=== Building operator: $OP_NAME ==="
cd "$OP_PROJECTS/$OP_NAME"
bash build.sh

echo ""
echo "=== Installing operator ==="
cd build_out/_CPack_Packages/Linux/External/custom_opp_*.run
bash install.sh

echo ""
echo "Done! Operator $OP_NAME installed successfully."
echo "Set LD_LIBRARY_PATH:"
echo "  export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:\$LD_LIBRARY_PATH"
