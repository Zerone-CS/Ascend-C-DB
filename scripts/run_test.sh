#!/bin/bash
# 运行测试脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/setup_env.sh" 2>/dev/null || true

usage() {
    echo "Usage: $0 <op_name> [options]"
    echo ""
    echo "Options:"
    echo "  --device <id>   NPU device ID (default: 0)"
    echo "  --shape <shape> Test shape (default: operator specific)"
    echo ""
    echo "Examples:"
    echo "  $0 softmax_op"
    echo "  $0 add_op --device 1"
}

DEVICE_ID=0
TARGET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE_ID="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            TARGET="$1"
            shift
            ;;
    esac
done

if [ -z "$TARGET" ]; then
    usage
    exit 1
fi

# 查找算子目录
OP_DIR=""
if [ -d "$PROJECT_ROOT/operators/$TARGET" ]; then
    OP_DIR="$PROJECT_ROOT/operators/$TARGET"
elif [ -d "$PROJECT_ROOT/$TARGET" ]; then
    OP_DIR="$PROJECT_ROOT/$TARGET"
else
    echo "❌ Operator not found: $TARGET"
    exit 1
fi

echo "🧪 Running test: $TARGET"
echo "   Device: $DEVICE_ID"
echo "   Directory: $OP_DIR"

cd "$OP_DIR"

# 查找并运行测试
if [ -f "run.sh" ]; then
    echo "\n➡️  Executing run.sh..."
    bash run.sh
elif [ -f "test_*.py" ]; then
    for test_file in test_*.py; do
        echo "\n➡️  Running $test_file..."
        python3 "$test_file"
    done
elif [ -f "build/${TARGET%_op}_npu" ]; then
    echo "\n➡️  Running binary..."
    ./build/${TARGET%_op}_npu
else
    echo "⚠️  No test script found"
    echo "   Expected: run.sh, test_*.py, or build/*_npu"
fi
