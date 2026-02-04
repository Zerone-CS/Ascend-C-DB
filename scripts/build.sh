#!/bin/bash
# 统一构建脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 加载环境
source "$SCRIPT_DIR/setup_env.sh" 2>/dev/null || true

usage() {
    echo "Usage: $0 <target> [options]"
    echo ""
    echo "Targets:"
    echo "  <op_name>     Build specific operator (e.g., softmax_op, add_op)"
    echo "  examples      Build all examples"
    echo "  clean         Clean build artifacts"
    echo ""
    echo "Options:"
    echo "  --soc <soc>   SOC version (default: Ascend910B2)"
    echo "  --debug       Enable debug mode"
    echo ""
    echo "Examples:"
    echo "  $0 softmax_op"
    echo "  $0 examples"
    echo "  $0 add_op --soc Ascend310P3"
}

# 默认参数
SOC_VERSION="Ascend910B2"
DEBUG_MODE=0
TARGET=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --soc)
            SOC_VERSION="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=1
            shift
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

# 确定架构
case $SOC_VERSION in
    Ascend910*)
        AICORE_ARCH="dav-c220"
        ;;
    Ascend310P*)
        AICORE_ARCH="dav-c100"
        ;;
    *)
        AICORE_ARCH="dav-c220"
        ;;
esac

# 编译单个文件
build_kernel() {
    local src_file="$1"
    local out_file="${src_file%.cpp}.o"
    
    echo "➡️  Compiling: $src_file"
    
    local flags="-O2"
    if [ $DEBUG_MODE -eq 1 ]; then
        flags="-O0 -g"
    fi
    
    ccec --cce-aicore-arch=$AICORE_ARCH $flags -c "$src_file" -o "$out_file"
    echo "✅ Output: $out_file"
}

# 构建算子目录
build_operator() {
    local op_dir="$1"
    
    if [ ! -d "$op_dir" ]; then
        echo "❌ Directory not found: $op_dir"
        exit 1
    fi
    
    echo "\n🔧 Building: $(basename $op_dir)"
    echo "   SOC: $SOC_VERSION"
    echo "   Arch: $AICORE_ARCH"
    
    cd "$op_dir"
    
    # 查找所有kernel文件
    for cpp_file in *_custom.cpp *_kernel.cpp; do
        if [ -f "$cpp_file" ]; then
            build_kernel "$cpp_file"
        fi
    done
    
    # 如果有CMakeLists.txt，也可以使用cmake构建
    if [ -f "CMakeLists.txt" ]; then
        echo "\n➡️  Running CMake build..."
        mkdir -p build && cd build
        cmake .. -DSOC_VERSION=$SOC_VERSION
        make -j$(nproc)
    fi
    
    echo "\n✅ Build complete!"
}

# 主逻辑
case $TARGET in
    examples)
        echo "📦 Building all examples..."
        for dir in "$PROJECT_ROOT/examples"/*/; do
            if [ -f "$dir"/*.cpp ]; then
                build_operator "$dir"
            fi
        done
        ;;
    clean)
        echo "🧹 Cleaning build artifacts..."
        find "$PROJECT_ROOT" -name "*.o" -delete
        find "$PROJECT_ROOT" -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
        echo "✅ Clean complete"
        ;;
    *)
        # 查找算子目录
        if [ -d "$PROJECT_ROOT/operators/$TARGET" ]; then
            build_operator "$PROJECT_ROOT/operators/$TARGET"
        elif [ -d "$PROJECT_ROOT/$TARGET" ]; then
            build_operator "$PROJECT_ROOT/$TARGET"
        else
            echo "❌ Target not found: $TARGET"
            echo "   Searched: $PROJECT_ROOT/operators/$TARGET"
            echo "             $PROJECT_ROOT/$TARGET"
            exit 1
        fi
        ;;
esac
