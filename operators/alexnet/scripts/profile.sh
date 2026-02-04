#!/bin/bash
# Profile operator performance with msprof
set -e

source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
PROF_DIR="$PROJECT_DIR/profiling/prof_$(date +%Y%m%d_%H%M%S)"

APPLICATION=${1:-"python3 $PROJECT_DIR/examples/python/demo_ascendc_relu.py"}

echo "=== Profiling: $APPLICATION ==="
echo "Output: $PROF_DIR"

mkdir -p "$PROF_DIR"

# Collect profiling data
msprof --output="$PROF_DIR" \
       --task-time=on \
       --ai-core=on \
       --aic-metrics=PipeUtilization \
       --application="$APPLICATION"

echo ""
echo "=== Exporting reports ==="
PROF_SUBDIR=$(ls -d "$PROF_DIR"/PROF_* 2>/dev/null | head -1)
if [ -n "$PROF_SUBDIR" ]; then
    msprof --export=on --output="$PROF_SUBDIR"
    echo ""
    echo "Reports exported to: $PROF_SUBDIR/mindstudio_profiler_output/"
    
    # Show summary if available
    SUMMARY="$PROF_SUBDIR/mindstudio_profiler_output/op_summary_*.csv"
    if ls $SUMMARY 1>/dev/null 2>&1; then
        echo ""
        echo "=== Op Summary ==="
        head -5 $SUMMARY | column -t -s,
    fi
fi
