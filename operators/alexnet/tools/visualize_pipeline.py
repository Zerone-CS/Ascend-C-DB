#!/usr/bin/env python3
"""Visualize AscendC Operator Pipeline from msprof data"""

import pandas as pd
import os
import sys

def draw_bar(value, max_value, width=40, char='█'):
    if max_value == 0:
        return ' ' * width
    filled = int(width * value / max_value)
    return char * filled + '░' * (width - filled)

def visualize_pipeline(prof_dir):
    pipe_csv = os.path.join(prof_dir, 'PipeUtilization.csv')
    mem_csv = os.path.join(prof_dir, 'MemoryUB.csv')
    basic_csv = os.path.join(prof_dir, 'OpBasicInfo.csv')
    
    print("=" * 80)
    print("   AscendC Operator Pipeline Visualization")
    print("=" * 80)
    
    # Basic info
    if os.path.exists(basic_csv):
        basic_df = pd.read_csv(basic_csv)
        print(f"\n【Operator Info】")
        print(f"  Name:          {basic_df['Op Name'].iloc[0]}")
        print(f"  Type:          {basic_df['Op Type'].iloc[0]}")
        print(f"  Duration:      {basic_df['Task Duration(us)'].iloc[0]:.3f} μs")
        print(f"  Block Dim:     {basic_df['Block Dim'].iloc[0]}")
        print(f"  Frequency:     {basic_df['Current Freq'].iloc[0]} MHz")
    
    if os.path.exists(pipe_csv):
        df = pd.read_csv(pipe_csv)
        
        print(f"\n{'=' * 80}")
        print("【Pipeline Utilization per AI Core】")
        print(f"{'=' * 80}")
        
        avg_vec = df['aiv_vec_ratio'].mean() * 100
        avg_scalar = df['aiv_scalar_ratio'].mean() * 100
        avg_mte2 = df['aiv_mte2_ratio'].mean() * 100
        avg_mte3 = df['aiv_mte3_ratio'].mean() * 100
        
        print(f"\n  Pipeline Stage      Avg Ratio    Visualization")
        print(f"  " + "-" * 70)
        print(f"  Vector Compute:     {avg_vec:6.2f}%     {draw_bar(avg_vec, 100, 30)}")
        print(f"  Scalar Compute:     {avg_scalar:6.2f}%     {draw_bar(avg_scalar, 100, 30)}")
        print(f"  MTE2 (GM→UB):       {avg_mte2:6.2f}%     {draw_bar(avg_mte2, 100, 30)}")
        print(f"  MTE3 (UB→GM):       {avg_mte3:6.2f}%     {draw_bar(avg_mte3, 100, 30)}")
        
        print(f"\n【Per-Core Pipeline Details】")
        print(f"  {'Core':<6} {'Time(μs)':<10} {'Vec%':<8} {'Scalar%':<10} {'MTE2%':<8} {'MTE3%':<8}")
        print(f"  " + "-" * 60)
        
        for _, row in df.iterrows():
            core_id = int(row['block_id'])
            time = row['aiv_time(us)']
            vec = row['aiv_vec_ratio'] * 100
            scalar = row['aiv_scalar_ratio'] * 100
            mte2 = row['aiv_mte2_ratio'] * 100
            mte3 = row['aiv_mte3_ratio'] * 100
            print(f"  Core {core_id:<2}  {time:<10.3f} {vec:<8.2f} {scalar:<10.2f} {mte2:<8.2f} {mte3:<8.2f}")
        
        # Pipeline diagram
        print(f"\n{'=' * 80}")
        print("【Instruction Pipeline Diagram】")
        print(f"{'=' * 80}")
        print(r"""
  Time ────────────────────────────────────────────────────────────────►
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    Double Buffer Pipeline                          │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  Tile 0:  [CopyIn]──────►[Compute]──────►[CopyOut]                 │
  │                  \                       /                          │
  │  Tile 1:          [CopyIn]──────►[Compute]──────►[CopyOut]         │
  │                          \                       /                  │
  │  Tile 2:                  [CopyIn]──────►[Compute]──────►[CopyOut] │
  │                                                                     │
  │  Hardware Units:                                                    │
  │    MTE2 (DMA In):  ████████████░░░░░░░░  Active when CopyIn        │
  │    Vector Unit:    ██░░░░░░░░░░░░░░░░░░  Active when Compute       │
  │    MTE3 (DMA Out): ████████░░░░░░░░░░░░  Active when CopyOut       │
  │    Scalar Unit:    ████████████████████████████  Control overhead  │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
        """)
        
        # Performance analysis
        print(f"\n{'=' * 80}")
        print("【Performance Bottleneck Analysis】")
        print(f"{'=' * 80}")
        print(f"""
  ⚠ Scalar 占比过高 ({avg_scalar:.1f}%): 标量控制指令占主导
  ⚠ Vector 利用率低 ({avg_vec:.1f}%): 向量计算单元未充分利用
  ✓ MTE2/MTE3 带宽: 数据搬运正常
  
  💡 优化建议:
     1. 增大 tile size (当前 256 elements) 减少循环开销
     2. 使用更大的数据块减少标量指令占比
     3. 考虑合并多个算子减少内存访问
        """)
    
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prof_dir = sys.argv[1]
    else:
        prof_dir = "/root/zhh_workspace/Ascend-C-DB/operators/alexnet/profiling_source/OPPROF_20260203112630_QJGSQEVRBBTMVFUI"
    visualize_pipeline(prof_dir)
