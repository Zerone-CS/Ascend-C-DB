#!/usr/bin/env python3
"""DivCustom 算子指令流水图生成器"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sqlite3
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 从profiling数据库获取数据
prof_dir = './prof_output/PROF_000001_20260203200107897_LJIKLAAPQKEHIKOB'
db_path = f'{prof_dir}/device_2/sqlite/ai_core_op_summary.db'

# 查询数据
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 获取算子信息
cursor.execute('SELECT op_name, op_type, block_dim, task_type FROM ge_summary')
op_info = cursor.fetchone()

# 获取性能指标 (从op_summary csv)
import csv
csv_path = f'{prof_dir}/mindstudio_profiler_output/'
csv_files = [f for f in os.listdir(csv_path) if f.startswith('op_summary')]

metrics = {}
if csv_files:
    with open(os.path.join(csv_path, csv_files[0]), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'DivCustom' in row.get('Op Name', ''):
                metrics = row
                break

conn.close()

# 解析流水线数据
aiv_vec_ratio = float(metrics.get('aiv_vec_ratio', 0))
aiv_scalar_ratio = float(metrics.get('aiv_scalar_ratio', 0))
aiv_mte2_ratio = float(metrics.get('aiv_mte2_ratio', 0))
aiv_mte3_ratio = float(metrics.get('aiv_mte3_ratio', 0))
task_duration = float(metrics.get('Task Duration(us)', 0))
aiv_time = float(metrics.get('aiv_time(us)', 0))
aiv_cycles = int(metrics.get('aiv_total_cycles', 0))

print("="*70)
print("DivCustom 算子性能分析报告")
print("="*70)
print(f"\n【算子基本信息】")
print(f"  算子名称: {op_info[0] if op_info else 'DivCustom'}")
print(f"  算子类型: {op_info[1] if op_info else 'DivCustom'}")
print(f"  执行单元: {op_info[3] if op_info else 'AI_VECTOR_CORE'}")
print(f"  Block数量: {op_info[2] if op_info else 8}")
print(f"  执行时间: {task_duration:.2f} us")
print(f"  AIV时间: {aiv_time:.2f} us")
print(f"  总Cycles: {aiv_cycles}")

print(f"\n【流水线利用率】")
print(f"  Vector单元:  {aiv_vec_ratio*100:.2f}%")
print(f"  Scalar单元:  {aiv_scalar_ratio*100:.2f}%")
print(f"  MTE2(搬入):  {aiv_mte2_ratio*100:.2f}%")
print(f"  MTE3(搬出):  {aiv_mte3_ratio*100:.2f}%")

# 创建图形
fig = plt.figure(figsize=(16, 10))

# 1. 流水线利用率饼图
ax1 = fig.add_subplot(2, 2, 1)
labels = ['Vector', 'Scalar', 'MTE2', 'MTE3', 'Idle']
ratios = [aiv_vec_ratio, aiv_scalar_ratio, aiv_mte2_ratio, aiv_mte3_ratio]
idle = max(0, 1 - sum(ratios))
sizes = ratios + [idle]
colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#bdc3c7']
explode = (0.05, 0.05, 0.05, 0.05, 0)

wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                   autopct='%1.1f%%', startangle=90, pctdistance=0.75)
ax1.set_title('Pipeline Utilization Breakdown', fontsize=14, fontweight='bold')

# 2. 流水线利用率柱状图
ax2 = fig.add_subplot(2, 2, 2)
units = ['Vector', 'Scalar', 'MTE2\n(CopyIn)', 'MTE3\n(CopyOut)']
values = [r * 100 for r in ratios]
bar_colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
bars = ax2.bar(units, values, color=bar_colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Utilization (%)', fontsize=12)
ax2.set_title('Pipeline Unit Utilization', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(axis='y', alpha=0.3)

# 3. 指令流水时序图 (模拟)
ax3 = fig.add_subplot(2, 1, 2)

# 模拟DivCustom的流水执行
# 假设有1个tile的执行流程: CopyIn -> Compute -> CopyOut
# 由于是double buffer, 流水线可以重叠

tile_count = 1  # 每个block处理1个tile (256元素 / 8核 = 32元素, 但tileNum=1)
block_count = 8

# 颜色定义
colors_stages = {
    'MTE2': '#3498db',   # 蓝色 - 搬入
    'Vec': '#2ecc71',    # 绿色 - 计算
    'MTE3': '#9b59b6',   # 紫色 - 搬出
    'Scalar': '#e74c3c'  # 红色 - 标量控制
}

# 根据实际比例计算时间片
total_cycles = 100  # 归一化到100个周期用于展示
scalar_cycles = int(aiv_scalar_ratio * total_cycles)
vec_cycles = int(aiv_vec_ratio * total_cycles)
mte2_cycles = int(aiv_mte2_ratio * total_cycles)
mte3_cycles = int(aiv_mte3_ratio * total_cycles)

# 为8个block绘制时序
for block in range(block_count):
    y_pos = block
    # 计算每个阶段的起始位置
    # 假设执行顺序: Scalar setup -> MTE2 -> Div计算 -> MTE3
    
    # Block错开启动以模拟并行执行
    offset = block * 2
    
    # Scalar (循环控制、地址计算)
    ax3.barh(y_pos, scalar_cycles * 0.3, left=offset, height=0.6, 
             color=colors_stages['Scalar'], alpha=0.8, label='Scalar' if block==0 else '')
    
    # MTE2 (CopyIn x, y)
    ax3.barh(y_pos, mte2_cycles, left=offset + scalar_cycles * 0.3, height=0.6,
             color=colors_stages['MTE2'], alpha=0.8, label='MTE2 (CopyIn)' if block==0 else '')
    
    # Vector (Div计算)
    ax3.barh(y_pos, vec_cycles * 2, left=offset + scalar_cycles * 0.3 + mte2_cycles, height=0.6,
             color=colors_stages['Vec'], alpha=0.8, label='Vector (Div)' if block==0 else '')
    
    # MTE3 (CopyOut z)
    ax3.barh(y_pos, mte3_cycles, left=offset + scalar_cycles * 0.3 + mte2_cycles + vec_cycles * 2, height=0.6,
             color=colors_stages['MTE3'], alpha=0.8, label='MTE3 (CopyOut)' if block==0 else '')
    
    # Scalar (循环结束检查)
    ax3.barh(y_pos, scalar_cycles * 0.2, left=offset + scalar_cycles * 0.3 + mte2_cycles + vec_cycles * 2 + mte3_cycles, 
             height=0.6, color=colors_stages['Scalar'], alpha=0.8)

# 设置Y轴
ax3.set_yticks(range(block_count))
ax3.set_yticklabels([f'Block {i}' for i in range(block_count)])
ax3.set_xlabel('Cycles (normalized)', fontsize=12)
ax3.set_title('Instruction Pipeline Timeline (8 AI Vector Cores)', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', ncol=4)
ax3.grid(axis='x', alpha=0.3)
ax3.set_xlim(0, 100)

# 添加注释
ax3.annotate('CopyIn\n(x,y)', xy=(25, 4), fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax3.annotate('Div\nCompute', xy=(45, 4), fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax3.annotate('CopyOut\n(z)', xy=(60, 4), fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))

plt.tight_layout()
plt.savefig('div_pipeline_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n[图表已保存到 div_pipeline_analysis.png]")

# 输出性能分析结论
print(f"\n【性能瓶颈分析】")
bottleneck = max([(aiv_scalar_ratio, 'Scalar'), (aiv_vec_ratio, 'Vector'), 
                  (aiv_mte2_ratio, 'MTE2'), (aiv_mte3_ratio, 'MTE3')])
print(f"  主要瓶颈: {bottleneck[1]} 单元 ({bottleneck[0]*100:.1f}%)")

if aiv_scalar_ratio > 0.4:
    print(f"  - Scalar占比较高({aiv_scalar_ratio*100:.1f}%), 循环控制开销较大")
    print(f"  - 建议: 增大TILE_SIZE减少循环迭代次数")
if aiv_vec_ratio < 0.1:
    print(f"  - Vector利用率低({aiv_vec_ratio*100:.1f}%), 计算密度不足")
    print(f"  - 建议: 增加每次计算的数据量")
if aiv_mte2_ratio + aiv_mte3_ratio > 0.5:
    print(f"  - 数据搬运占比高({(aiv_mte2_ratio+aiv_mte3_ratio)*100:.1f}%), 为内存受限算子")
    print(f"  - 建议: 优化数据复用或使用更大的分块")

print(f"\n" + "="*70)
