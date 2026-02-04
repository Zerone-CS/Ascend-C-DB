#!/usr/bin/env python3
"""Generate Chrome Tracing compatible timeline from profiling data"""

import pandas as pd
import json
import os
import sys

def generate_timeline(prof_dir, output_file):
    pipe_csv = os.path.join(prof_dir, 'PipeUtilization.csv')
    basic_csv = os.path.join(prof_dir, 'OpBasicInfo.csv')
    
    events = []
    
    # Process metadata
    events.append({"name": "process_name", "ph": "M", "pid": 0, "args": {"name": "ReluCustom Kernel"}})
    
    if os.path.exists(pipe_csv):
        df = pd.read_csv(pipe_csv)
        
        for _, row in df.iterrows():
            core_id = int(row['block_id'])
            total_time = row['aiv_time(us)']
            
            # Thread name for each core
            events.append({"name": "thread_name", "ph": "M", "pid": 0, "tid": core_id, 
                          "args": {"name": f"AI Core {core_id}"}})
            
            # Calculate time for each pipeline stage
            vec_time = row['aiv_vec_ratio'] * total_time
            scalar_time = row['aiv_scalar_ratio'] * total_time  
            mte2_time = row['aiv_mte2_ratio'] * total_time
            mte3_time = row['aiv_mte3_ratio'] * total_time
            
            base_ts = core_id * 0.1  # Slight offset per core for visualization
            
            # MTE2 (CopyIn) - starts first
            events.append({
                "name": "MTE2 (CopyIn)",
                "ph": "X",
                "pid": 0,
                "tid": core_id,
                "ts": base_ts,
                "dur": mte2_time,
                "args": {"ratio": f"{row['aiv_mte2_ratio']*100:.1f}%"}
            })
            
            # Vector Compute
            events.append({
                "name": "Vector Compute",
                "ph": "X",
                "pid": 0,
                "tid": core_id,
                "ts": base_ts + mte2_time * 0.5,  # Overlapped
                "dur": vec_time,
                "args": {"ratio": f"{row['aiv_vec_ratio']*100:.1f}%"}
            })
            
            # Scalar Control (runs throughout)
            events.append({
                "name": "Scalar Control",
                "ph": "X",
                "pid": 0,
                "tid": core_id,
                "ts": base_ts,
                "dur": scalar_time,
                "args": {"ratio": f"{row['aiv_scalar_ratio']*100:.1f}%"}
            })
            
            # MTE3 (CopyOut)
            events.append({
                "name": "MTE3 (CopyOut)",
                "ph": "X",
                "pid": 0,
                "tid": core_id,
                "ts": base_ts + mte2_time + vec_time * 0.5,
                "dur": mte3_time,
                "args": {"ratio": f"{row['aiv_mte3_ratio']*100:.1f}%"}
            })
    
    with open(output_file, 'w') as f:
        json.dump(events, f, indent=2)
    
    print(f"Timeline saved to: {output_file}")
    print(f"Open in Chrome: chrome://tracing and load the JSON file")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prof_dir = sys.argv[1]
    else:
        prof_dir = "/root/zhh_workspace/Ascend-C-DB/operators/alexnet/profiling_source/OPPROF_20260203112630_QJGSQEVRBBTMVFUI"
    
    output = os.path.join(os.path.dirname(prof_dir), "pipeline_timeline.json")
    generate_timeline(prof_dir, output)
