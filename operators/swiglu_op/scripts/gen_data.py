#!/usr/bin/env python3
import numpy as np
import os

def gen_swiglu_data():
    # SwiGLU: y = x * swish(gate) = x * (gate * sigmoid(gate))
    np.random.seed(42)
    
    total_length = 8 * 256  # 8 rows, 256 cols
    
    # Generate input data
    x = np.random.randn(total_length).astype(np.float32)
    gate = np.random.randn(total_length).astype(np.float32)
    
    # Compute expected output: y = x * swish(gate)
    # swish(gate) = gate * sigmoid(gate)
    sigmoid_gate = 1.0 / (1.0 + np.exp(-gate))
    swish_gate = gate * sigmoid_gate
    golden = x * swish_gate
    
    # Save to files
    os.makedirs('../input', exist_ok=True)
    x.tofile('../input/input_x.bin')
    gate.tofile('../input/input_gate.bin')
    golden.tofile('../input/golden.bin')
    
    print(f"Generated test data:")
    print(f"  x shape: {x.shape}, range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  gate shape: {gate.shape}, range: [{gate.min():.4f}, {gate.max():.4f}]")
    print(f"  golden shape: {golden.shape}")
    print(f"  x[0:4]: {x[:4]}")
    print(f"  gate[0:4]: {gate[:4]}")
    print(f"  golden[0:4]: {golden[:4]}")

if __name__ == '__main__':
    gen_swiglu_data()
