#!/usr/bin/env python3
import torch
import torch_npu
import numpy as np

def test_swiglu_npu():
    print("Testing SwiGLU on NPU via PyTorch")
    
    # Set device
    device = torch.device("npu:0")
    
    # Generate test data
    np.random.seed(42)
    total_length = 8 * 256
    
    x_np = np.random.randn(total_length).astype(np.float32)
    gate_np = np.random.randn(total_length).astype(np.float32)
    
    # Compute CPU reference
    sigmoid_gate = 1.0 / (1.0 + np.exp(-gate_np))
    swish_gate = gate_np * sigmoid_gate
    expected_np = x_np * swish_gate
    
    # Move to NPU
    x = torch.from_numpy(x_np).to(device)
    gate = torch.from_numpy(gate_np).to(device)
    
    # Compute SwiGLU on NPU: y = x * silu(gate)
    # SiLU (Swish) = x * sigmoid(x)
    swish_gate_npu = torch.nn.functional.silu(gate)
    y_npu = x * swish_gate_npu
    
    # Copy back to CPU
    y_cpu = y_npu.cpu().numpy()
    
    # Compare
    print(f"\nInput x[0:4]: {x_np[:4]}")
    print(f"Input gate[0:4]: {gate_np[:4]}")
    print(f"Expected[0:4]: {expected_np[:4]}")
    print(f"NPU Output[0:4]: {y_cpu[:4]}")
    
    max_err = np.max(np.abs(y_cpu - expected_np))
    mean_err = np.mean(np.abs(y_cpu - expected_np))
    
    print(f"\nMax error: {max_err:.2e}")
    print(f"Mean error: {mean_err:.2e}")
    
    if max_err < 1e-5:
        print("\n\033[32m✅ SwiGLU NPU TEST PASSED!\033[0m")
        return 0
    else:
        print("\n\033[31m❌ SwiGLU NPU TEST FAILED!\033[0m")
        return 1

if __name__ == "__main__":
    exit(test_swiglu_npu())
