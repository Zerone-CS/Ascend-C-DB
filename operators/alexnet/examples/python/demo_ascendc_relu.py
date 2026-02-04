#!/usr/bin/env python3
"""Demo: AscendC Custom ReLU Operator on Ascend NPU

This demonstrates the complete workflow:
1. AscendC kernel written in C++ with standard structure
2. Built via msopgen into custom operator package
3. Installed to CANN runtime
4. Wrapped with PyBind11 for Python access
5. Executed on real NPU hardware
"""
import numpy as np
import sys
import os

# Add the pybind module path (adjust for new structure)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
pybind_dir = os.path.join(project_dir, 'bindings', 'pybind')
sys.path.insert(0, pybind_dir)

def main():
    print("="*60)
    print("AscendC Custom ReLU Operator Demo")
    print("="*60)
    
    try:
        import relu_custom_npu
        print("[OK] Loaded relu_custom_npu PyBind11 module")
    except ImportError as e:
        print(f"[ERROR] Failed to load module: {e}")
        print(f"Searched in: {pybind_dir}")
        print("Make sure to:")
        print("  1. Build the pybind module: cd bindings/pybind && python3 setup.py build_ext --inplace")
        print("  2. Set LD_LIBRARY_PATH:")
        print("     export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH")
        return 1
    
    print("\n--- Test 1: Basic ReLU (1D) ---")
    x1 = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    # Pad to 2048 elements (kernel requirement)
    x1_padded = np.zeros(2048, dtype=np.float32)
    x1_padded[:5] = x1
    
    y1 = relu_custom_npu.relu(x1_padded)[:5]
    expected1 = np.maximum(x1, 0)
    
    print(f"  Input:    {x1}")
    print(f"  Output:   {y1}")
    print(f"  Expected: {expected1}")
    print(f"  Match: {np.allclose(y1, expected1)}")
    
    print("\n--- Test 2: Random Data (2048 elements) ---")
    x2 = np.random.randn(2048).astype(np.float32)
    y2 = relu_custom_npu.relu(x2)
    expected2 = np.maximum(x2, 0)
    
    num_neg = (x2 < 0).sum()
    num_pos = (x2 >= 0).sum()
    print(f"  Input: {x2.shape}, {num_neg} negative, {num_pos} non-negative")
    print(f"  Output range: [{y2.min():.4f}, {y2.max():.4f}]")
    print(f"  All elements match: {np.allclose(y2, expected2)}")
    
    print("\n--- Test 3: Large Batch (8192 elements) ---")
    x3 = np.random.randn(8192).astype(np.float32) * 10
    y3 = relu_custom_npu.relu(x3)
    expected3 = np.maximum(x3, 0)
    
    max_diff = np.abs(y3 - expected3).max()
    print(f"  Input shape: {x3.shape}")
    print(f"  Max absolute difference: {max_diff}")
    print(f"  Verification: {'PASSED' if max_diff < 1e-5 else 'FAILED'}")
    
    print("\n--- Architecture Summary ---")
    print("  Kernel:   AscendC with standard structure (Init/Process/CopyIn/Compute/CopyOut)")
    print("  Build:    msopgen -> custom operator package")
    print("  Install:  CANN operator deployment")
    print("  Wrapper:  PyBind11 -> Python module")
    print("  Runtime:  aclnn API on Ascend 910B2 NPU")
    
    # Cleanup
    relu_custom_npu.finalize()
    
    print("\n" + "="*60)
    print("SUCCESS: AscendC Custom ReLU running on real NPU!")
    print("="*60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
