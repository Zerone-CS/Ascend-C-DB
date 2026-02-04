#!/usr/bin/env python3
import numpy as np
import sys

def verify():
    golden = np.fromfile('../input/golden.bin', dtype=np.float32)
    output = np.fromfile('../output/output_y.bin', dtype=np.float32)
    
    print(f"Golden shape: {golden.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Golden[0:4]: {golden[:4]}")
    print(f"Output[0:4]: {output[:4]}")
    
    diff = np.abs(golden - output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\nMax absolute error: {max_diff:.2e}")
    print(f"Mean absolute error: {mean_diff:.2e}")
    
    if max_diff < 1e-5:
        print("\n✅ TEST PASSED!")
        return 0
    else:
        print("\n❌ TEST FAILED!")
        # Find first mismatch
        mismatch_idx = np.argmax(diff)
        print(f"First large diff at index {mismatch_idx}:")
        print(f"  Golden: {golden[mismatch_idx]:.8f}")
        print(f"  Output: {output[mismatch_idx]:.8f}")
        return 1

if __name__ == '__main__':
    sys.exit(verify())
