import numpy as np
import os

def generate_test_data():
    total_length = 8 * 256
    
    x = np.random.randn(total_length).astype(np.float32)
    y = np.random.randn(total_length).astype(np.float32)
    
    golden = x + y
    
    os.makedirs('input', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    x.tofile('input/input_x.bin')
    y.tofile('input/input_y.bin')
    golden.tofile('output/golden.bin')
    
    print(f"Generated test data with shape: ({total_length},)")
    print(f"x[0:4]: {x[:4]}")
    print(f"y[0:4]: {y[:4]}")
    print(f"golden[0:4]: {golden[:4]}")

def verify_result():
    golden = np.fromfile('output/golden.bin', dtype=np.float32)
    result = np.fromfile('output/output_z.bin', dtype=np.float32)
    
    if np.allclose(golden, result, rtol=1e-5, atol=1e-5):
        print("Verification PASSED!")
    else:
        print("Verification FAILED!")
        diff = np.abs(golden - result)
        print(f"Max diff: {np.max(diff)}")
        print(f"Mean diff: {np.mean(diff)}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'verify':
        verify_result()
    else:
        generate_test_data()
