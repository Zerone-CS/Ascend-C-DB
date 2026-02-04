import numpy as np
import sys

def verify():
    total_rows = 8
    row_size = 256
    
    output_y = np.fromfile('../output/output_y.bin', dtype=np.float32).reshape(total_rows, row_size)
    golden_y = np.fromfile('../output/golden_y.bin', dtype=np.float32).reshape(total_rows, row_size)
    
    rtol = 1e-3
    atol = 1e-5
    
    if np.allclose(output_y, golden_y, rtol=rtol, atol=atol):
        print('TEST PASSED!')
        print(f'Max absolute error: {np.max(np.abs(output_y - golden_y))}')
        print(f'Max relative error: {np.max(np.abs((output_y - golden_y) / (golden_y + 1e-10)))}')
        return 0
    else:
        print('TEST FAILED!')
        diff = np.abs(output_y - golden_y)
        print(f'Max absolute error: {np.max(diff)}')
        print(f'Error locations: {np.where(diff > atol)}')
        return 1

if __name__ == '__main__':
    sys.exit(verify())
