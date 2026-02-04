import numpy as np
import os

def gen_golden_data():
    total_rows = 8
    row_size = 256
    
    input_x = np.random.uniform(-5, 5, [total_rows, row_size]).astype(np.float32)
    
    # softmax = exp(x - max) / sum(exp(x - max))
    max_val = np.max(input_x, axis=-1, keepdims=True)
    exp_x = np.exp(input_x - max_val)
    sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
    golden_y = exp_x / sum_exp
    
    os.makedirs('../input', exist_ok=True)
    os.makedirs('../output', exist_ok=True)
    
    input_x.tofile('../input/input_x.bin')
    golden_y.tofile('../output/golden_y.bin')
    print(f'Generated input shape: {input_x.shape}')
    print(f'Generated golden output shape: {golden_y.shape}')

if __name__ == '__main__':
    gen_golden_data()
