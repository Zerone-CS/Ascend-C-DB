import torch
import torch_npu
import numpy as np

def test_softmax_npu():
    torch_npu.npu.set_device(0)
    
    total_rows = 8
    row_size = 256
    
    # Create random input
    input_x = np.random.uniform(-5, 5, [total_rows, row_size]).astype(np.float32)
    
    # Convert to NPU tensor
    x_npu = torch.from_numpy(input_x).npu()
    
    # Run softmax on NPU
    y_npu = torch.nn.functional.softmax(x_npu, dim=-1)
    
    # Get result back to CPU
    y_result = y_npu.cpu().numpy()
    
    # Calculate golden result
    max_val = np.max(input_x, axis=-1, keepdims=True)
    exp_x = np.exp(input_x - max_val)
    sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
    golden_y = exp_x / sum_exp
    
    # Verify
    rtol = 1e-3
    atol = 1e-5
    
    if np.allclose(y_result, golden_y, rtol=rtol, atol=atol):
        print('TEST PASSED!')
        print(f'Max absolute error: {np.max(np.abs(y_result - golden_y))}')
        return True
    else:
        print('TEST FAILED!')
        print(f'Max absolute error: {np.max(np.abs(y_result - golden_y))}')
        return False

if __name__ == '__main__':
    test_softmax_npu()
