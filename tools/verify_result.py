#!/usr/bin/env python3
"""
通用结果验证工具

用法:
    python verify_result.py --output output.bin --golden golden.bin --dtype float32
    python verify_result.py --output output.bin --golden golden.bin --atol 1e-5 --rtol 1e-5
"""

import argparse
import numpy as np
import sys

def load_bin(path, dtype):
    """加载二进制文件"""
    return np.fromfile(path, dtype=dtype)

def verify(output, golden, atol=1e-5, rtol=1e-5):
    """验证结果"""
    if output.shape != golden.shape:
        print(f'❌ Shape不匹配: output={output.shape}, golden={golden.shape}')
        return False
    
    abs_diff = np.abs(output - golden)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    rel_diff = abs_diff / (np.abs(golden) + 1e-10)
    max_rel_diff = np.max(rel_diff)
    
    passed = np.allclose(output, golden, atol=atol, rtol=rtol)
    
    print(f'\n验证结果:')
    print(f'  Shape: {output.shape}')
    print(f'  Max Abs Diff: {max_abs_diff:.2e}')
    print(f'  Mean Abs Diff: {mean_abs_diff:.2e}')
    print(f'  Max Rel Diff: {max_rel_diff:.2e}')
    print(f'  Tolerance: atol={atol}, rtol={rtol}')
    
    if passed:
        print(f'\n✅ TEST PASSED!')
    else:
        print(f'\n❌ TEST FAILED!')
        # 显示最大差异位置
        max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f'  Max diff at index {max_idx}:')
        print(f'    output: {output[max_idx]}')
        print(f'    golden: {golden[max_idx]}')
    
    return passed

def main():
    parser = argparse.ArgumentParser(description='验证计算结果')
    parser.add_argument('--output', type=str, required=True, help='输出文件')
    parser.add_argument('--golden', type=str, required=True, help='参考结果文件')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float16', 'int32'],
                        help='数据类型')
    parser.add_argument('--atol', type=float, default=1e-5, help='绝对容差')
    parser.add_argument('--rtol', type=float, default=1e-5, help='相对容差')
    parser.add_argument('--shape', type=str, default=None, help='数据形状 (可选)')
    
    args = parser.parse_args()
    
    dtype_map = {
        'float32': np.float32,
        'float16': np.float16,
        'int32': np.int32,
    }
    dtype = dtype_map[args.dtype]
    
    output = load_bin(args.output, dtype)
    golden = load_bin(args.golden, dtype)
    
    if args.shape:
        shape = tuple(int(x) for x in args.shape.split(','))
        output = output.reshape(shape)
        golden = golden.reshape(shape)
    
    passed = verify(output, golden, args.atol, args.rtol)
    sys.exit(0 if passed else 1)

if __name__ == '__main__':
    main()
