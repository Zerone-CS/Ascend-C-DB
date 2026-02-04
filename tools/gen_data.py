#!/usr/bin/env python3
"""
通用测试数据生成工具

用法:
    python gen_data.py --shape 8,256 --dtype float32 --output input.bin
    python gen_data.py --shape 8,256 --dtype float16 --output input.bin --mode random
    python gen_data.py --shape 8,256 --output input.bin --mode zeros
    python gen_data.py --shape 8,256 --output input.bin --mode ones
    python gen_data.py --shape 8,256 --output input.bin --mode range
"""

import argparse
import numpy as np
import os

def generate_data(shape, dtype, mode='random', low=-1.0, high=1.0):
    """生成测试数据"""
    if mode == 'random':
        data = np.random.uniform(low, high, shape).astype(dtype)
    elif mode == 'zeros':
        data = np.zeros(shape, dtype=dtype)
    elif mode == 'ones':
        data = np.ones(shape, dtype=dtype)
    elif mode == 'range':
        data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    elif mode == 'normal':
        data = np.random.randn(*shape).astype(dtype)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return data

def main():
    parser = argparse.ArgumentParser(description='生成测试数据')
    parser.add_argument('--shape', type=str, required=True, help='数据形状, 如 8,256')
    parser.add_argument('--dtype', type=str, default='float32', 
                        choices=['float32', 'float16', 'int32', 'int8'],
                        help='数据类型')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--mode', type=str, default='random',
                        choices=['random', 'zeros', 'ones', 'range', 'normal'],
                        help='生成模式')
    parser.add_argument('--low', type=float, default=-1.0, help='random模式下界')
    parser.add_argument('--high', type=float, default=1.0, help='random模式上界')
    
    args = parser.parse_args()
    
    # 解析shape
    shape = tuple(int(x) for x in args.shape.split(','))
    
    # 映射dtype
    dtype_map = {
        'float32': np.float32,
        'float16': np.float16,
        'int32': np.int32,
        'int8': np.int8,
    }
    dtype = dtype_map[args.dtype]
    
    # 生成数据
    data = generate_data(shape, dtype, args.mode, args.low, args.high)
    
    # 保存
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    data.tofile(args.output)
    
    print(f'✅ 生成数据:')
    print(f'   Shape: {shape}')
    print(f'   Dtype: {args.dtype}')
    print(f'   Mode: {args.mode}')
    print(f'   File: {args.output}')
    print(f'   Size: {data.nbytes} bytes')

if __name__ == '__main__':
    main()
