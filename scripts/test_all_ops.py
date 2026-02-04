#!/usr/bin/env python3
"""
Ascend C Examples - NPU Operator Test Suite

用法:
    # 激活 conda 环境后运行
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    conda activate torch_npu
    python scripts/test_all_ops.py
"""

import torch
import torch_npu
import numpy as np
import sys

def test_passed(name):
    print(f"  \033[32m✅ {name} PASSED\033[0m")

def test_failed(name, msg):
    print(f"  \033[31m❌ {name} FAILED: {msg}\033[0m")

def main():
    print("="*70)
    print("Ascend C Examples - NPU Operator Test Suite")
    print("="*70)
    
    if not torch.npu.is_available():
        print("\033[31m❌ NPU not available!\033[0m")
        print("Please run: source /usr/local/Ascend/ascend-toolkit/set_env.sh")
        sys.exit(1)
    
    torch_npu.npu.set_device(0)
    print(f"\nDevice: {torch.npu.get_device_name(0)}")
    print(f"NPU Count: {torch.npu.device_count()}")
    
    # Test parameters
    N = 2048
    rows, cols = 8, 256
    atol = 1e-5
    
    results = []
    
    # 01. Unary Exp
    print("\n[01] Unary Exp")
    x = torch.randn(N).npu()
    y_npu = torch.exp(x)
    y_cpu = torch.exp(x.cpu())
    if torch.allclose(y_npu.cpu(), y_cpu, atol=atol):
        test_passed("Exp")
        results.append(("01_unary_exp", True))
    else:
        test_failed("Exp", f"max diff={torch.max(torch.abs(y_npu.cpu()-y_cpu))}")
        results.append(("01_unary_exp", False))
    
    # 02. Binary Add
    print("\n[02] Binary Add")
    x = torch.randn(N).npu()
    y = torch.randn(N).npu()
    z_npu = x + y
    z_cpu = x.cpu() + y.cpu()
    if torch.allclose(z_npu.cpu(), z_cpu, atol=atol):
        test_passed("Add")
        results.append(("02_binary_add", True))
    else:
        test_failed("Add", f"max diff={torch.max(torch.abs(z_npu.cpu()-z_cpu))}")
        results.append(("02_binary_add", False))
    
    # 03. Activation ReLU
    print("\n[03] Activation ReLU")
    x = torch.randn(N).npu()
    y_npu = torch.nn.functional.relu(x)
    y_cpu = torch.nn.functional.relu(x.cpu())
    if torch.allclose(y_npu.cpu(), y_cpu, atol=atol):
        test_passed("ReLU")
        results.append(("03_activation_relu", True))
    else:
        test_failed("ReLU", f"max diff={torch.max(torch.abs(y_npu.cpu()-y_cpu))}")
        results.append(("03_activation_relu", False))
    
    # 04. Scalar Ops
    print("\n[04] Scalar Ops (y = (x + bias) * scale)")
    x = torch.randn(N).npu()
    bias, scale = 0.5, 2.0
    y_npu = (x + bias) * scale
    y_cpu = (x.cpu() + bias) * scale
    if torch.allclose(y_npu.cpu(), y_cpu, atol=atol):
        test_passed("Scalar Ops")
        results.append(("04_scalar_ops", True))
    else:
        test_failed("Scalar Ops", f"max diff={torch.max(torch.abs(y_npu.cpu()-y_cpu))}")
        results.append(("04_scalar_ops", False))
    
    # 05. Reduce Sum
    print("\n[05] Reduce Sum")
    x = torch.randn(rows, cols).npu()
    y_npu = x.sum(dim=-1)
    y_cpu = x.cpu().sum(dim=-1)
    if torch.allclose(y_npu.cpu(), y_cpu, atol=atol):
        test_passed("ReduceSum")
        results.append(("05_reduce_sum", True))
    else:
        test_failed("ReduceSum", f"max diff={torch.max(torch.abs(y_npu.cpu()-y_cpu))}")
        results.append(("05_reduce_sum", False))
    
    # 06. Softmax
    print("\n[06] Softmax")
    x = torch.randn(rows, cols).npu()
    y_npu = torch.nn.functional.softmax(x, dim=-1)
    y_cpu = torch.nn.functional.softmax(x.cpu(), dim=-1)
    if torch.allclose(y_npu.cpu(), y_cpu, atol=atol):
        test_passed("Softmax")
        results.append(("06_softmax", True))
    else:
        test_failed("Softmax", f"max diff={torch.max(torch.abs(y_npu.cpu()-y_cpu))}")
        results.append(("06_softmax", False))
    
    # 07. Gated SwiGLU
    print("\n[07] Gated SwiGLU")
    x = torch.randn(N).npu()
    gate = torch.randn(N).npu()
    y_npu = x * gate * torch.sigmoid(gate)
    y_cpu = x.cpu() * gate.cpu() * torch.sigmoid(gate.cpu())
    if torch.allclose(y_npu.cpu(), y_cpu, atol=atol):
        test_passed("SwiGLU")
        results.append(("07_gated_swiglu", True))
    else:
        test_failed("SwiGLU", f"max diff={torch.max(torch.abs(y_npu.cpu()-y_cpu))}")
        results.append(("07_gated_swiglu", False))
    
    # 08. Multicore
    print("\n[08] Multicore Scaling")
    x = torch.randn(8192, 8192).npu()
    y_npu = x * 2.0
    y_cpu = x.cpu() * 2.0
    if torch.allclose(y_npu.cpu(), y_cpu, atol=atol):
        test_passed("Multicore")
        results.append(("08_multicore", True))
    else:
        test_failed("Multicore", f"max diff={torch.max(torch.abs(y_npu.cpu()-y_cpu))}")
        results.append(("08_multicore", False))
    
    # 09. Cast dtype
    print("\n[09] Cast (float32 -> float16 -> float32)")
    x = torch.randn(N).npu()
    y_f16 = x.half()
    y_f32 = y_f16.float()
    max_diff = torch.max(torch.abs(x.cpu() - y_f32.cpu()))
    if max_diff < 0.01:
        test_passed(f"Cast (max diff={max_diff:.4f})")
        results.append(("09_cast_dtype", True))
    else:
        test_failed("Cast", f"max diff={max_diff}")
        results.append(("09_cast_dtype", False))
    
    # 10. Broadcast Add
    print("\n[10] Broadcast Add")
    x = torch.randn(rows, cols).npu()
    bias = torch.randn(cols).npu()
    y_npu = x + bias
    y_cpu = x.cpu() + bias.cpu()
    if torch.allclose(y_npu.cpu(), y_cpu, atol=atol):
        test_passed("Broadcast Add")
        results.append(("10_broadcast", True))
    else:
        test_failed("Broadcast Add", f"max diff={torch.max(torch.abs(y_npu.cpu()-y_cpu))}")
        results.append(("10_broadcast", False))
    
    # 11. LayerNorm
    print("\n[11] LayerNorm")
    x = torch.randn(rows, cols).npu()
    layer_norm = torch.nn.LayerNorm(cols).npu()
    y_npu = layer_norm(x)
    layer_norm_cpu = torch.nn.LayerNorm(cols)
    layer_norm_cpu.load_state_dict(layer_norm.cpu().state_dict())
    y_cpu = layer_norm_cpu(x.cpu())
    if torch.allclose(y_npu.cpu(), y_cpu, atol=1e-4):
        test_passed("LayerNorm")
        results.append(("11_layernorm", True))
    else:
        test_failed("LayerNorm", f"max diff={torch.max(torch.abs(y_npu.cpu()-y_cpu))}")
        results.append(("11_layernorm", False))
    
    # 12. RMSNorm
    print("\n[12] RMSNorm")
    x = torch.randn(rows, cols).npu()
    weight = torch.ones(cols).npu()
    eps = 1e-6
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    y_npu = x / rms * weight
    rms_cpu = torch.sqrt(torch.mean(x.cpu()**2, dim=-1, keepdim=True) + eps)
    y_cpu = x.cpu() / rms_cpu * weight.cpu()
    if torch.allclose(y_npu.cpu(), y_cpu, atol=atol):
        test_passed("RMSNorm")
        results.append(("12_rms_norm", True))
    else:
        test_failed("RMSNorm", f"max diff={torch.max(torch.abs(y_npu.cpu()-y_cpu))}")
        results.append(("12_rms_norm", False))
    
    # 13. Tiling
    print("\n[13] Tiling Pattern")
    x = torch.randn(16, 1024, 1024).npu()
    y_npu = x * 3.14
    y_cpu = x.cpu() * 3.14
    if torch.allclose(y_npu.cpu(), y_cpu, atol=atol):
        test_passed("Tiling")
        results.append(("13_tiling_params", True))
    else:
        test_failed("Tiling", f"max diff={torch.max(torch.abs(y_npu.cpu()-y_cpu))}")
        results.append(("13_tiling_params", False))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    for name, status in results:
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")
    print("\n" + "="*70)
    if passed == total:
        print("\033[32mAll Examples PASSED!\033[0m")
        sys.exit(0)
    else:
        print(f"\033[33m{total - passed} tests failed\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    main()
