#!/usr/bin/env python3
"""
MiniGPT 综合测试

测试 CPU 参考实现和 NPU 实现
"""

import sys
import os
import time

# 记录开始时间
START_TIME = time.time()

def main():
    print("="*60)
    print("MiniGPT AscendC NPU 实现测试")
    print("="*60)
    print(f"\n开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # 测试 1: CPU 参考实现
    print("\n" + "-"*60)
    print("Test 1: CPU 参考实现")
    print("-"*60)
    try:
        from minigpt_inference import test_minigpt_cpu
        results['CPU'] = test_minigpt_cpu()
    except Exception as e:
        print(f"❌ CPU 测试失败: {e}")
        results['CPU'] = False
    
    # 测试 2: NPU 实现
    print("\n" + "-"*60)
    print("Test 2: NPU 实现")
    print("-"*60)
    try:
        # 设置环境变量
        ASCEND_PATH = "/usr/local/Ascend/ascend-toolkit/latest"
        os.environ["LD_LIBRARY_PATH"] = f"{ASCEND_PATH}/lib64:{ASCEND_PATH}/runtime/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
        sys.path.insert(0, f"{ASCEND_PATH}/pyACL/python/site-packages")
        
        from minigpt_npu import test_minigpt_npu
        results['NPU'] = test_minigpt_npu()
    except Exception as e:
        print(f"❌ NPU 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['NPU'] = False
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    # 计算总耗时
    total_time = time.time() - START_TIME
    print(f"\n结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_time:.2f} 秒")
    
    # 返回总体结果
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠️ 部分测试失败")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
