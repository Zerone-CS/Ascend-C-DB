import numpy as np
import sys
sys.path.insert(0, '.')
import relu_custom_npu

print('=== Testing AscendC ReluCustom via PyBind11 ===')

x = np.random.randn(2048).astype(np.float32)
y = relu_custom_npu.relu(x)
expected = np.maximum(x, 0)

print(f'Input[0:8]:    {x[:8]}')
print(f'Output[0:8]:   {y[:8]}')
print(f'Expected[0:8]: {expected[:8]}')

match = np.allclose(y, expected, rtol=1e-5)
print(f'All {len(x)} elements match: {match}')

if match:
    print('\n*** AscendC ReluCustom via PyBind11 on NPU: SUCCESS! ***')
else:
    diff = np.abs(y - expected).max()
    print(f'MISMATCH: max diff = {diff}')

relu_custom_npu.finalize()
