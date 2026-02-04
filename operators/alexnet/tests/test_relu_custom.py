#!/usr/bin/env python3
"""Test custom ReluCustom operator on NPU via aclnn API"""
import numpy as np
import acl

def check(ret, msg):
    if ret != 0:
        raise RuntimeError(f"{msg} failed with code {ret}: {acl.get_recent_err_msg()}")

def main():
    print("=== Testing AscendC ReluCustom on NPU ===")
    
    # Initialize ACL
    check(acl.init(), "acl.init")
    check(acl.rt.set_device(0), "set_device")
    context, _ = acl.rt.create_context(0)
    stream, _ = acl.rt.create_stream()
    
    print(f"[ACL] Device: {acl.get_soc_name()}")
    
    # Test data
    N = 8 * 256  # Must be divisible by 8*256 for our kernel
    x_host = np.random.randn(N).astype(np.float32)
    expected = np.maximum(x_host, 0)
    
    print(f"Input shape: ({N},)")
    print(f"Input[0:8]: {x_host[:8]}")
    
    # Allocate device memory
    x_dev, _ = acl.rt.malloc(x_host.nbytes, acl.rt.ACL_MEM_MALLOC_HUGE_FIRST)
    y_dev, _ = acl.rt.malloc(x_host.nbytes, acl.rt.ACL_MEM_MALLOC_HUGE_FIRST)
    
    # Copy input to device
    acl.rt.memcpy(x_dev, x_host.nbytes, acl.util.numpy_to_ptr(x_host), x_host.nbytes, 
                  acl.rt.ACL_MEMCPY_HOST_TO_DEVICE)
    
    # Try to call the custom operator
    # The operator was registered with aclnn, so we need to use aclnn API
    try:
        # Create tensor descriptors
        x_desc = acl.create_tensor_desc(acl.ACL_FLOAT, [N], acl.ACL_FORMAT_ND)
        y_desc = acl.create_tensor_desc(acl.ACL_FLOAT, [N], acl.ACL_FORMAT_ND)
        
        # Create data buffers
        x_buf = acl.create_data_buffer(x_dev, x_host.nbytes)
        y_buf = acl.create_data_buffer(y_dev, x_host.nbytes)
        
        # Try execute via acl.op
        input_list = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(input_list, x_buf)
        
        output_list = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(output_list, y_buf)
        
        # The operator name should be "ReluCustom"
        ret = acl.op.execute_v2("ReluCustom", [x_desc], [x_buf], [y_desc], [y_buf], None, stream)
        print(f"acl.op.execute_v2 returned: {ret}")
        
        if ret == 0:
            acl.rt.synchronize_stream(stream)
            
            # Copy output back
            y_host = np.zeros_like(x_host)
            acl.rt.memcpy(acl.util.numpy_to_ptr(y_host), y_host.nbytes, y_dev, y_host.nbytes,
                          acl.rt.ACL_MEMCPY_DEVICE_TO_HOST)
            
            print(f"Output[0:8]: {y_host[:8]}")
            print(f"Expected[0:8]: {expected[:8]}")
            
            if np.allclose(y_host, expected, rtol=1e-5):
                print("\n*** SUCCESS: ReluCustom works on NPU! ***")
            else:
                diff = np.abs(y_host - expected).max()
                print(f"\n*** MISMATCH: max diff = {diff} ***")
        else:
            print(f"Operator execution failed. Trying alternative...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Cleanup
    acl.rt.free(x_dev)
    acl.rt.free(y_dev)
    acl.rt.destroy_stream(stream)
    acl.rt.destroy_context(context)
    acl.rt.reset_device(0)
    acl.finalize()
    
    print("[ACL] Finalized")

if __name__ == "__main__":
    main()
