#include "data_utils.h"
#include "acl/acl.h"
#include <cstdio>
#include <cmath>
#include <cstring>

int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = 1;
    uint32_t totalLength = 8 * 256;  // 8 rows x 256 cols
    size_t dataByteSize = totalLength * sizeof(float);

    // Init ACL
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // Allocate host & device memory
    uint8_t *xHost, *gateHost, *yHost;
    uint8_t *xDevice, *gateDevice, *yDevice;
    aclrtMallocHost((void**)(&xHost), dataByteSize);
    aclrtMallocHost((void**)(&gateHost), dataByteSize);
    aclrtMallocHost((void**)(&yHost), dataByteSize);
    aclrtMalloc((void**)&xDevice, dataByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&gateDevice, dataByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&yDevice, dataByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // Read input data
    ReadFile("./input/input_x.bin", dataByteSize, xHost, dataByteSize);
    ReadFile("./input/input_gate.bin", dataByteSize, gateHost, dataByteSize);
    printf("Input x[0:4]: %.6f, %.6f, %.6f, %.6f\n",
           ((float*)xHost)[0], ((float*)xHost)[1], ((float*)xHost)[2], ((float*)xHost)[3]);
    printf("Input gate[0:4]: %.6f, %.6f, %.6f, %.6f\n",
           ((float*)gateHost)[0], ((float*)gateHost)[1], ((float*)gateHost)[2], ((float*)gateHost)[3]);

    // Copy to device
    aclrtMemcpy(xDevice, dataByteSize, xHost, dataByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(gateDevice, dataByteSize, gateHost, dataByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // Load kernel binary
    aclrtBinHandle binHandle = nullptr;
    aclError ret = aclrtBinaryLoadFromFile("./swiglu_custom_kernel.o", nullptr, &binHandle);
    if (ret != ACL_SUCCESS) {
        printf("aclrtBinaryLoadFromFile failed, ret=%d\n", ret);
        return -1;
    }
    printf("Kernel binary loaded\n");

    // Get function handle
    aclrtFuncHandle funcHandle = nullptr;
    ret = aclrtBinaryGetFunction(binHandle, "swiglu_custom", &funcHandle);
    if (ret != ACL_SUCCESS) {
        printf("aclrtBinaryGetFunction failed, ret=%d\n", ret);
        return -1;
    }
    printf("Function handle obtained\n");

    // Launch kernel
    struct {
        uint64_t x;
        uint64_t gate;
        uint64_t y;
        uint32_t totalLength;
    } args;
    args.x = (uint64_t)xDevice;
    args.gate = (uint64_t)gateDevice;
    args.y = (uint64_t)yDevice;
    args.totalLength = totalLength;

    ret = aclrtLaunchKernel(funcHandle, blockDim, &args, sizeof(args), stream);
    if (ret != ACL_SUCCESS) {
        printf("aclrtLaunchKernel failed, ret=%d\n", ret);
        return -1;
    }
    printf("Kernel launched: blockDim=%u, totalLength=%u\n", blockDim, totalLength);

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        printf("aclrtSynchronizeStream failed, ret=%d\n", ret);
        return -1;
    }
    printf("Stream synchronized\n");

    // Copy result back
    aclrtMemcpy(yHost, dataByteSize, yDevice, dataByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    printf("Output y[0:4]: %.6f, %.6f, %.6f, %.6f\n",
           ((float*)yHost)[0], ((float*)yHost)[1], ((float*)yHost)[2], ((float*)yHost)[3]);
    WriteFile("./output/output_y.bin", yHost, dataByteSize);

    // Quick host-side verification
    float *xf = (float*)xHost;
    float *gf = (float*)gateHost;
    float *yf = (float*)yHost;
    float maxErr = 0.0f;
    for (uint32_t i = 0; i < totalLength; i++) {
        // swiglu: y = x * swish(gate) = x * gate * sigmoid(gate)
        float sig = 1.0f / (1.0f + expf(-gf[i]));
        float expected = xf[i] * gf[i] * sig;
        float err = fabsf(yf[i] - expected);
        if (err > maxErr) maxErr = err;
    }
    printf("\nMax error: %.2e\n", maxErr);
    if (maxErr < 1e-5f) {
        printf("\033[32m\u2705 SwiGLU TEST PASSED!\033[0m\n");
    } else {
        printf("\033[31m\u274c SwiGLU TEST FAILED! (maxErr=%.2e)\033[0m\n", maxErr);
    }

    // Cleanup
    aclrtBinaryUnLoad(binHandle);
    aclrtFree(xDevice);
    aclrtFree(gateDevice);
    aclrtFree(yDevice);
    aclrtFreeHost(xHost);
    aclrtFreeHost(gateHost);
    aclrtFreeHost(yHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
