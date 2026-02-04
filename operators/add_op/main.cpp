#include "data_utils.h"
#include "acl/acl.h"
#include <string>
#include <cstdio>

int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = 8;
    uint32_t totalLength = 8 * 256;
    size_t inputByteSize = totalLength * sizeof(float);
    size_t outputByteSize = totalLength * sizeof(float);

    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    uint8_t *xHost, *yHost, *zHost;
    uint8_t *xDevice, *yDevice, *zDevice;
    aclrtMallocHost((void**)(&xHost), inputByteSize);
    aclrtMallocHost((void**)(&yHost), inputByteSize);
    aclrtMallocHost((void**)(&zHost), outputByteSize);
    aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, yHost, inputByteSize);
    aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // Load kernel binary
    const char* kernelPath = "./add_custom_kernel.o";
    aclrtBinHandle binHandle = nullptr;
    aclError ret = aclrtBinaryLoadFromFile(kernelPath, nullptr, &binHandle);
    if (ret != ACL_SUCCESS) {
        printf("aclrtBinaryLoadFromFile failed, ret=%d\n", ret);
        return -1;
    }
    printf("Kernel binary loaded successfully\n");

    aclrtFuncHandle funcHandle = nullptr;
    ret = aclrtBinaryGetFunction(binHandle, "add_custom", &funcHandle);
    if (ret != ACL_SUCCESS) {
        printf("aclrtBinaryGetFunction failed, ret=%d\n", ret);
        return -1;
    }
    printf("Function handle obtained successfully\n");

    struct {
        uint64_t x;
        uint64_t y;
        uint64_t z;
        uint32_t totalLength;
    } args;
    args.x = (uint64_t)xDevice;
    args.y = (uint64_t)yDevice;
    args.z = (uint64_t)zDevice;
    args.totalLength = totalLength;
    
    ret = aclrtLaunchKernel(funcHandle, blockDim, &args, sizeof(args), stream);
    if (ret != ACL_SUCCESS) {
        printf("aclrtLaunchKernel failed, ret=%d\n", ret);
        return -1;
    }
    
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        printf("aclrtSynchronizeStream failed, ret=%d\n", ret);
        return -1;
    }

    aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile("./output/output_z.bin", zHost, outputByteSize);
    
    printf("Add operator executed successfully!\n");
    printf("Input x[0:3]: %.2f, %.2f, %.2f, %.2f\n",
           ((float*)xHost)[0], ((float*)xHost)[1], ((float*)xHost)[2], ((float*)xHost)[3]);
    printf("Input y[0:3]: %.2f, %.2f, %.2f, %.2f\n",
           ((float*)yHost)[0], ((float*)yHost)[1], ((float*)yHost)[2], ((float*)yHost)[3]);
    printf("Output z[0:3]: %.2f, %.2f, %.2f, %.2f\n",
           ((float*)zHost)[0], ((float*)zHost)[1], ((float*)zHost)[2], ((float*)zHost)[3]);

    aclrtBinaryUnLoad(binHandle);
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(zDevice);
    aclrtFreeHost(xHost);
    aclrtFreeHost(yHost);
    aclrtFreeHost(zHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
