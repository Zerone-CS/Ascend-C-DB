#include "data_utils.h"
#include "acl/acl.h"
#include <string>
#include <cstdio>

int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = 1;
    uint32_t totalRows = 8;
    uint32_t rowSize = 256;
    size_t inputByteSize = totalRows * rowSize * sizeof(float);
    size_t outputByteSize = totalRows * rowSize * sizeof(float);

    aclInit(nullptr);
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    uint8_t *xHost, *yHost;
    uint8_t *xDevice, *yDevice;
    aclrtMallocHost((void**)(&xHost), inputByteSize);
    aclrtMallocHost((void**)(&yHost), outputByteSize);
    aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&yDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    // Initialize output buffer to non-zero to detect if kernel writes anything
    memset(yHost, 0xFF, outputByteSize);
    aclrtMemcpy(yDevice, outputByteSize, yHost, outputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    
    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
    printf("Input x[0:4]: %.6f, %.6f, %.6f, %.6f\n",
           ((float*)xHost)[0], ((float*)xHost)[1], ((float*)xHost)[2], ((float*)xHost)[3]);
    aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    const char* kernelPath = "./softmax_custom_kernel.o";
    aclrtBinHandle binHandle = nullptr;
    aclError ret = aclrtBinaryLoadFromFile(kernelPath, nullptr, &binHandle);
    if (ret != ACL_SUCCESS) {
        printf("aclrtBinaryLoadFromFile failed, ret=%d\n", ret);
        return -1;
    }
    printf("Kernel binary loaded successfully\n");

    aclrtFuncHandle funcHandle = nullptr;
    ret = aclrtBinaryGetFunction(binHandle, "softmax_custom", &funcHandle);
    if (ret != ACL_SUCCESS) {
        printf("aclrtBinaryGetFunction failed, ret=%d\n", ret);
        return -1;
    }
    printf("Function handle obtained successfully\n");

    struct {
        uint64_t x;
        uint64_t y;
        uint32_t totalRows;
        uint32_t rowSize;
    } args;
    args.x = (uint64_t)xDevice;
    args.y = (uint64_t)yDevice;
    args.totalRows = totalRows;
    args.rowSize = rowSize;
    
    printf("Args: x=%lx, y=%lx, totalRows=%u, rowSize=%u\n", 
           args.x, args.y, args.totalRows, args.rowSize);

    ret = aclrtLaunchKernel(funcHandle, blockDim, &args, sizeof(args), stream);
    if (ret != ACL_SUCCESS) {
        printf("aclrtLaunchKernel failed, ret=%d\n", ret);
        return -1;
    }
    printf("Kernel launched with blockDim=%u, totalRows=%u, rowSize=%u\n", blockDim, totalRows, rowSize);

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        printf("aclrtSynchronizeStream failed, ret=%d\n", ret);
        return -1;
    }
    printf("Stream synchronized\n");

    aclrtMemcpy(yHost, outputByteSize, yDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    printf("Output y[0:4]: %.6f, %.6f, %.6f, %.6f\n",
           ((float*)yHost)[0], ((float*)yHost)[1], ((float*)yHost)[2], ((float*)yHost)[3]);
    WriteFile("./output/output_y.bin", yHost, outputByteSize);

    aclrtBinaryUnLoad(binHandle);
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFreeHost(xHost);
    aclrtFreeHost(yHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    printf("Softmax operator completed successfully!\n");
    return 0;
}
