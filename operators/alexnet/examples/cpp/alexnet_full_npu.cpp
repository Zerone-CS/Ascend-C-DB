/**
 * AlexNet Complete on NPU - True NPU Execution
 * 使用 aclnn API 在 NPU 上执行完整的 AlexNet
 */
#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include "acl/acl.h"
#include "aclnnop/aclnn_relu.h"
#include "aclnnop/aclnn_softmax.h"
#include "aclnnop/aclnn_matmul.h"
#include "aclnnop/aclnn_convolution.h"
#include "aclnnop/aclnn_max_pool.h"

#define CHECK_ACL(ret, msg) do { \
    if ((ret) != ACL_SUCCESS) { \
        std::cerr << "Error: " << (msg) << ", ret=" << (ret) << std::endl; \
        std::cerr << "ACL: " << aclGetRecentErrMsg() << std::endl; \
        return -1; \
    } \
} while(0)

class NPUTensor {
public:
    std::vector<int64_t> shape;
    int64_t byteSize = 0;
    void* devPtr = nullptr;
    aclTensor* tensor = nullptr;
    
    int create(const std::vector<int64_t>& sh, aclFormat fmt = ACL_FORMAT_NCHW) {
        shape = sh;
        int64_t numElements = 1;
        for (auto s : shape) numElements *= s;
        byteSize = numElements * sizeof(float);
        
        std::vector<int64_t> strides(shape.size());
        int64_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        
        if (aclrtMalloc(&devPtr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return -1;
        tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, fmt, shape.data(), shape.size(), devPtr);
        return tensor ? 0 : -1;
    }
    
    int fromHost(const float* data) { return aclrtMemcpy(devPtr, byteSize, data, byteSize, ACL_MEMCPY_HOST_TO_DEVICE); }
    int toHost(float* data) { return aclrtMemcpy(data, byteSize, devPtr, byteSize, ACL_MEMCPY_DEVICE_TO_HOST); }
    void destroy() {
        if (tensor) { aclDestroyTensor(tensor); tensor = nullptr; }
        if (devPtr) { aclrtFree(devPtr); devPtr = nullptr; }
    }
};

aclIntArray* makeIntArray(const std::vector<int64_t>& v) { return aclCreateIntArray(v.data(), v.size()); }

int conv2d_npu(NPUTensor& in, NPUTensor& w, NPUTensor& out, int stride, int pad, aclrtStream stream) {
    auto s = makeIntArray({stride, stride});
    auto p = makeIntArray({pad, pad, pad, pad});
    auto d = makeIntArray({1, 1});
    auto op = makeIntArray({0, 0});
    
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    aclError ret = aclnnConvolutionGetWorkspaceSize(in.tensor, w.tensor, nullptr, s, p, d, false, op, 1, out.tensor, 0, &ws, &exec);
    aclDestroyIntArray(s); aclDestroyIntArray(p); aclDestroyIntArray(d); aclDestroyIntArray(op);
    if (ret != ACL_SUCCESS) return ret;
    
    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnConvolution(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

int maxpool_npu(NPUTensor& in, NPUTensor& out, int k, int s, aclrtStream stream) {
    auto ks = makeIntArray({k, k});
    auto ss = makeIntArray({s, s});
    auto ps = makeIntArray({0, 0, 0, 0});
    auto ds = makeIntArray({1, 1});
    
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    aclError ret = aclnnMaxPoolGetWorkspaceSize(in.tensor, ks, ss, 0, ps, ds, 0, out.tensor, &ws, &exec);
    aclDestroyIntArray(ks); aclDestroyIntArray(ss); aclDestroyIntArray(ps); aclDestroyIntArray(ds);
    if (ret != ACL_SUCCESS) return ret;
    
    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnMaxPool(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

int relu_npu(NPUTensor& in, NPUTensor& out, aclrtStream stream) {
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    aclError ret = aclnnReluGetWorkspaceSize(in.tensor, out.tensor, &ws, &exec);
    if (ret != ACL_SUCCESS) return ret;
    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnRelu(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

int matmul_npu(NPUTensor& a, NPUTensor& b, NPUTensor& out, aclrtStream stream) {
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    aclError ret = aclnnMatmulGetWorkspaceSize(a.tensor, b.tensor, out.tensor, 0, &ws, &exec);
    if (ret != ACL_SUCCESS) return ret;
    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnMatmul(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

int softmax_npu(NPUTensor& in, NPUTensor& out, aclrtStream stream) {
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    aclError ret = aclnnSoftmaxGetWorkspaceSize(in.tensor, -1, out.tensor, &ws, &exec);
    if (ret != ACL_SUCCESS) return ret;
    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnSoftmax(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "   AlexNet Complete on NPU" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Init
    CHECK_ACL(aclInit(nullptr), "aclInit");
    CHECK_ACL(aclrtSetDevice(0), "setDevice");
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream), "createStream");
    std::cout << "[NPU] Device initialized" << std::endl;
    
    const int B = 4, C = 10;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    auto randVec = [&](int64_t n) { std::vector<float> v(n); for(auto& x:v) x=dist(gen); return v; };
    
    // Host data
    auto h_input = randVec(B*3*32*32);
    auto h_conv1_w = randVec(16*3*5*5);
    auto h_conv2_w = randVec(32*16*3*3);
    auto h_conv3_w = randVec(64*32*3*3);
    auto h_fc1_w = randVec(1024*256);
    auto h_fc2_w = randVec(256*C);
    std::vector<float> h_output(B*C);
    
    std::cout << "\nArchitecture: Conv1->Pool->Conv2->Pool->Conv3->Pool->FC1->FC2" << std::endl;
    std::cout << "Input: [" << B << ",3,32,32] -> Output: [" << B << "," << C << "]" << std::endl;
    
    // Create tensors
    NPUTensor input, conv1_w, conv1_out, relu1, pool1;
    NPUTensor conv2_w, conv2_out, relu2, pool2;
    NPUTensor conv3_w, conv3_out, relu3, pool3;
    NPUTensor flat, fc1_w, fc1_out, relu4, fc2_w, fc2_out, softmax_out;
    
    std::cout << "\n[NPU] Allocating tensors..." << std::endl;
    CHECK_ACL(input.create({B,3,32,32})?-1:0, "input");
    CHECK_ACL(conv1_w.create({16,3,5,5})?-1:0, "conv1_w");
    CHECK_ACL(conv1_out.create({B,16,32,32})?-1:0, "conv1_out");
    CHECK_ACL(relu1.create({B,16,32,32})?-1:0, "relu1");
    CHECK_ACL(pool1.create({B,16,16,16})?-1:0, "pool1");
    
    CHECK_ACL(conv2_w.create({32,16,3,3})?-1:0, "conv2_w");
    CHECK_ACL(conv2_out.create({B,32,16,16})?-1:0, "conv2_out");
    CHECK_ACL(relu2.create({B,32,16,16})?-1:0, "relu2");
    CHECK_ACL(pool2.create({B,32,8,8})?-1:0, "pool2");
    
    CHECK_ACL(conv3_w.create({64,32,3,3})?-1:0, "conv3_w");
    CHECK_ACL(conv3_out.create({B,64,8,8})?-1:0, "conv3_out");
    CHECK_ACL(relu3.create({B,64,8,8})?-1:0, "relu3");
    CHECK_ACL(pool3.create({B,64,4,4})?-1:0, "pool3");
    
    CHECK_ACL(flat.create({B,1024}, ACL_FORMAT_ND)?-1:0, "flat");
    CHECK_ACL(fc1_w.create({1024,256}, ACL_FORMAT_ND)?-1:0, "fc1_w");
    CHECK_ACL(fc1_out.create({B,256}, ACL_FORMAT_ND)?-1:0, "fc1_out");
    CHECK_ACL(relu4.create({B,256}, ACL_FORMAT_ND)?-1:0, "relu4");
    CHECK_ACL(fc2_w.create({256,C}, ACL_FORMAT_ND)?-1:0, "fc2_w");
    CHECK_ACL(fc2_out.create({B,C}, ACL_FORMAT_ND)?-1:0, "fc2_out");
    CHECK_ACL(softmax_out.create({B,C}, ACL_FORMAT_ND)?-1:0, "softmax_out");
    
    // H2D
    std::cout << "[NPU] H2D transfer..." << std::endl;
    CHECK_ACL(input.fromHost(h_input.data()), "H2D input");
    CHECK_ACL(conv1_w.fromHost(h_conv1_w.data()), "H2D conv1_w");
    CHECK_ACL(conv2_w.fromHost(h_conv2_w.data()), "H2D conv2_w");
    CHECK_ACL(conv3_w.fromHost(h_conv3_w.data()), "H2D conv3_w");
    CHECK_ACL(fc1_w.fromHost(h_fc1_w.data()), "H2D fc1_w");
    CHECK_ACL(fc2_w.fromHost(h_fc2_w.data()), "H2D fc2_w");
    
    // Forward
    std::cout << "\n[NPU] Forward pass..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    
    std::cout << "  Conv1" << std::flush;
    CHECK_ACL(conv2d_npu(input, conv1_w, conv1_out, 1, 2, stream), "conv1");
    std::cout << " -> ReLU" << std::flush;
    CHECK_ACL(relu_npu(conv1_out, relu1, stream), "relu1");
    std::cout << " -> Pool" << std::flush;
    CHECK_ACL(maxpool_npu(relu1, pool1, 2, 2, stream), "pool1");
    std::cout << std::endl;
    
    std::cout << "  Conv2" << std::flush;
    CHECK_ACL(conv2d_npu(pool1, conv2_w, conv2_out, 1, 1, stream), "conv2");
    std::cout << " -> ReLU" << std::flush;
    CHECK_ACL(relu_npu(conv2_out, relu2, stream), "relu2");
    std::cout << " -> Pool" << std::flush;
    CHECK_ACL(maxpool_npu(relu2, pool2, 2, 2, stream), "pool2");
    std::cout << std::endl;
    
    std::cout << "  Conv3" << std::flush;
    CHECK_ACL(conv2d_npu(pool2, conv3_w, conv3_out, 1, 1, stream), "conv3");
    std::cout << " -> ReLU" << std::flush;
    CHECK_ACL(relu_npu(conv3_out, relu3, stream), "relu3");
    std::cout << " -> Pool" << std::flush;
    CHECK_ACL(maxpool_npu(relu3, pool3, 2, 2, stream), "pool3");
    std::cout << std::endl;
    
    std::cout << "  Flatten" << std::flush;
    aclrtMemcpy(flat.devPtr, flat.byteSize, pool3.devPtr, pool3.byteSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
    std::cout << " -> FC1" << std::flush;
    CHECK_ACL(matmul_npu(flat, fc1_w, fc1_out, stream), "fc1");
    std::cout << " -> ReLU" << std::flush;
    CHECK_ACL(relu_npu(fc1_out, relu4, stream), "relu4");
    std::cout << std::endl;
    
    std::cout << "  FC2" << std::flush;
    CHECK_ACL(matmul_npu(relu4, fc2_w, fc2_out, stream), "fc2");
    std::cout << " -> Softmax" << std::flush;
    CHECK_ACL(softmax_npu(fc2_out, softmax_out, stream), "softmax");
    std::cout << std::endl;
    
    aclrtSynchronizeStream(stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // D2H
    std::cout << "\n[NPU] D2H transfer..." << std::endl;
    CHECK_ACL(softmax_out.toHost(h_output.data()), "D2H");
    
    // Results
    std::cout << "\n============================================" << std::endl;
    std::cout << "NPU compute time: " << ms << " ms" << std::endl;
    std::cout << "\nPredictions:" << std::endl;
    for (int b = 0; b < B; b++) {
        float sum = 0; int maxIdx = 0; float maxVal = h_output[b*C];
        for (int c = 0; c < C; c++) {
            sum += h_output[b*C+c];
            if (h_output[b*C+c] > maxVal) { maxVal = h_output[b*C+c]; maxIdx = c; }
        }
        std::cout << "  Sample " << b << ": Class " << maxIdx << ", Conf " << maxVal << ", Sum " << sum << std::endl;
    }
    
    // Cleanup
    input.destroy(); conv1_w.destroy(); conv1_out.destroy(); relu1.destroy(); pool1.destroy();
    conv2_w.destroy(); conv2_out.destroy(); relu2.destroy(); pool2.destroy();
    conv3_w.destroy(); conv3_out.destroy(); relu3.destroy(); pool3.destroy();
    flat.destroy(); fc1_w.destroy(); fc1_out.destroy(); relu4.destroy();
    fc2_w.destroy(); fc2_out.destroy(); softmax_out.destroy();
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "*** AlexNet Complete NPU SUCCESS! ***" << std::endl;
    std::cout << "============================================" << std::endl;
    return 0;
}
