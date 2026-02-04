/**
 * AlexNet on NPU - True NPU Execution
 * 
 * 使用 aclnn API 在 NPU 上执行计算
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
#include "aclnnop/aclnn_add.h"

#define CHECK_RET(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "Error at line " << __LINE__ << ": " << msg << std::endl; \
        std::cerr << "ACL Error: " << aclGetRecentErrMsg() << std::endl; \
        return -1; \
    } \
} while(0)

class NPUContext {
public:
    aclrtStream stream = nullptr;
    
    int init(int deviceId = 0) {
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS && ret != 100002) {
            std::cerr << "aclInit failed: " << ret << std::endl;
            return -1;
        }
        ret = aclrtSetDevice(deviceId);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtSetDevice failed: " << ret << std::endl;
            return -1;
        }
        ret = aclrtCreateStream(&stream);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtCreateStream failed: " << ret << std::endl;
            return -1;
        }
        return 0;
    }
    
    void finalize(int deviceId = 0) {
        if (stream) aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    }
};

class NPUTensor {
public:
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int64_t numElements = 0;
    int64_t byteSize = 0;
    void* devPtr = nullptr;
    aclTensor* tensor = nullptr;
    
    int create(const std::vector<int64_t>& sh) {
        shape = sh;
        numElements = 1;
        for (auto s : shape) numElements *= s;
        byteSize = numElements * sizeof(float);
        
        // Calculate strides
        strides.resize(shape.size());
        int64_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        
        // Allocate device memory
        aclError ret = aclrtMalloc(&devPtr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return -1;
        
        // Create tensor
        tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, 
                                  strides.data(), 0, ACL_FORMAT_ND,
                                  shape.data(), shape.size(), devPtr);
        return tensor ? 0 : -1;
    }
    
    int fromHost(const float* data) {
        return aclrtMemcpy(devPtr, byteSize, data, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    
    int toHost(float* data) {
        return aclrtMemcpy(data, byteSize, devPtr, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    
    void destroy() {
        if (tensor) { aclDestroyTensor(tensor); tensor = nullptr; }
        if (devPtr) { aclrtFree(devPtr); devPtr = nullptr; }
    }
};

// ReLU on NPU
int relu_npu(NPUTensor& input, NPUTensor& output, aclrtStream stream) {
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    
    aclError ret = aclnnReluGetWorkspaceSize(input.tensor, output.tensor, &wsSize, &executor);
    if (ret != ACL_SUCCESS) return ret;
    
    void* ws = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return ret;
    }
    
    ret = aclnnRelu(ws, wsSize, executor, stream);
    if (ws) aclrtFree(ws);
    
    return ret;
}

// Softmax on NPU
int softmax_npu(NPUTensor& input, NPUTensor& output, int64_t dim, aclrtStream stream) {
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    
    aclError ret = aclnnSoftmaxGetWorkspaceSize(input.tensor, dim, output.tensor, &wsSize, &executor);
    if (ret != ACL_SUCCESS) return ret;
    
    void* ws = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return ret;
    }
    
    ret = aclnnSoftmax(ws, wsSize, executor, stream);
    if (ws) aclrtFree(ws);
    
    return ret;
}

// Matrix multiplication on NPU
int matmul_npu(NPUTensor& a, NPUTensor& b, NPUTensor& output, aclrtStream stream) {
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    
    aclError ret = aclnnMatmulGetWorkspaceSize(a.tensor, b.tensor, output.tensor, 0, &wsSize, &executor);
    if (ret != ACL_SUCCESS) return ret;
    
    void* ws = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return ret;
    }
    
    ret = aclnnMatmul(ws, wsSize, executor, stream);
    if (ws) aclrtFree(ws);
    
    return ret;
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "   AlexNet FC Layers on NPU" << std::endl;
    std::cout << "   (True NPU Execution with aclnn)" << std::endl;
    std::cout << "============================================" << std::endl;
    
    NPUContext ctx;
    CHECK_RET(ctx.init(0) == 0, "NPU init failed");
    std::cout << "[NPU] Device initialized" << std::endl;
    
    // Simulate AlexNet FC layers
    // Input: flatten features after conv layers: [batch, 1024]
    // FC1: 1024 -> 256
    // FC2: 256 -> 10
    
    const int batch = 4;
    const int feat_in = 1024;
    const int hidden = 256;
    const int num_classes = 10;
    
    // Random input
    std::vector<float> h_input(batch * feat_in);
    std::vector<float> h_fc1_w(feat_in * hidden);  // [1024, 256]
    std::vector<float> h_fc2_w(hidden * num_classes);  // [256, 10]
    std::vector<float> h_output(batch * num_classes);
    
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (auto& v : h_input) v = dist(gen);
    for (auto& v : h_fc1_w) v = dist(gen);
    for (auto& v : h_fc2_w) v = dist(gen);
    
    std::cout << "\nInput shape: [" << batch << ", " << feat_in << "]" << std::endl;
    std::cout << "FC1: " << feat_in << " -> " << hidden << std::endl;
    std::cout << "FC2: " << hidden << " -> " << num_classes << std::endl;
    
    // Create tensors
    NPUTensor input, fc1_w, fc1_out, relu1_out;
    NPUTensor fc2_w, fc2_out, softmax_out;
    
    CHECK_RET(input.create({batch, feat_in}) == 0, "create input failed");
    CHECK_RET(fc1_w.create({feat_in, hidden}) == 0, "create fc1_w failed");
    CHECK_RET(fc1_out.create({batch, hidden}) == 0, "create fc1_out failed");
    CHECK_RET(relu1_out.create({batch, hidden}) == 0, "create relu1_out failed");
    CHECK_RET(fc2_w.create({hidden, num_classes}) == 0, "create fc2_w failed");
    CHECK_RET(fc2_out.create({batch, num_classes}) == 0, "create fc2_out failed");
    CHECK_RET(softmax_out.create({batch, num_classes}) == 0, "create softmax_out failed");
    
    // H2D
    std::cout << "\n[NPU] Transferring data to device..." << std::endl;
    CHECK_RET(input.fromHost(h_input.data()) == 0, "H2D input failed");
    CHECK_RET(fc1_w.fromHost(h_fc1_w.data()) == 0, "H2D fc1_w failed");
    CHECK_RET(fc2_w.fromHost(h_fc2_w.data()) == 0, "H2D fc2_w failed");
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // FC1: input @ fc1_w
    std::cout << "[NPU] FC1 (MatMul)..." << std::endl;
    aclError ret = matmul_npu(input, fc1_w, fc1_out, ctx.stream);
    CHECK_RET(ret == ACL_SUCCESS, "FC1 matmul failed");
    
    // ReLU
    std::cout << "[NPU] ReLU..." << std::endl;
    ret = relu_npu(fc1_out, relu1_out, ctx.stream);
    CHECK_RET(ret == ACL_SUCCESS, "ReLU failed");
    
    // FC2: relu1_out @ fc2_w
    std::cout << "[NPU] FC2 (MatMul)..." << std::endl;
    ret = matmul_npu(relu1_out, fc2_w, fc2_out, ctx.stream);
    CHECK_RET(ret == ACL_SUCCESS, "FC2 matmul failed");
    
    // Softmax
    std::cout << "[NPU] Softmax..." << std::endl;
    ret = softmax_npu(fc2_out, softmax_out, -1, ctx.stream);
    CHECK_RET(ret == ACL_SUCCESS, "Softmax failed");
    
    // Sync
    aclrtSynchronizeStream(ctx.stream);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // D2H
    std::cout << "[NPU] Transferring results to host..." << std::endl;
    CHECK_RET(softmax_out.toHost(h_output.data()) == 0, "D2H output failed");
    
    // Results
    std::cout << "\n============================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "NPU compute time: " << ms << " ms" << std::endl;
    std::cout << "Output shape: [" << batch << ", " << num_classes << "]" << std::endl;
    
    std::cout << "\nPredictions:" << std::endl;
    for (int b = 0; b < batch; b++) {
        float sum = 0;
        int maxIdx = 0;
        float maxVal = h_output[b * num_classes];
        for (int c = 0; c < num_classes; c++) {
            sum += h_output[b * num_classes + c];
            if (h_output[b * num_classes + c] > maxVal) {
                maxVal = h_output[b * num_classes + c];
                maxIdx = c;
            }
        }
        std::cout << "  Sample " << b << ": Class " << maxIdx 
                  << ", Confidence " << maxVal 
                  << ", Sum " << sum << std::endl;
    }
    
    // Cleanup
    input.destroy();
    fc1_w.destroy();
    fc1_out.destroy();
    relu1_out.destroy();
    fc2_w.destroy();
    fc2_out.destroy();
    softmax_out.destroy();
    ctx.finalize();
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "*** AlexNet NPU Execution SUCCESS! ***" << std::endl;
    std::cout << "============================================" << std::endl;
    
    return 0;
}
