/**
 * ResNet-18 on Ascend NPU
 * 使用 aclnn API 实现完整 ResNet-18 推理
 * 输入: [1, 3, 224, 224] float32
 * 输出: [1, 1000] float32 (ImageNet 分类)
 */
#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include "acl/acl.h"
#include "aclnnop/aclnn_convolution.h"
#include "aclnnop/aclnn_batch_norm.h"
#include "aclnnop/aclnn_relu.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_max_pool.h"
#include "aclnnop/aclnn_adaptive_avg_pool2d.h"
#include "aclnnop/aclnn_matmul.h"

#define CHECK_ACL(ret, msg) do { \
    if ((ret) != ACL_SUCCESS) { \
        std::cerr << "[ERROR] " << (msg) << ", ret=" << (ret) << std::endl; \
        std::cerr << "  ACL: " << aclGetRecentErrMsg() << std::endl; \
        return -1; \
    } \
} while(0)

class NPUTensor {
public:
    std::vector<int64_t> shape;
    int64_t numElements = 0;
    int64_t byteSize = 0;
    void* devPtr = nullptr;
    aclTensor* tensor = nullptr;

    int create(const std::vector<int64_t>& sh, aclDataType dtype = ACL_FLOAT,
               aclFormat fmt = ACL_FORMAT_NCHW) {
        shape = sh;
        numElements = 1;
        for (auto s : shape) numElements *= s;
        int elemSize = (dtype == ACL_FLOAT) ? 4 : 2;
        byteSize = numElements * elemSize;

        std::vector<int64_t> strides(shape.size());
        int64_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }

        auto ret = aclrtMalloc(&devPtr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return ret;
        tensor = aclCreateTensor(shape.data(), shape.size(), dtype,
                                 strides.data(), 0, fmt,
                                 shape.data(), shape.size(), devPtr);
        return tensor ? 0 : -1;
    }

    int fromHost(const void* data) {
        return aclrtMemcpy(devPtr, byteSize, data, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    int toHost(void* data) {
        return aclrtMemcpy(data, byteSize, devPtr, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    void destroy() {
        if (tensor) { aclDestroyTensor(tensor); tensor = nullptr; }
        if (devPtr) { aclrtFree(devPtr); devPtr = nullptr; }
    }
};

aclIntArray* makeIntArray(const std::vector<int64_t>& v) {
    return aclCreateIntArray(v.data(), v.size());
}

int conv2d_npu(NPUTensor& input, NPUTensor& weight, NPUTensor& output,
               int stride, int pad, aclrtStream stream) {
    auto s = makeIntArray({stride, stride});
    auto p = makeIntArray({pad, pad, pad, pad});
    auto d = makeIntArray({1, 1});
    auto op = makeIntArray({0, 0});

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclError ret = aclnnConvolutionGetWorkspaceSize(
        input.tensor, weight.tensor, nullptr, s, p, d, false, op, 1,
        output.tensor, 0, &ws, &exec);
    aclDestroyIntArray(s);
    aclDestroyIntArray(p);
    aclDestroyIntArray(d);
    aclDestroyIntArray(op);
    if (ret != ACL_SUCCESS) return ret;

    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnConvolution(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

int batchnorm_npu(NPUTensor& input, NPUTensor& weight, NPUTensor& bias,
                  NPUTensor& runMean, NPUTensor& runVar,
                  NPUTensor& output, NPUTensor& saveMean, NPUTensor& saveInvstd,
                  aclrtStream stream) {
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclError ret = aclnnBatchNormGetWorkspaceSize(
        input.tensor, weight.tensor, bias.tensor,
        runMean.tensor, runVar.tensor,
        false, 0.1, 1e-5,
        output.tensor, saveMean.tensor, saveInvstd.tensor,
        &ws, &exec);
    if (ret != ACL_SUCCESS) return ret;

    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnBatchNorm(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

int relu_npu(NPUTensor& input, NPUTensor& output, aclrtStream stream) {
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclError ret = aclnnReluGetWorkspaceSize(input.tensor, output.tensor, &ws, &exec);
    if (ret != ACL_SUCCESS) return ret;
    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnRelu(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

int add_npu(NPUTensor& a, NPUTensor& b, NPUTensor& output, aclrtStream stream) {
    float alphaVal = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclError ret = aclnnAddGetWorkspaceSize(
        a.tensor, b.tensor, alpha, output.tensor, &ws, &exec);
    aclDestroyScalar(alpha);
    if (ret != ACL_SUCCESS) return ret;
    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnAdd(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

int maxpool_npu(NPUTensor& input, NPUTensor& output,
                int k, int s, int pad, aclrtStream stream) {
    auto ks = makeIntArray({k, k});
    auto ss = makeIntArray({s, s});
    auto ps = makeIntArray({pad, pad, pad, pad});
    auto ds = makeIntArray({1, 1});

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclError ret = aclnnMaxPoolGetWorkspaceSize(
        input.tensor, ks, ss, 0, ps, ds, 0, output.tensor, &ws, &exec);
    aclDestroyIntArray(ks);
    aclDestroyIntArray(ss);
    aclDestroyIntArray(ps);
    aclDestroyIntArray(ds);
    if (ret != ACL_SUCCESS) return ret;

    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnMaxPool(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

int avgpool_npu(NPUTensor& input, NPUTensor& output, aclrtStream stream) {
    auto outSize = makeIntArray({1, 1});
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclError ret = aclnnAdaptiveAvgPool2dGetWorkspaceSize(
        input.tensor, outSize, output.tensor, &ws, &exec);
    aclDestroyIntArray(outSize);
    if (ret != ACL_SUCCESS) return ret;

    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnAdaptiveAvgPool2d(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

int matmul_npu(NPUTensor& a, NPUTensor& b, NPUTensor& output, aclrtStream stream) {
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclError ret = aclnnMatmulGetWorkspaceSize(
        a.tensor, b.tensor, output.tensor, 0, &ws, &exec);
    if (ret != ACL_SUCCESS) return ret;
    void* workspace = nullptr;
    if (ws > 0) aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclnnMatmul(workspace, ws, exec, stream);
    if (workspace) aclrtFree(workspace);
    return ret;
}

struct BNParams {
    NPUTensor weight, bias, runMean, runVar, saveMean, saveInvstd;
    int init(int channels) {
        if (weight.create({channels})) return -1;
        if (bias.create({channels})) return -1;
        if (runMean.create({channels})) return -1;
        if (runVar.create({channels})) return -1;
        if (saveMean.create({channels})) return -1;
        if (saveInvstd.create({channels})) return -1;
        std::vector<float> ones(channels, 1.0f);
        std::vector<float> zeros(channels, 0.0f);
        weight.fromHost(ones.data());
        bias.fromHost(zeros.data());
        runMean.fromHost(zeros.data());
        runVar.fromHost(ones.data());
        return 0;
    }
    void destroy() {
        weight.destroy(); bias.destroy();
        runMean.destroy(); runVar.destroy();
        saveMean.destroy(); saveInvstd.destroy();
    }
};

struct ConvLayer {
    NPUTensor weight;
    int outC, kH, kW, stride, pad;

    int init(int inC, int outC_, int kSize, int stride_, int pad_) {
        outC = outC_; kH = kSize; kW = kSize;
        stride = stride_; pad = pad_;
        if (weight.create({outC, inC, kH, kW})) return -1;
        int n = weight.numElements;
        float stddev = std::sqrt(2.0f / (inC * kH * kW));
        std::vector<float> w(n);
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, stddev);
        for (int i = 0; i < n; i++) w[i] = dist(gen);
        weight.fromHost(w.data());
        return 0;
    }
    void destroy() { weight.destroy(); }
};

struct BasicBlock {
    ConvLayer conv1, conv2;
    BNParams bn1, bn2;
    ConvLayer downsampleConv;
    BNParams downsampleBN;
    bool hasDownsample;

    int init(int inC, int outC, int stride) {
        hasDownsample = (stride != 1 || inC != outC);
        if (conv1.init(inC, outC, 3, stride, 1)) return -1;
        if (bn1.init(outC)) return -1;
        if (conv2.init(outC, outC, 3, 1, 1)) return -1;
        if (bn2.init(outC)) return -1;
        if (hasDownsample) {
            if (downsampleConv.init(inC, outC, 1, stride, 0)) return -1;
            if (downsampleBN.init(outC)) return -1;
        }
        return 0;
    }

    int forward(NPUTensor& input, NPUTensor& output,
                int N, int inC, int inH, int inW, aclrtStream stream) {
        int outC = conv1.outC;
        int outH = (inH + 2 * conv1.pad - conv1.kH) / conv1.stride + 1;
        int outW = (inW + 2 * conv1.pad - conv1.kW) / conv1.stride + 1;

        NPUTensor c1out, bn1out, r1out;
        c1out.create({N, outC, outH, outW});
        bn1out.create({N, outC, outH, outW});
        r1out.create({N, outC, outH, outW});

        int ret = conv2d_npu(input, conv1.weight, c1out, conv1.stride, conv1.pad, stream);
        if (ret) { std::cerr << "BasicBlock conv1 failed" << std::endl; return ret; }
        ret = batchnorm_npu(c1out, bn1.weight, bn1.bias, bn1.runMean, bn1.runVar,
                            bn1out, bn1.saveMean, bn1.saveInvstd, stream);
        if (ret) { std::cerr << "BasicBlock bn1 failed" << std::endl; return ret; }
        ret = relu_npu(bn1out, r1out, stream);
        if (ret) { std::cerr << "BasicBlock relu1 failed" << std::endl; return ret; }

        NPUTensor c2out, bn2out;
        c2out.create({N, outC, outH, outW});
        bn2out.create({N, outC, outH, outW});

        ret = conv2d_npu(r1out, conv2.weight, c2out, conv2.stride, conv2.pad, stream);
        if (ret) { std::cerr << "BasicBlock conv2 failed" << std::endl; return ret; }
        ret = batchnorm_npu(c2out, bn2.weight, bn2.bias, bn2.runMean, bn2.runVar,
                            bn2out, bn2.saveMean, bn2.saveInvstd, stream);
        if (ret) { std::cerr << "BasicBlock bn2 failed" << std::endl; return ret; }

        NPUTensor identity;
        if (hasDownsample) {
            NPUTensor dsConvOut, dsBnOut;
            dsConvOut.create({N, outC, outH, outW});
            dsBnOut.create({N, outC, outH, outW});
            ret = conv2d_npu(input, downsampleConv.weight, dsConvOut,
                             downsampleConv.stride, downsampleConv.pad, stream);
            if (ret) { std::cerr << "BasicBlock downsample conv failed" << std::endl; return ret; }
            ret = batchnorm_npu(dsConvOut, downsampleBN.weight, downsampleBN.bias,
                                downsampleBN.runMean, downsampleBN.runVar,
                                dsBnOut, downsampleBN.saveMean, downsampleBN.saveInvstd, stream);
            if (ret) { std::cerr << "BasicBlock downsample bn failed" << std::endl; return ret; }
            identity = dsBnOut;
            dsConvOut.destroy();
        } else {
            identity = input;
        }

        NPUTensor addOut;
        addOut.create({N, outC, outH, outW});
        ret = add_npu(bn2out, identity, addOut, stream);
        if (ret) { std::cerr << "BasicBlock add failed" << std::endl; return ret; }

        output.create({N, outC, outH, outW});
        ret = relu_npu(addOut, output, stream);
        if (ret) { std::cerr << "BasicBlock relu2 failed" << std::endl; return ret; }

        c1out.destroy(); bn1out.destroy(); r1out.destroy();
        c2out.destroy(); bn2out.destroy();
        addOut.destroy();
        if (hasDownsample) identity.destroy();

        return 0;
    }

    void destroy() {
        conv1.destroy(); conv2.destroy();
        bn1.destroy(); bn2.destroy();
        if (hasDownsample) {
            downsampleConv.destroy();
            downsampleBN.destroy();
        }
    }
};

struct ResNet18 {
    ConvLayer conv1;
    BNParams bn1;
    BasicBlock layer1[2], layer2[2], layer3[2], layer4[2];
    NPUTensor fcWeight;
    int numClasses;

    int init(int nClasses = 1000) {
        numClasses = nClasses;
        if (conv1.init(3, 64, 7, 2, 3)) return -1;
        if (bn1.init(64)) return -1;

        if (layer1[0].init(64, 64, 1)) return -1;
        if (layer1[1].init(64, 64, 1)) return -1;

        if (layer2[0].init(64, 128, 2)) return -1;
        if (layer2[1].init(128, 128, 1)) return -1;

        if (layer3[0].init(128, 256, 2)) return -1;
        if (layer3[1].init(256, 256, 1)) return -1;

        if (layer4[0].init(256, 512, 2)) return -1;
        if (layer4[1].init(512, 512, 1)) return -1;

        if (fcWeight.create({512, numClasses})) return -1;
        int n = fcWeight.numElements;
        float stddev = 1.0f / std::sqrt(512.0f);
        std::vector<float> w(n);
        std::mt19937 gen(123);
        std::normal_distribution<float> dist(0.0f, stddev);
        for (int i = 0; i < n; i++) w[i] = dist(gen);
        fcWeight.fromHost(w.data());
        return 0;
    }

    int forward(NPUTensor& input, NPUTensor& output, aclrtStream stream) {
        int N = input.shape[0];
        int ret;

        NPUTensor c1out, bn1out, r1out, mpout;
        c1out.create({N, 64, 112, 112});
        bn1out.create({N, 64, 112, 112});
        r1out.create({N, 64, 112, 112});
        mpout.create({N, 64, 56, 56});

        std::cout << "  [conv1] 3->64, 7x7, stride=2 ..." << std::endl;
        ret = conv2d_npu(input, conv1.weight, c1out, 2, 3, stream);
        CHECK_ACL(ret, "conv1");

        ret = batchnorm_npu(c1out, bn1.weight, bn1.bias, bn1.runMean, bn1.runVar,
                            bn1out, bn1.saveMean, bn1.saveInvstd, stream);
        CHECK_ACL(ret, "bn1");

        ret = relu_npu(bn1out, r1out, stream);
        CHECK_ACL(ret, "relu1");

        ret = maxpool_npu(r1out, mpout, 3, 2, 1, stream);
        CHECK_ACL(ret, "maxpool");

        c1out.destroy(); bn1out.destroy(); r1out.destroy();

        std::cout << "  [layer1] 64->64, 2 blocks ..." << std::endl;
        NPUTensor l1b0out, l1b1out;
        ret = layer1[0].forward(mpout, l1b0out, N, 64, 56, 56, stream);
        CHECK_ACL(ret, "layer1[0]");
        mpout.destroy();

        ret = layer1[1].forward(l1b0out, l1b1out, N, 64, 56, 56, stream);
        CHECK_ACL(ret, "layer1[1]");
        l1b0out.destroy();

        std::cout << "  [layer2] 64->128, 2 blocks ..." << std::endl;
        NPUTensor l2b0out, l2b1out;
        ret = layer2[0].forward(l1b1out, l2b0out, N, 64, 56, 56, stream);
        CHECK_ACL(ret, "layer2[0]");
        l1b1out.destroy();

        ret = layer2[1].forward(l2b0out, l2b1out, N, 128, 28, 28, stream);
        CHECK_ACL(ret, "layer2[1]");
        l2b0out.destroy();

        std::cout << "  [layer3] 128->256, 2 blocks ..." << std::endl;
        NPUTensor l3b0out, l3b1out;
        ret = layer3[0].forward(l2b1out, l3b0out, N, 128, 28, 28, stream);
        CHECK_ACL(ret, "layer3[0]");
        l2b1out.destroy();

        ret = layer3[1].forward(l3b0out, l3b1out, N, 256, 14, 14, stream);
        CHECK_ACL(ret, "layer3[1]");
        l3b0out.destroy();

        std::cout << "  [layer4] 256->512, 2 blocks ..." << std::endl;
        NPUTensor l4b0out, l4b1out;
        ret = layer4[0].forward(l3b1out, l4b0out, N, 256, 14, 14, stream);
        CHECK_ACL(ret, "layer4[0]");
        l3b1out.destroy();

        ret = layer4[1].forward(l4b0out, l4b1out, N, 512, 7, 7, stream);
        CHECK_ACL(ret, "layer4[1]");
        l4b0out.destroy();

        std::cout << "  [avgpool] global ..." << std::endl;
        NPUTensor poolOut;
        poolOut.create({N, 512, 1, 1});
        ret = avgpool_npu(l4b1out, poolOut, stream);
        CHECK_ACL(ret, "avgpool");
        l4b1out.destroy();

        std::cout << "  [fc] 512->" << numClasses << " ..." << std::endl;
        NPUTensor flat;
        flat.create({N, 512});
        ret = aclrtMemcpy(flat.devPtr, flat.byteSize, poolOut.devPtr, flat.byteSize,
                          ACL_MEMCPY_DEVICE_TO_DEVICE);
        CHECK_ACL(ret, "flatten");
        poolOut.destroy();

        output.create({N, numClasses});
        ret = matmul_npu(flat, fcWeight, output, stream);
        CHECK_ACL(ret, "fc");
        flat.destroy();

        aclrtSynchronizeStream(stream);
        return 0;
    }

    void destroy() {
        conv1.destroy(); bn1.destroy();
        for (int i = 0; i < 2; i++) {
            layer1[i].destroy(); layer2[i].destroy();
            layer3[i].destroy(); layer4[i].destroy();
        }
        fcWeight.destroy();
    }
};

int main(int argc, char** argv) {
    int deviceId = 2;
    if (argc > 1) deviceId = atoi(argv[1]);

    std::cout << "=== ResNet-18 on Ascend NPU (910B) ==="<< std::endl;
    std::cout << "Using NPU device: " << deviceId << std::endl;

    aclError ret = aclInit(nullptr);
    CHECK_ACL(ret, "aclInit");
    ret = aclrtSetDevice(deviceId);
    CHECK_ACL(ret, "aclrtSetDevice");

    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    CHECK_ACL(ret, "aclrtCreateStream");

    std::cout << "\n[1/3] Initializing ResNet-18 weights ..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    ResNet18 model;
    if (model.init(1000) != 0) {
        std::cerr << "Failed to initialize model" << std::endl;
        return -1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double initMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  Model initialized in " << initMs << " ms" << std::endl;

    std::cout << "\n[2/3] Preparing input [1, 3, 224, 224] ..." << std::endl;
    NPUTensor input;
    input.create({1, 3, 224, 224});

    int inputSize = 1 * 3 * 224 * 224;
    std::vector<float> inputData(inputSize);
    std::mt19937 gen(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < inputSize; i++) inputData[i] = dist(gen);
    input.fromHost(inputData.data());

    std::cout << "\n[3/3] Running forward pass ..." << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();

    NPUTensor output;
    ret = model.forward(input, output, stream);
    if (ret != 0) {
        std::cerr << "Forward pass failed" << std::endl;
        model.destroy();
        input.destroy();
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return -1;
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    double fwdMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "\n  Forward pass completed in " << fwdMs << " ms" << std::endl;

    std::vector<float> result(1000);
    output.toHost(result.data());

    int topIdx = 0;
    float topVal = result[0];
    for (int i = 1; i < 1000; i++) {
        if (result[i] > topVal) {
            topVal = result[i];
            topIdx = i;
        }
    }

    std::cout << "\n=== Results ==" << std::endl;
    std::cout << "  Output shape: [1, 1000]" << std::endl;
    std::cout << "  Top prediction: class " << topIdx << " (score: " << topVal << ")" << std::endl;
    std::cout << "  First 10 logits: [";
    for (int i = 0; i < 10; i++) {
        std::cout << result[i];
        if (i < 9) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "\n=== Benchmark (5 runs) ===" << std::endl;
    double totalMs = 0;
    for (int run = 0; run < 5; run++) {
        NPUTensor benchOut;
        auto ts = std::chrono::high_resolution_clock::now();
        model.forward(input, benchOut, stream);
        auto te = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(te - ts).count();
        totalMs += ms;
        std::cout << "  Run " << (run + 1) << ": " << ms << " ms" << std::endl;
        benchOut.destroy();
    }
    std::cout << "  Average: " << (totalMs / 5.0) << " ms" << std::endl;

    output.destroy();
    input.destroy();
    model.destroy();
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "\nResNet-18 inference completed successfully!" << std::endl;
    return 0;
}
