/**
 * AlexNet 推理网络 - AscendC 实现
 *
 * 网络结构:
 * =============================================
 * Layer          | Output Shape | Kernel/Stride
 * =============================================
 * Input          | 224x224x3    | -
 * Conv1 + ReLU   | 55x55x96     | 11x11, s=4, p=2
 * MaxPool1       | 27x27x96     | 3x3, s=2
 * Conv2 + ReLU   | 27x27x256    | 5x5, s=1, p=2
 * MaxPool2       | 13x13x256    | 3x3, s=2
 * Conv3 + ReLU   | 13x13x384    | 3x3, s=1, p=1
 * Conv4 + ReLU   | 13x13x384    | 3x3, s=1, p=1
 * Conv5 + ReLU   | 13x13x256    | 3x3, s=1, p=1
 * MaxPool3       | 6x6x256      | 3x3, s=2
 * Flatten        | 9216         | -
 * FC1 + ReLU     | 4096         | -
 * FC2 + ReLU     | 4096         | -
 * FC3            | 1000         | -
 * Softmax        | 1000         | -
 * =============================================
 *
 * Host 端网络推理流程演示
 */
#include <iostream>
#include <cstring>
#include <cstdlib>
#include "acl/acl.h"

// AlexNet 网络参数
namespace AlexNetConfig {
    // 输入尺寸
    constexpr int INPUT_H = 224;
    constexpr int INPUT_W = 224;
    constexpr int INPUT_C = 3;

    // Conv1: 3 -> 96, kernel=11, stride=4, pad=2
    constexpr int CONV1_OUT_C = 96;
    constexpr int CONV1_K = 11;
    constexpr int CONV1_S = 4;
    constexpr int CONV1_P = 2;
    constexpr int CONV1_OUT_H = (INPUT_H + 2*CONV1_P - CONV1_K) / CONV1_S + 1;  // 55
    constexpr int CONV1_OUT_W = (INPUT_W + 2*CONV1_P - CONV1_K) / CONV1_S + 1;  // 55

    // MaxPool1: 3x3, stride=2
    constexpr int POOL1_K = 3;
    constexpr int POOL1_S = 2;
    constexpr int POOL1_OUT_H = (CONV1_OUT_H - POOL1_K) / POOL1_S + 1;  // 27
    constexpr int POOL1_OUT_W = (CONV1_OUT_W - POOL1_K) / POOL1_S + 1;  // 27

    // Conv2: 96 -> 256, kernel=5, stride=1, pad=2
    constexpr int CONV2_OUT_C = 256;
    constexpr int CONV2_K = 5;
    constexpr int CONV2_S = 1;
    constexpr int CONV2_P = 2;
    constexpr int CONV2_OUT_H = (POOL1_OUT_H + 2*CONV2_P - CONV2_K) / CONV2_S + 1;  // 27
    constexpr int CONV2_OUT_W = (POOL1_OUT_W + 2*CONV2_P - CONV2_K) / CONV2_S + 1;  // 27

    // MaxPool2: 3x3, stride=2
    constexpr int POOL2_K = 3;
    constexpr int POOL2_S = 2;
    constexpr int POOL2_OUT_H = (CONV2_OUT_H - POOL2_K) / POOL2_S + 1;  // 13
    constexpr int POOL2_OUT_W = (CONV2_OUT_W - POOL2_K) / POOL2_S + 1;  // 13

    // Conv3: 256 -> 384, kernel=3, stride=1, pad=1
    constexpr int CONV3_OUT_C = 384;
    constexpr int CONV3_K = 3;
    constexpr int CONV3_S = 1;
    constexpr int CONV3_P = 1;
    constexpr int CONV3_OUT_H = (POOL2_OUT_H + 2*CONV3_P - CONV3_K) / CONV3_S + 1;  // 13
    constexpr int CONV3_OUT_W = (POOL2_OUT_W + 2*CONV3_P - CONV3_K) / CONV3_S + 1;  // 13

    // Conv4: 384 -> 384, kernel=3, stride=1, pad=1
    constexpr int CONV4_OUT_C = 384;
    constexpr int CONV4_K = 3;
    constexpr int CONV4_S = 1;
    constexpr int CONV4_P = 1;
    constexpr int CONV4_OUT_H = (CONV3_OUT_H + 2*CONV4_P - CONV4_K) / CONV4_S + 1;  // 13
    constexpr int CONV4_OUT_W = (CONV3_OUT_W + 2*CONV4_P - CONV4_K) / CONV4_S + 1;  // 13

    // Conv5: 384 -> 256, kernel=3, stride=1, pad=1
    constexpr int CONV5_OUT_C = 256;
    constexpr int CONV5_K = 3;
    constexpr int CONV5_S = 1;
    constexpr int CONV5_P = 1;
    constexpr int CONV5_OUT_H = (CONV4_OUT_H + 2*CONV5_P - CONV5_K) / CONV5_S + 1;  // 13
    constexpr int CONV5_OUT_W = (CONV4_OUT_W + 2*CONV5_P - CONV5_K) / CONV5_S + 1;  // 13

    // MaxPool3: 3x3, stride=2
    constexpr int POOL3_K = 3;
    constexpr int POOL3_S = 2;
    constexpr int POOL3_OUT_H = (CONV5_OUT_H - POOL3_K) / POOL3_S + 1;  // 6
    constexpr int POOL3_OUT_W = (CONV5_OUT_W - POOL3_K) / POOL3_S + 1;  // 6

    // Flatten size
    constexpr int FLATTEN_SIZE = CONV5_OUT_C * POOL3_OUT_H * POOL3_OUT_W;  // 256*6*6=9216

    // FC layers
    constexpr int FC1_OUT = 4096;
    constexpr int FC2_OUT = 4096;
    constexpr int NUM_CLASSES = 1000;
}

// 外部 kernel 声明
extern "C" void conv2d_kernel(void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" void maxpool2d_kernel(void*, void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                 uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" void relu_kernel(void*, void*, uint32_t);
extern "C" void linear_kernel(void*, void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" void softmax_kernel(void*, void*, uint32_t, uint32_t);

/**
 * AlexNet 推理类
 *
 * 使用方法:
 * 1. 创建 AlexNetInference 对象
 * 2. 调用 Init() 初始化
 * 3. 调用 LoadWeights() 加载预训练权重
 * 4. 调用 Forward() 执行推理
 * 5. 调用 Cleanup() 释放资源
 */
class AlexNetInference {
public:
    AlexNetInference() : initialized_(false) {}

    bool Init(int deviceId = 0) {
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            std::cerr << "ACL init failed: " << ret << std::endl;
            return false;
        }

        ret = aclrtSetDevice(deviceId);
        if (ret != ACL_SUCCESS) {
            std::cerr << "Set device failed: " << ret << std::endl;
            return false;
        }

        ret = aclrtCreateStream(&stream_);
        if (ret != ACL_SUCCESS) {
            std::cerr << "Create stream failed: " << ret << std::endl;
            return false;
        }

        // 分配中间缓冲区
        if (!AllocateBuffers()) {
            return false;
        }

        initialized_ = true;
        return true;
    }

    bool LoadWeights(const char* weightPath) {
        // TODO: 从文件加载预训练权重
        // 格式: 二进制文件，按层序存储
        std::cout << "Loading weights from: " << weightPath << std::endl;
        return true;
    }

    /**
     * 前向推理
     * @param input  输入图像 [1, 3, 224, 224] NCHW
     * @param output 输出概率 [1, 1000]
     */
    bool Forward(const float* input, float* output) {
        if (!initialized_) {
            std::cerr << "Not initialized!" << std::endl;
            return false;
        }

        using namespace AlexNetConfig;

        // 拷贝输入到设备
        size_t inputSize = INPUT_C * INPUT_H * INPUT_W * sizeof(float);
        aclrtMemcpy(inputBuffer_, inputSize, input, inputSize, ACL_MEMCPY_HOST_TO_DEVICE);

        // ========== Conv1 + ReLU + Pool1 ==========
        // conv2d_kernel(inputBuffer_, conv1Weight_, conv1Out_, 1, INPUT_C, INPUT_H, INPUT_W,
        //               CONV1_OUT_C, CONV1_K, CONV1_K, CONV1_S, CONV1_S, CONV1_P, CONV1_P);
        // relu_kernel(conv1Out_, conv1Out_, CONV1_OUT_C * CONV1_OUT_H * CONV1_OUT_W);
        // maxpool2d_kernel(conv1Out_, pool1Out_, 1, CONV1_OUT_C, CONV1_OUT_H, CONV1_OUT_W,
        //                  POOL1_K, POOL1_K, POOL1_S, POOL1_S);

        // ... 其他层类似 ...

        // 拷贝输出到 Host
        size_t outputSize = NUM_CLASSES * sizeof(float);
        aclrtMemcpy(output, outputSize, outputBuffer_, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

        return true;
    }

    void Cleanup() {
        if (!initialized_) return;

        // 释放缓冲区
        FreeBuffers();

        aclrtDestroyStream(stream_);
        aclrtResetDevice(0);
        aclFinalize();

        initialized_ = false;
    }

private:
    bool AllocateBuffers() {
        using namespace AlexNetConfig;

        // 输入缓冲区
        size_t inputSize = INPUT_C * INPUT_H * INPUT_W * sizeof(float);
        aclrtMalloc(&inputBuffer_, inputSize, ACL_MEM_MALLOC_NORMAL_ONLY);

        // 中间层缓冲区 (可以复用)
        size_t maxIntermediateSize = CONV1_OUT_C * CONV1_OUT_H * CONV1_OUT_W * sizeof(float);
        aclrtMalloc(&conv1Out_, maxIntermediateSize, ACL_MEM_MALLOC_NORMAL_ONLY);

        size_t pool1Size = CONV1_OUT_C * POOL1_OUT_H * POOL1_OUT_W * sizeof(float);
        aclrtMalloc(&pool1Out_, pool1Size, ACL_MEM_MALLOC_NORMAL_ONLY);

        // ... 其他层缓冲区 ...

        // 输出缓冲区
        size_t outputSize = NUM_CLASSES * sizeof(float);
        aclrtMalloc(&outputBuffer_, outputSize, ACL_MEM_MALLOC_NORMAL_ONLY);

        // 权重缓冲区
        size_t conv1WeightSize = CONV1_OUT_C * INPUT_C * CONV1_K * CONV1_K * sizeof(float);
        aclrtMalloc(&conv1Weight_, conv1WeightSize, ACL_MEM_MALLOC_NORMAL_ONLY);

        // ... 其他层权重 ...

        return true;
    }

    void FreeBuffers() {
        if (inputBuffer_) aclrtFree(inputBuffer_);
        if (conv1Out_) aclrtFree(conv1Out_);
        if (pool1Out_) aclrtFree(pool1Out_);
        if (outputBuffer_) aclrtFree(outputBuffer_);
        if (conv1Weight_) aclrtFree(conv1Weight_);
        // ... 释放其他缓冲区 ...
    }

private:
    bool initialized_;
    aclrtStream stream_;

    // 缓冲区指针
    void* inputBuffer_ = nullptr;
    void* conv1Out_ = nullptr;
    void* pool1Out_ = nullptr;
    void* outputBuffer_ = nullptr;

    // 权重指针
    void* conv1Weight_ = nullptr;
    // ... 其他层权重 ...
};

// 主函数示例
int main() {
    AlexNetInference alexnet;

    // 初始化
    if (!alexnet.Init(0)) {
        std::cerr << "Failed to initialize AlexNet" << std::endl;
        return -1;
    }

    // 加载权重
    // alexnet.LoadWeights("alexnet_weights.bin");

    // 准备输入 (224x224x3 RGB图像)
    const int inputSize = 3 * 224 * 224;
    float* input = new float[inputSize];
    // TODO: 加载并预处理图像

    // 推理
    float output[1000];
    // alexnet.Forward(input, output);

    // 找到最大概率类别
    int maxClass = 0;
    float maxProb = output[0];
    for (int i = 1; i < 1000; i++) {
        if (output[i] > maxProb) {
            maxProb = output[i];
            maxClass = i;
        }
    }
    std::cout << "Predicted class: " << maxClass << " with probability: " << maxProb << std::endl;

    // 清理
    delete[] input;
    alexnet.Cleanup();

    return 0;
}
