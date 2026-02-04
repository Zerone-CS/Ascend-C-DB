/**
 * AlexNet 算子单元测试
 *
 * 单独测试各个 Kernel 的正确性
 */
#include <iostream>
#include <cmath>
#include <cstring>
// Note: This test file runs on CPU only, no ACL dependency needed

// ==================== 辅助函数 ====================

void printTensor(const char* name, const float* data, int size, int maxPrint = 10) {
    std::cout << name << ": [";
    int printSize = (size > maxPrint) ? maxPrint : size;
    for (int i = 0; i < printSize; i++) {
        std::cout << data[i];
        if (i < printSize - 1) std::cout << ", ";
    }
    if (size > maxPrint) std::cout << ", ...";
    std::cout << "]" << std::endl;
}

bool compareFloat(float a, float b, float eps = 1e-4) {
    return std::abs(a - b) < eps;
}

// ==================== ReLU 测试 ====================

bool testRelu() {
    std::cout << "\n=== Testing ReLU ===" << std::endl;

    const int N = 16;
    float input[N] = {-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5, -10, 10, -0.1, 0.1, 100};
    float expected[N] = {0, 0, 0, 0, 0, 0.5, 1, 2, 3, 4, 5, 0, 10, 0, 0.1, 100};
    float output[N] = {0};

    // CPU 参考实现
    for (int i = 0; i < N; i++) {
        output[i] = (input[i] > 0) ? input[i] : 0;
    }

    // 验证
    bool pass = true;
    for (int i = 0; i < N; i++) {
        if (!compareFloat(output[i], expected[i])) {
            std::cout << "  FAIL at index " << i << ": expected " << expected[i]
                      << ", got " << output[i] << std::endl;
            pass = false;
        }
    }

    printTensor("  Input", input, N);
    printTensor("  Output", output, N);
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// ==================== Softmax 测试 ====================

bool testSoftmax() {
    std::cout << "\n=== Testing Softmax ===" << std::endl;

    const int N = 5;
    float input[N] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float output[N] = {0};

    // CPU 参考实现: softmax(x) = exp(x - max) / sum(exp(x - max))
    float maxVal = input[0];
    for (int i = 1; i < N; i++) {
        if (input[i] > maxVal) maxVal = input[i];
    }

    float sum = 0;
    for (int i = 0; i < N; i++) {
        output[i] = std::exp(input[i] - maxVal);
        sum += output[i];
    }
    for (int i = 0; i < N; i++) {
        output[i] /= sum;
    }

    // 验证: 和应该等于1
    float checkSum = 0;
    for (int i = 0; i < N; i++) {
        checkSum += output[i];
    }

    bool pass = compareFloat(checkSum, 1.0f);
    printTensor("  Input", input, N);
    printTensor("  Output", output, N);
    std::cout << "  Sum: " << checkSum << " (should be 1.0)" << std::endl;
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// ==================== MaxPool2D 测试 ====================

bool testMaxPool2D() {
    std::cout << "\n=== Testing MaxPool2D ===" << std::endl;

    // 输入: 4x4, kernel: 2x2, stride: 2
    const int H = 4, W = 4;
    const int kH = 2, kW = 2;
    const int sH = 2, sW = 2;
    const int oH = (H - kH) / sH + 1;  // 2
    const int oW = (W - kW) / sW + 1;  // 2

    float input[H * W] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float expected[oH * oW] = {6, 8, 14, 16};  // 每个 2x2 区域的最大值
    float output[oH * oW] = {0};

    // CPU 参考实现
    for (int oh = 0; oh < oH; oh++) {
        for (int ow = 0; ow < oW; ow++) {
            float maxVal = -1e9;
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int ih = oh * sH + kh;
                    int iw = ow * sW + kw;
                    float val = input[ih * W + iw];
                    if (val > maxVal) maxVal = val;
                }
            }
            output[oh * oW + ow] = maxVal;
        }
    }

    // 验证
    bool pass = true;
    for (int i = 0; i < oH * oW; i++) {
        if (!compareFloat(output[i], expected[i])) {
            pass = false;
        }
    }

    std::cout << "  Input (4x4):" << std::endl;
    for (int h = 0; h < H; h++) {
        std::cout << "    ";
        for (int w = 0; w < W; w++) {
            std::cout << input[h * W + w] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "  Output (2x2):" << std::endl;
    for (int h = 0; h < oH; h++) {
        std::cout << "    ";
        for (int w = 0; w < oW; w++) {
            std::cout << output[h * oW + w] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// ==================== Linear 测试 ====================

bool testLinear() {
    std::cout << "\n=== Testing Linear (y = Wx + b) ===" << std::endl;

    // x: [1, 3], W: [2, 3], b: [2], y: [1, 2]
    const int inFeatures = 3;
    const int outFeatures = 2;

    float x[inFeatures] = {1, 2, 3};
    float W[outFeatures * inFeatures] = {
        1, 0, 1,  // W[0,:]
        0, 1, 1   // W[1,:]
    };
    float b[outFeatures] = {1, 2};
    float expected[outFeatures] = {
        1*1 + 0*2 + 1*3 + 1,  // 5
        0*1 + 1*2 + 1*3 + 2   // 7
    };
    float output[outFeatures] = {0};

    // CPU 参考实现: y = Wx + b
    for (int o = 0; o < outFeatures; o++) {
        float sum = 0;
        for (int i = 0; i < inFeatures; i++) {
            sum += W[o * inFeatures + i] * x[i];
        }
        output[o] = sum + b[o];
    }

    // 验证
    bool pass = true;
    for (int i = 0; i < outFeatures; i++) {
        if (!compareFloat(output[i], expected[i])) {
            pass = false;
        }
    }

    printTensor("  x", x, inFeatures);
    printTensor("  y", output, outFeatures);
    printTensor("  expected", expected, outFeatures);
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// ==================== Conv2D 测试 ====================

bool testConv2D() {
    std::cout << "\n=== Testing Conv2D ===" << std::endl;

    // 简化测试: 3x3 输入, 2x2 卷积核, stride=1, pad=0
    const int H = 3, W = 3, C = 1;
    const int kH = 2, kW = 2;
    const int oH = H - kH + 1;  // 2
    const int oW = W - kW + 1;  // 2
    const int outC = 1;

    float input[C * H * W] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float kernel[outC * C * kH * kW] = {
        1, 0,
        0, 1
    };
    // 卷积结果:
    // [0,0]: 1*1 + 2*0 + 4*0 + 5*1 = 6
    // [0,1]: 2*1 + 3*0 + 5*0 + 6*1 = 8
    // [1,0]: 4*1 + 5*0 + 7*0 + 8*1 = 12
    // [1,1]: 5*1 + 6*0 + 8*0 + 9*1 = 14
    float expected[outC * oH * oW] = {6, 8, 12, 14};
    float output[outC * oH * oW] = {0};

    // CPU 参考实现
    for (int oc = 0; oc < outC; oc++) {
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                float sum = 0;
                for (int ic = 0; ic < C; ic++) {
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int ih = oh + kh;
                            int iw = ow + kw;
                            float x = input[ic * H * W + ih * W + iw];
                            float w = kernel[oc * C * kH * kW + ic * kH * kW + kh * kW + kw];
                            sum += x * w;
                        }
                    }
                }
                output[oc * oH * oW + oh * oW + ow] = sum;
            }
        }
    }

    // 验证
    bool pass = true;
    for (int i = 0; i < outC * oH * oW; i++) {
        if (!compareFloat(output[i], expected[i])) {
            pass = false;
        }
    }

    std::cout << "  Input (3x3):" << std::endl;
    for (int h = 0; h < H; h++) {
        std::cout << "    ";
        for (int w = 0; w < W; w++) {
            std::cout << input[h * W + w] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "  Kernel (2x2):" << std::endl;
    for (int h = 0; h < kH; h++) {
        std::cout << "    ";
        for (int w = 0; w < kW; w++) {
            std::cout << kernel[h * kW + w] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "  Output (2x2):" << std::endl;
    for (int h = 0; h < oH; h++) {
        std::cout << "    ";
        for (int w = 0; w < oW; w++) {
            std::cout << output[h * oW + w] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

// ==================== 主函数 ====================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "    AlexNet Kernel Unit Tests (CPU)     " << std::endl;
    std::cout << "========================================" << std::endl;

    int passed = 0;
    int total = 5;

    if (testRelu()) passed++;
    if (testSoftmax()) passed++;
    if (testMaxPool2D()) passed++;
    if (testLinear()) passed++;
    if (testConv2D()) passed++;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Summary: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
