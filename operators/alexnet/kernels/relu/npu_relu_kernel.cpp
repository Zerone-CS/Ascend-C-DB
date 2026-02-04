/**
 * ReLU Kernel for NPU - AlexNet 组件
 * 
 * 简化版本，直接传入参数
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_SIZE = 256;

class KernelRelu {
public:
    __aicore__ inline KernelRelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        this->totalLength = totalLength;
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_SIZE * sizeof(float));
    }

    __aicore__ inline void Process() {
        uint32_t loopCount = (totalLength + TILE_SIZE - 1) / TILE_SIZE;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t offset = i * TILE_SIZE;
            uint32_t len = (offset + TILE_SIZE > totalLength) ? (totalLength - offset) : TILE_SIZE;
            CopyIn(offset, len);
            Compute(len);
            CopyOut(offset, len);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        uint32_t alignedLen = ((len + 7) / 8) * 8;
        DataCopy(xLocal, xGm[offset], alignedLen);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        uint32_t alignedLen = ((len + 7) / 8) * 8;
        Relu(yLocal, xLocal, alignedLen);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        uint32_t alignedLen = ((len + 7) / 8) * 8;
        DataCopy(yGm[offset], yLocal, alignedLen);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm, yGm;
    uint32_t totalLength;
};

extern "C" __global__ __aicore__ void relu_npu(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
    KernelRelu op;
    op.Init(x, y, totalLength);
}
