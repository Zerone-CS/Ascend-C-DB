/**
 * AscendC ReLU Kernel
 * y = max(0, x)
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_SIZE = 256;

class KernelRelu {
public:
    __aicore__ inline KernelRelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t lengthPerBlock = (totalLength + blockNum - 1) / blockNum;
        uint32_t startOffset = blockIdx * lengthPerBlock;
        this->processLength = (startOffset + lengthPerBlock > totalLength) 
                              ? (totalLength - startOffset) : lengthPerBlock;
        if (startOffset >= totalLength) this->processLength = 0;

        xGm.SetGlobalBuffer((__gm__ float*)x + startOffset, processLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + startOffset, processLength);

        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_SIZE * sizeof(float));
    }

    __aicore__ inline void Process() {
        uint32_t loopCount = (processLength + TILE_SIZE - 1) / TILE_SIZE;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t len = (i == loopCount - 1) ? (processLength - i * TILE_SIZE) : TILE_SIZE;
            CopyIn(i, len);
            Compute(len);
            CopyOut(i, len);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx, uint32_t len) {
        LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        uint32_t alignedLen = ((len + 7) / 8) * 8;
        DataCopy(xLocal, xGm[idx * TILE_SIZE], alignedLen);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<float> xLocal = inQueue.DeQue<float>();
        LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
        uint32_t alignedLen = ((len + 7) / 8) * 8;
        
        // AscendC ReLU API: y = max(0, x)
        Relu(yLocal, xLocal, alignedLen);
        
        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t len) {
        LocalTensor<float> yLocal = outQueue.DeQue<float>();
        uint32_t alignedLen = ((len + 7) / 8) * 8;
        DataCopy(yGm[idx * TILE_SIZE], yLocal, alignedLen);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<float> xGm, yGm;
    uint32_t processLength;
};

extern "C" __global__ __aicore__ void relu_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelRelu op;
    op.Init(x, y, tilingData.totalLength);
    op.Process();
}
