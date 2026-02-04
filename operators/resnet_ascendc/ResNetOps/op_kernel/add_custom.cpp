/**
 * AscendC Add Kernel - 残差连接
 * y = x1 + x2
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_SIZE = 256;

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength) {
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t lengthPerBlock = (totalLength + blockNum - 1) / blockNum;
        uint32_t startOffset = blockIdx * lengthPerBlock;
        this->processLength = (startOffset + lengthPerBlock > totalLength)
                              ? (totalLength - startOffset) : lengthPerBlock;
        if (startOffset >= totalLength) this->processLength = 0;

        x1Gm.SetGlobalBuffer((__gm__ float*)x1 + startOffset, processLength);
        x2Gm.SetGlobalBuffer((__gm__ float*)x2 + startOffset, processLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + startOffset, processLength);

        pipe.InitBuffer(inQueueX1, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, TILE_SIZE * sizeof(float));
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
        LocalTensor<float> x1Local = inQueueX1.AllocTensor<float>();
        LocalTensor<float> x2Local = inQueueX2.AllocTensor<float>();
        uint32_t alignedLen = ((len + 7) / 8) * 8;
        DataCopy(x1Local, x1Gm[idx * TILE_SIZE], alignedLen);
        DataCopy(x2Local, x2Gm[idx * TILE_SIZE], alignedLen);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<float> x1Local = inQueueX1.DeQue<float>();
        LocalTensor<float> x2Local = inQueueX2.DeQue<float>();
        LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
        uint32_t alignedLen = ((len + 7) / 8) * 8;

        Add(yLocal, x1Local, x2Local, alignedLen);

        outQueue.EnQue(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t len) {
        LocalTensor<float> yLocal = outQueue.DeQue<float>();
        uint32_t alignedLen = ((len + 7) / 8) * 8;
        DataCopy(yGm[idx * TILE_SIZE], yLocal, alignedLen);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<float> x1Gm, x2Gm, yGm;
    uint32_t processLength;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelAdd op;
    op.Init(x1, x2, y, tilingData.totalLength);
    op.Process();
}
