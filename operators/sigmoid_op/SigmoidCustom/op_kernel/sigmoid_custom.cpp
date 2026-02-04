#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_SIZE = 256;

class KernelSigmoid {
public:
    __aicore__ inline KernelSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = this->blockLength / TILE_SIZE;

        xGm.SetGlobalBuffer((__gm__ float*)x + GetBlockIdx() * this->blockLength, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + GetBlockIdx() * this->blockLength, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(tmpBuffer, TILE_SIZE * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t idx) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[idx * TILE_SIZE], TILE_SIZE);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t idx) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();

        Muls(tmpLocal, xLocal, -1.0f, TILE_SIZE);
        Exp(yLocal, tmpLocal, TILE_SIZE);
        Adds(yLocal, yLocal, 1.0f, TILE_SIZE);
        Reciprocal(yLocal, yLocal, TILE_SIZE);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t idx) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[idx * TILE_SIZE], yLocal, TILE_SIZE);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuffer;
    GlobalTensor<float> xGm, yGm;
    uint32_t blockLength, tileNum;
};

extern "C" __global__ __aicore__ void sigmoid_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSigmoid op;
    op.Init(x, y, tiling_data.size);
    op.Process();
}
