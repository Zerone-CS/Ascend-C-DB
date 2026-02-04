#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_SIZE = 256;

class KernelDiv {
public:
    __aicore__ inline KernelDiv() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength) {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = this->blockLength / TILE_SIZE;

        xGm.SetGlobalBuffer((__gm__ float*)x + GetBlockIdx() * this->blockLength, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + GetBlockIdx() * this->blockLength, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float*)z + GetBlockIdx() * this->blockLength, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_SIZE * sizeof(float));
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
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        DataCopy(xLocal, xGm[idx * TILE_SIZE], TILE_SIZE);
        DataCopy(yLocal, yGm[idx * TILE_SIZE], TILE_SIZE);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t idx) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();

        Div(zLocal, xLocal, yLocal, TILE_SIZE);

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t idx) {
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        DataCopy(zGm[idx * TILE_SIZE], zLocal, TILE_SIZE);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<float> xGm, yGm, zGm;
    uint32_t blockLength, tileNum;
};

extern "C" __global__ __aicore__ void div_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelDiv op;
    op.Init(x, y, z, tiling_data.size);
    op.Process();
}
