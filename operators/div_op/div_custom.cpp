#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_SIZE = 256;

class KernelDiv {
public:
    __aicore__ inline KernelDiv() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = blockLength / TILE_SIZE;
        
        uint32_t offset = GetBlockIdx() * blockLength;
        xGm.SetGlobalBuffer((__gm__ float*)x + offset, blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + offset, blockLength);
        zGm.SetGlobalBuffer((__gm__ float*)z + offset, blockLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(tmpBuffer, TILE_SIZE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        DataCopy(xLocal, xGm[idx * TILE_SIZE], TILE_SIZE);
        DataCopy(yLocal, yGm[idx * TILE_SIZE], TILE_SIZE);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
        
        Reciprocal(tmpLocal, yLocal, TILE_SIZE);
        Mul(zLocal, xLocal, tmpLocal, TILE_SIZE);
        
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx)
    {
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        DataCopy(zGm[idx * TILE_SIZE], zLocal, TILE_SIZE);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TBuf<QuePosition::VECCALC> tmpBuffer;
    GlobalTensor<float> xGm, yGm, zGm;
    uint32_t blockLength, tileNum;
};

extern "C" __global__ __aicore__ void div_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
{
    KernelDiv op;
    op.Init(x, y, z, totalLength);
    op.Process();
}

void div_custom_do(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* z, uint32_t totalLength)
{
    div_custom<<<blockDim, nullptr, stream>>>(x, y, z, totalLength);
}
