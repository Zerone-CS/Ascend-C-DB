/**
 * Add (Residual Connection) - AscendC 实现
 * y = a + b  逐元素加法
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR y, uint32_t totalLength) {
        this->totalLength = totalLength;
        this->tileSize = 256;
        this->tileNum = (totalLength + tileSize - 1) / tileSize;
        
        aGm.SetGlobalBuffer((__gm__ float*)a, totalLength);
        bGm.SetGlobalBuffer((__gm__ float*)b, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        
        pipe.InitBuffer(inQueueA, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t offset = i * tileSize;
            uint32_t len = (offset + tileSize > totalLength) ? (totalLength - offset) : tileSize;
            uint32_t alignedLen = ((len + 7) / 8) * 8;
            CopyIn(i, alignedLen);
            Compute(alignedLen);
            CopyOut(i, alignedLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx, uint32_t len) {
        LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
        LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
        DataCopy(aLocal, aGm[idx * tileSize], len);
        DataCopy(bLocal, bGm[idx * tileSize], len);
        inQueueA.EnQue(aLocal);
        inQueueB.EnQue(bLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<float> aLocal = inQueueA.DeQue<float>();
        LocalTensor<float> bLocal = inQueueB.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        Add(yLocal, aLocal, bLocal, len);
        
        outQueueY.EnQue(yLocal);
        inQueueA.FreeTensor(aLocal);
        inQueueB.FreeTensor(bLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t len) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[idx * tileSize], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA, inQueueB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> aGm, bGm, yGm;
    uint32_t totalLength, tileSize, tileNum;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR a, GM_ADDR b, GM_ADDR y,
                                                  GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAdd op;
    op.Init(a, b, y, tiling_data.totalLength);
    op.Process();
}
