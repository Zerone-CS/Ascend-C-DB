/**
 * 02_binary_add - 二元算子示例 (Add)
 * 
 * 覆盖特性:
 * - 二元运算 API (Add)
 * - 多输入队列管理
 * - 多核并行 (GetBlockIdx/GetBlockNum)
 * - 负载均衡
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength) {
        this->totalLength = totalLength;
        this->blockIdx = GetBlockIdx();
        
        // 多核负载均衡
        uint32_t blockNum = GetBlockNum();
        uint32_t blockLength = totalLength / blockNum;
        uint32_t blockTail = totalLength % blockNum;
        
        if (blockIdx < blockTail) {
            this->tileLength = blockLength + 1;
            this->startOffset = blockIdx * this->tileLength;
        } else {
            this->tileLength = blockLength;
            this->startOffset = blockIdx * blockLength + blockTail;
        }
        
        xGm.SetGlobalBuffer((__gm__ float*)x + startOffset, tileLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + startOffset, tileLength);
        zGm.SetGlobalBuffer((__gm__ float*)z + startOffset, tileLength);
        
        this->tileSize = 256;
        this->tileNum = (tileLength + tileSize - 1) / tileSize;
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t currentLen = (i == tileNum - 1 && tileLength % tileSize != 0) 
                                  ? tileLength % tileSize : tileSize;
            CopyIn(i, currentLen);
            Compute(currentLen);
            CopyOut(i, currentLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx, uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        
        DataCopy(xLocal, xGm[idx * tileSize], len);
        DataCopy(yLocal, yGm[idx * tileSize], len);
        
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        
        // 核心计算: z = x + y
        Add(zLocal, xLocal, yLocal, len);
        
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t len) {
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        DataCopy(zGm[idx * tileSize], zLocal, len);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<float> xGm, yGm, zGm;
    uint32_t totalLength, tileLength, tileSize, tileNum, startOffset, blockIdx;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength) {
    KernelAdd op;
    op.Init(x, y, z, totalLength);
    op.Process();
}
