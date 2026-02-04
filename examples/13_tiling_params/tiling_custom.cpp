/**
 * 13_tiling_params - Tiling参数传递示例
 * 
 * 覆盖特性:
 * - Tiling结构体定义
 * - GM_ADDR tiling指针传递
 * - 动态分块策略
 * - 多维度形状处理
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// Tiling结构体 - Host侧计算，Kernel侧使用
struct TilingData {
    uint32_t totalLength;    // 总元素数
    uint32_t tileNum;        // 分块数
    uint32_t tileSize;       // 每块大小
    uint32_t lastTileSize;   // 最后一块大小
    uint32_t blockNum;       // 核数
    float scale;             // 计算参数
};

class KernelWithTiling {
public:
    __aicore__ inline KernelWithTiling() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGm) {
        // 从tiling GM地址读取参数
        auto* tiling = reinterpret_cast<__gm__ TilingData*>(tilingGm);
        this->totalLength = tiling->totalLength;
        this->tileNum = tiling->tileNum;
        this->tileSize = tiling->tileSize;
        this->lastTileSize = tiling->lastTileSize;
        this->scale = tiling->scale;
        
        // 多核分配
        uint32_t blockNum = tiling->blockNum;
        uint32_t blockIdx = GetBlockIdx();
        
        uint32_t tilesPerCore = (tileNum + blockNum - 1) / blockNum;
        this->startTile = blockIdx * tilesPerCore;
        this->endTile = (startTile + tilesPerCore > tileNum) ? tileNum : (startTile + tilesPerCore);
        
        if (startTile >= tileNum) {
            this->processTiles = 0;
            return;
        }
        this->processTiles = endTile - startTile;
        
        // 计算当前核的偏移
        uint32_t startOffset = startTile * tileSize;
        uint32_t processLength = (endTile < tileNum) ? 
                                  (processTiles * tileSize) :
                                  ((processTiles - 1) * tileSize + lastTileSize);
        
        xGm.SetGlobalBuffer((__gm__ float*)x + startOffset, processLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + startOffset, processLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (processTiles == 0) return;
        
        for (uint32_t i = 0; i < processTiles; i++) {
            uint32_t globalTileIdx = startTile + i;
            uint32_t currentLen = (globalTileIdx == tileNum - 1) ? lastTileSize : tileSize;
            
            CopyIn(i, currentLen);
            Compute(currentLen);
            CopyOut(i, currentLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t localIdx, uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[localIdx * tileSize], len);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        // 示例计算: y = x * scale
        Muls(yLocal, xLocal, scale, len);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t localIdx, uint32_t len) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[localIdx * tileSize], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm, yGm;
    uint32_t totalLength, tileNum, tileSize, lastTileSize;
    uint32_t startTile, endTile, processTiles;
    float scale;
};

extern "C" __global__ __aicore__ void tiling_custom(GM_ADDR x, GM_ADDR y, GM_ADDR tiling) {
    KernelWithTiling op;
    op.Init(x, y, tiling);
    op.Process();
}
