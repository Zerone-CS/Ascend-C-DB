/**
 * 01_unary_exp - 一元算子示例 (Exp)
 * 
 * 覆盖特性:
 * - 基础一元运算 API
 * - TQue 队列操作
 * - Double Buffer 机制
 * - DataCopy 数据搬运
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;  // Double Buffer

class KernelExp {
public:
    __aicore__ inline KernelExp() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        this->totalLength = totalLength;
        this->tileSize = 256;  // 每次处理256个元素
        this->tileNum = (totalLength + tileSize - 1) / tileSize;
        
        // 设置全局内存指针
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        
        // 初始化队列缓冲区
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t currentLen = (i == tileNum - 1 && totalLength % tileSize != 0) 
                                  ? totalLength % tileSize : tileSize;
            CopyIn(i, currentLen);
            Compute(currentLen);
            CopyOut(i, currentLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx, uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[idx * tileSize], len);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        // 核心计算: y = exp(x)
        Exp(yLocal, xLocal, len);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t len) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[idx * tileSize], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm, yGm;
    uint32_t totalLength, tileSize, tileNum;
};

extern "C" __global__ __aicore__ void exp_custom(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
    KernelExp op;
    op.Init(x, y, totalLength);
    op.Process();
}
