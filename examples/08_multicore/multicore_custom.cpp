/**
 * 08_multicore - 多核并行示例
 * 
 * 覆盖特性:
 * - GetBlockIdx() 获取当前核ID
 * - GetBlockNum() 获取总核数
 * - 负载均衡策略
 * - 分核处理边界检查
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelMultiCore {
public:
    __aicore__ inline KernelMultiCore() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        // 获取多核信息
        this->blockIdx = GetBlockIdx();      // 当前核索引
        uint32_t blockNum = GetBlockNum();   // 总核数
        
        // 方案A: 均匀分配 + 余数处理
        uint32_t baseLength = totalLength / blockNum;
        uint32_t remainder = totalLength % blockNum;
        
        // 前remainder个核多处理一个元素
        if (blockIdx < remainder) {
            this->processLength = baseLength + 1;
            this->startOffset = blockIdx * (baseLength + 1);
        } else {
            this->processLength = baseLength;
            this->startOffset = remainder * (baseLength + 1) + (blockIdx - remainder) * baseLength;
        }
        
        // 边界检查
        if (startOffset >= totalLength) {
            this->processLength = 0;
            return;
        }
        
        xGm.SetGlobalBuffer((__gm__ float*)x + startOffset, processLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + startOffset, processLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, 256 * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, 256 * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (processLength == 0) return;  // 无任务核直接返回
        
        uint32_t tileSize = 256;
        uint32_t tileNum = (processLength + tileSize - 1) / tileSize;
        
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t offset = i * tileSize;
            uint32_t len = (i == tileNum - 1) ? (processLength - offset) : tileSize;
            
            CopyIn(offset, len);
            Compute(len);
            CopyOut(offset, len);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[offset], len);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        // 示例计算: y = x * 2
        Muls(yLocal, xLocal, 2.0f, len);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[offset], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm, yGm;
    uint32_t processLength, startOffset, blockIdx;
};

extern "C" __global__ __aicore__ void multicore_custom(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
    KernelMultiCore op;
    op.Init(x, y, totalLength);
    op.Process();
}
