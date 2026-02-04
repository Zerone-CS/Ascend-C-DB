/**
 * 03_activation_relu - 激活函数示例 (Relu)
 * 
 * 覆盖特性:
 * - 激活函数 API (Relu)
 * - 数据对齐处理 (8字节对齐)
 * - 多核并行
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelRelu {
public:
    __aicore__ inline KernelRelu() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        this->totalLength = totalLength;
        this->blockIdx = GetBlockIdx();
        
        uint32_t blockNum = GetBlockNum();
        uint32_t lengthPerCore = (totalLength + blockNum - 1) / blockNum;
        uint32_t startIdx = blockIdx * lengthPerCore;
        uint32_t endIdx = (startIdx + lengthPerCore > totalLength) ? totalLength : (startIdx + lengthPerCore);
        this->processLength = (startIdx >= totalLength) ? 0 : (endIdx - startIdx);
        
        xGm.SetGlobalBuffer((__gm__ float*)x + startIdx, processLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + startIdx, processLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, 256 * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, 256 * sizeof(float));
    }

    __aicore__ inline void Process() {
        uint32_t tileLength = 256;
        uint32_t loopCount = (processLength + tileLength - 1) / tileLength;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t currentLen = (i == loopCount - 1) ? (processLength - i * tileLength) : tileLength;
            if (currentLen == 0) continue;
            CopyIn(i * tileLength, currentLen);
            Compute(currentLen);
            CopyOut(i * tileLength, currentLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        // 8字节对齐 (float=4bytes, 所以是2个元素对齐)
        uint32_t alignedLen = ((len + 7) / 8) * 8;
        DataCopy(xLocal, xGm[offset], alignedLen);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        uint32_t alignedLen = ((len + 7) / 8) * 8;
        
        // 核心计算: y = max(0, x)
        Relu(yLocal, xLocal, alignedLen);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        uint32_t alignedLen = ((len + 7) / 8) * 8;
        DataCopy(yGm[offset], yLocal, alignedLen);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm, yGm;
    uint32_t totalLength, processLength, blockIdx;
};

extern "C" __global__ __aicore__ void relu_custom(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
    KernelRelu op;
    op.Init(x, y, totalLength);
    op.Process();
}
