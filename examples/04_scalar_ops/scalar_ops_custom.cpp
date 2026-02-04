/**
 * 04_scalar_ops - 标量运算示例 (Adds, Muls)
 * 
 * 覆盖特性:
 * - 标量加法 Adds(dst, src, scalar, count)
 * - 标量乘法 Muls(dst, src, scalar, count)
 * - 组合运算: y = (x + bias) * scale
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelScalarOps {
public:
    __aicore__ inline KernelScalarOps() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, float bias, float scale) {
        this->totalLength = totalLength;
        this->bias = bias;
        this->scale = scale;
        this->tileSize = 256;
        this->tileNum = (totalLength + tileSize - 1) / tileSize;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer, tileSize * sizeof(float));  // 中间缓冲区
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
        LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
        
        // 核心计算: y = (x + bias) * scale
        // Step 1: tmp = x + bias
        Adds(tmpLocal, xLocal, bias, len);
        // Step 2: y = tmp * scale
        Muls(yLocal, tmpLocal, scale, len);
        
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
    TBuf<QuePosition::VECCALC> tmpBuffer;  // TBuf用于中间计算
    GlobalTensor<float> xGm, yGm;
    uint32_t totalLength, tileSize, tileNum;
    float bias, scale;
};

extern "C" __global__ __aicore__ void scalar_ops_custom(GM_ADDR x, GM_ADDR y, 
                                                         uint32_t totalLength, 
                                                         float bias, float scale) {
    KernelScalarOps op;
    op.Init(x, y, totalLength, bias, scale);
    op.Process();
}
