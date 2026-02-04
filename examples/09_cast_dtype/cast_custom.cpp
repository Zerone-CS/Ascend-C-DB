/**
 * 09_cast_dtype - 数据类型转换示例 (Cast)
 * 
 * 覆盖特性:
 * - Cast 类型转换 API
 * - half (float16) 数据类型
 * - 混合精度计算
 * - RoundMode 舍入模式
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// 示例1: float32 -> float16
class KernelCastF32ToF16 {
public:
    __aicore__ inline KernelCastF32ToF16() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        this->totalLength = totalLength;
        this->tileSize = 256;
        this->tileNum = (totalLength + tileSize - 1) / tileSize;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ half*)y, totalLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileSize * sizeof(half));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t len = (i == tileNum - 1 && totalLength % tileSize != 0) 
                          ? totalLength % tileSize : tileSize;
            CopyIn(i, len);
            Compute(len);
            CopyOut(i, len);
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
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        
        // 核心计算: float32 -> float16
        // Cast(dst, src, RoundMode, count)
        Cast(yLocal, xLocal, RoundMode::CAST_ROUND, len);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t len) {
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        DataCopy(yGm[idx * tileSize], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm;
    GlobalTensor<half> yGm;
    uint32_t totalLength, tileSize, tileNum;
};

// 示例2: float16 -> float32
class KernelCastF16ToF32 {
public:
    __aicore__ inline KernelCastF16ToF32() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        this->totalLength = totalLength;
        this->tileSize = 256;
        this->tileNum = (totalLength + tileSize - 1) / tileSize;
        
        xGm.SetGlobalBuffer((__gm__ half*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileSize * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t len = (i == tileNum - 1 && totalLength % tileSize != 0) 
                          ? totalLength % tileSize : tileSize;
            CopyIn(i, len);
            Compute(len);
            CopyOut(i, len);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx, uint32_t len) {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopy(xLocal, xGm[idx * tileSize], len);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        // 核心计算: float16 -> float32
        Cast(yLocal, xLocal, RoundMode::CAST_NONE, len);
        
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
    GlobalTensor<half> xGm;
    GlobalTensor<float> yGm;
    uint32_t totalLength, tileSize, tileNum;
};

extern "C" __global__ __aicore__ void cast_f32_to_f16(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
    KernelCastF32ToF16 op;
    op.Init(x, y, totalLength);
    op.Process();
}

extern "C" __global__ __aicore__ void cast_f16_to_f32(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
    KernelCastF16ToF32 op;
    op.Init(x, y, totalLength);
    op.Process();
}
