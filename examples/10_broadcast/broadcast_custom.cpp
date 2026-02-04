/**
 * 10_broadcast - 广播操作示例
 * 
 * 覆盖特性:
 * - 标量广播到向量
 * - 行广播 (bias 加到每行)
 * - Duplicate 复制填充
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// 示例: 每行加上一个bias向量
class KernelBroadcastAdd {
public:
    __aicore__ inline KernelBroadcastAdd() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, 
                                 uint32_t totalRows, uint32_t rowSize) {
        this->totalRows = totalRows;
        this->rowSize = rowSize;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalRows * rowSize);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, rowSize);  // bias大小 = rowSize
        yGm.SetGlobalBuffer((__gm__ float*)y, totalRows * rowSize);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, rowSize * sizeof(float));
        pipe.InitBuffer(inQueueBias, 1, rowSize * sizeof(float));  // bias只读一次
        pipe.InitBuffer(outQueueY, BUFFER_NUM, rowSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        // 先读bias到本地
        LoadBias();
        
        // 处理每一行
        for (uint32_t i = 0; i < totalRows; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void LoadBias() {
        biasLocal = inQueueBias.AllocTensor<float>();
        DataCopy(biasLocal, biasGm, rowSize);
        inQueueBias.EnQue(biasLocal);
        biasLocal = inQueueBias.DeQue<float>();
    }

    __aicore__ inline void CopyIn(uint32_t row) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[row * rowSize], rowSize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        // 核心计算: y = x + bias (广播加法)
        Add(yLocal, xLocal, biasLocal, rowSize);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t row) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[row * rowSize], yLocal, rowSize);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueBias;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm, biasGm, yGm;
    LocalTensor<float> biasLocal;  // 缓存bias
    uint32_t totalRows, rowSize;
};

// 示例2: 标量广播填充
class KernelScalarBroadcast {
public:
    __aicore__ inline KernelScalarBroadcast() {}
    
    __aicore__ inline void Init(GM_ADDR y, uint32_t totalLength, float value) {
        this->totalLength = totalLength;
        this->value = value;
        this->tileSize = 256;
        this->tileNum = (totalLength + tileSize - 1) / tileSize;
        
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t len = (i == tileNum - 1 && totalLength % tileSize != 0) 
                          ? totalLength % tileSize : tileSize;
            Compute(len);
            CopyOut(i, len);
        }
    }

private:
    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        // 核心计算: 用标量值填充整个tensor
        // Duplicate(dst, scalar, count) - 复制标量到向量
        Duplicate(yLocal, value, len);
        
        outQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t len) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[idx * tileSize], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> yGm;
    uint32_t totalLength, tileSize, tileNum;
    float value;
};

extern "C" __global__ __aicore__ void broadcast_add(GM_ADDR x, GM_ADDR bias, GM_ADDR y,
                                                     uint32_t totalRows, uint32_t rowSize) {
    KernelBroadcastAdd op;
    op.Init(x, bias, y, totalRows, rowSize);
    op.Process();
}

extern "C" __global__ __aicore__ void scalar_broadcast(GM_ADDR y, uint32_t totalLength, float value) {
    KernelScalarBroadcast op;
    op.Init(y, totalLength, value);
    op.Process();
}
