/**
 * GELU (Gaussian Error Linear Unit) - AscendC 实现
 * 
 * GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * 近似: GELU(x) ≈ x * sigmoid(1.702 * x)  (快速版本)
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        this->totalLength = totalLength;
        this->tileSize = 256;
        this->tileNum = (totalLength + tileSize - 1) / tileSize;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer, tileSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t offset = i * tileSize;
            uint32_t len = (offset + tileSize > totalLength) ? (totalLength - offset) : tileSize;
            uint32_t alignedLen = ((len + 7) / 8) * 8;
            CopyIn(i, alignedLen);
            Compute(len, alignedLen);
            CopyOut(i, alignedLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx, uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[idx * tileSize], len);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t actualLen, uint32_t alignedLen) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmp = tmpBuffer.Get<float>();
        
        // Fast GELU: y = x * sigmoid(1.702 * x)
        // sigmoid(z) = 1 / (1 + exp(-z))
        
        // Step 1: tmp = 1.702 * x
        Muls(tmp, xLocal, 1.702f, alignedLen);
        
        // Step 2: tmp = -1.702 * x
        Muls(tmp, tmp, -1.0f, alignedLen);
        
        // Step 3: tmp = exp(-1.702 * x)
        Exp(tmp, tmp, alignedLen);
        
        // Step 4: tmp = 1 + exp(-1.702 * x)
        Adds(tmp, tmp, 1.0f, alignedLen);
        
        // Step 5: tmp = 1 / (1 + exp(-1.702 * x)) = sigmoid(1.702 * x)
        Reciprocal(tmp, tmp, alignedLen);
        
        // Step 6: y = x * sigmoid(1.702 * x)
        Mul(yLocal, xLocal, tmp, alignedLen);
        
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
    TBuf<QuePosition::VECCALC> tmpBuffer;
    GlobalTensor<float> xGm, yGm;
    uint32_t totalLength, tileSize, tileNum;
};

extern "C" __global__ __aicore__ void gelu_custom(GM_ADDR x, GM_ADDR y, 
                                                   GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGelu op;
    op.Init(x, y, tiling_data.totalLength);
    op.Process();
}
