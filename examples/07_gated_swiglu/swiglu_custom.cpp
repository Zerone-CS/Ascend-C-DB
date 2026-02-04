/**
 * 07_gated_swiglu - 门控激活示例 (SwiGLU)
 * 
 * 覆盖特性:
 * - 双输入处理 (x, gate)
 * - 多队列管理
 * - Div 除法运算
 * - 复杂计算流程
 * 
 * 算法: SwiGLU(x, gate) = x * gate * sigmoid(gate)
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelSwiGLU {
public:
    __aicore__ inline KernelSwiGLU() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gate, GM_ADDR y, uint32_t totalLength) {
        this->totalLength = totalLength;
        this->tileSize = 256;
        this->tileNum = (totalLength + tileSize - 1) / tileSize;

        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        gateGm.SetGlobalBuffer((__gm__ float*)gate, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);

        // 双输入队列
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(inQueueGate, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileSize * sizeof(float));
        // 中间计算缓冲区
        pipe.InitBuffer(tmpBuffer1, tileSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer2, tileSize * sizeof(float));
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
        LocalTensor<float> gateLocal = inQueueGate.AllocTensor<float>();
        DataCopy(xLocal, xGm[idx * tileSize], len);
        DataCopy(gateLocal, gateGm[idx * tileSize], len);
        inQueueX.EnQue(xLocal);
        inQueueGate.EnQue(gateLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> gateLocal = inQueueGate.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmp1 = tmpBuffer1.Get<float>();
        LocalTensor<float> tmp2 = tmpBuffer2.Get<float>();

        // SwiGLU: y = x * gate * sigmoid(gate)
        // sigmoid(gate) = 1 / (1 + exp(-gate))
        
        // Step 1: tmp1 = -gate
        Muls(tmp1, gateLocal, -1.0f, len);
        // Step 2: tmp1 = exp(-gate)
        Exp(tmp1, tmp1, len);
        // Step 3: tmp1 = 1 + exp(-gate)
        Adds(tmp1, tmp1, 1.0f, len);
        // Step 4: tmp2 = gate / (1 + exp(-gate)) = gate * sigmoid(gate)
        Div(tmp2, gateLocal, tmp1, len);
        // Step 5: y = x * (gate * sigmoid(gate))
        Mul(yLocal, xLocal, tmp2, len);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueGate.FreeTensor(gateLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t len) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[idx * tileSize], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueGate;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2;
    GlobalTensor<float> xGm, gateGm, yGm;
    uint32_t totalLength, tileSize, tileNum;
};

extern "C" __global__ __aicore__ void swiglu_custom(GM_ADDR x, GM_ADDR gate, GM_ADDR y, 
                                                     uint32_t totalLength) {
    KernelSwiGLU op;
    op.Init(x, gate, y, totalLength);
    op.Process();
}
