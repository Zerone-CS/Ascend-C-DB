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

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(inQueueGate, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer1, tileSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer2, tileSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t currentTileLength = (i == tileNum - 1 && totalLength % tileSize != 0) 
                                          ? totalLength % tileSize : tileSize;
            CopyIn(i, currentTileLength);
            Compute(currentTileLength);
            CopyOut(i, currentTileLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx, uint32_t length) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> gateLocal = inQueueGate.AllocTensor<float>();
        DataCopy(xLocal, xGm[idx * tileSize], length);
        DataCopy(gateLocal, gateGm[idx * tileSize], length);
        inQueueX.EnQue(xLocal);
        inQueueGate.EnQue(gateLocal);
    }

    __aicore__ inline void Compute(uint32_t length) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> gateLocal = inQueueGate.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmpLocal1 = tmpBuffer1.Get<float>();
        LocalTensor<float> tmpLocal2 = tmpBuffer2.Get<float>();

        // SwiGLU: y = x * gate * sigmoid(gate)
        // sigmoid(gate) = 1 / (1 + exp(-gate))
        
        // Step 1: tmpLocal1 = -gate
        Muls(tmpLocal1, gateLocal, static_cast<float>(-1.0f), length);
        // Step 2: tmpLocal1 = exp(-gate)
        Exp(tmpLocal1, tmpLocal1, length);
        // Step 3: tmpLocal1 = 1 + exp(-gate)
        Adds(tmpLocal1, tmpLocal1, static_cast<float>(1.0f), length);
        // Step 4: tmpLocal2 = gate / (1 + exp(-gate)) = gate * sigmoid(gate)
        Div(tmpLocal2, gateLocal, tmpLocal1, length);
        // Step 5: yLocal = x * (gate * sigmoid(gate))
        Mul(yLocal, xLocal, tmpLocal2, length);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueGate.FreeTensor(gateLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx, uint32_t length) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[idx * tileSize], yLocal, length);
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

extern "C" __global__ __aicore__ void swiglu_custom(GM_ADDR x, GM_ADDR gate, GM_ADDR y, uint32_t totalLength) {
    KernelSwiGLU op;
    op.Init(x, gate, y, totalLength);
    op.Process();
}

void swiglu_custom_do(uint32_t blockDim, void* stream,
                      uint8_t* x, uint8_t* gate, uint8_t* y, uint32_t totalLength) {
    swiglu_custom<<<blockDim, nullptr, stream>>>(x, gate, y, totalLength);
}
