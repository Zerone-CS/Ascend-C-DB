#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelSwiGLU {
public:
    __aicore__ inline KernelSwiGLU() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gate, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileLength) {
        this->totalLength = totalLength;
        this->tileLength = tileLength;
        this->tileNum = totalLength / tileLength;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        gateGm.SetGlobalBuffer((__gm__ float*)gate, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(inQueueGate, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer, tileLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileNum; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> gateLocal = inQueueGate.AllocTensor<float>();
        DataCopy(xLocal, xGm[idx * tileLength], tileLength);
        DataCopy(gateLocal, gateGm[idx * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueGate.EnQue(gateLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> gateLocal = inQueueGate.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
        
        // SwiGLU: y = x * gate * sigmoid(gate)
        // sigmoid(gate) = 1 / (1 + exp(-gate))
        
        // Step 1: -gate -> tmpLocal
        Muls(tmpLocal, gateLocal, (float)-1.0f, tileLength);
        // Step 2: exp(-gate) -> tmpLocal
        Exp(tmpLocal, tmpLocal, tileLength);
        // Step 3: 1 + exp(-gate) -> tmpLocal
        Adds(tmpLocal, tmpLocal, (float)1.0f, tileLength);
        // Step 4: gate / (1 + exp(-gate)) = gate * sigmoid(gate) -> yLocal
        Div(yLocal, gateLocal, tmpLocal, tileLength);
        // Step 5: x * (gate * sigmoid(gate)) -> yLocal
        Mul(yLocal, xLocal, yLocal, tileLength);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueGate.FreeTensor(gateLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[idx * tileLength], yLocal, tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueGate;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuffer;
    GlobalTensor<float> xGm, gateGm, yGm;
    uint32_t totalLength, tileLength, tileNum;
};

extern "C" __global__ __aicore__ void swi_glu_custom(GM_ADDR x, GM_ADDR gate, GM_ADDR y, 
                                                      GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSwiGLU op;
    op.Init(x, gate, y, tiling_data.totalLength, tiling_data.tileLength);
    op.Process();
}
