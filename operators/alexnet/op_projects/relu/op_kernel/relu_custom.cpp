#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelRelu {
public:
    __aicore__ inline KernelRelu() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = this->blockLength / 256;  // 256 floats per tile
        
        xGm.SetGlobalBuffer((__gm__ float*)x + GetBlockIdx() * this->blockLength, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + GetBlockIdx() * this->blockLength, this->blockLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, 256 * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, 256 * sizeof(float));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * 256], 256);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        Relu(yLocal, xLocal, 256);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[progress * 256], yLocal, 256);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm, yGm;
    uint32_t blockLength, tileNum;
};

extern "C" __global__ __aicore__ void relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelRelu op;
    op.Init(x, y, tiling_data.size);
    op.Process();
}
