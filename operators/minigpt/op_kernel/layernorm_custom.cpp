/**
 * LayerNorm - AscendC 实现
 * y = gamma * (x - mean) / sqrt(var + eps) + beta
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelLayerNorm {
public:
    __aicore__ inline KernelLayerNorm() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                 uint32_t totalRows, uint32_t rowSize, float eps) {
        this->totalRows = totalRows;
        this->rowSize = rowSize;
        this->eps = eps;
        this->invRowSize = 1.0f / (float)rowSize;
        uint32_t alignedRowSize = ((rowSize + 7) / 8) * 8;
        this->alignedRowSize = alignedRowSize;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalRows * rowSize);
        gammaGm.SetGlobalBuffer((__gm__ float*)gamma, rowSize);
        betaGm.SetGlobalBuffer((__gm__ float*)beta, rowSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalRows * rowSize);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, alignedRowSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, alignedRowSize * sizeof(float));
        pipe.InitBuffer(gammaBuffer, alignedRowSize * sizeof(float));
        pipe.InitBuffer(betaBuffer, alignedRowSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer1, alignedRowSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer2, alignedRowSize * sizeof(float));
        pipe.InitBuffer(workBuffer, alignedRowSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        LoadParams();
        for (uint32_t i = 0; i < totalRows; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void LoadParams() {
        gammaLocal = gammaBuffer.Get<float>();
        betaLocal = betaBuffer.Get<float>();
        DataCopy(gammaLocal, gammaGm, alignedRowSize);
        DataCopy(betaLocal, betaGm, alignedRowSize);
    }

    __aicore__ inline void CopyIn(uint32_t row) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[row * rowSize], alignedRowSize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmp1 = tmpBuffer1.Get<float>();
        LocalTensor<float> tmp2 = tmpBuffer2.Get<float>();
        LocalTensor<float> work = workBuffer.Get<float>();
        
        ReduceSum(work, xLocal, work, rowSize);
        float meanVal = work.GetValue(0) * invRowSize;
        
        Adds(tmp1, xLocal, -meanVal, alignedRowSize);
        Mul(tmp2, tmp1, tmp1, alignedRowSize);
        
        ReduceSum(work, tmp2, work, rowSize);
        float varVal = work.GetValue(0) * invRowSize;
        float invStd = 1.0f / (float)__builtin_sqrtf(varVal + eps);
        
        Muls(yLocal, tmp1, invStd, alignedRowSize);
        Mul(yLocal, yLocal, gammaLocal, alignedRowSize);
        Add(yLocal, yLocal, betaLocal, alignedRowSize);
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t row) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[row * rowSize], yLocal, alignedRowSize);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> gammaBuffer, betaBuffer;
    TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, workBuffer;
    GlobalTensor<float> xGm, gammaGm, betaGm, yGm;
    LocalTensor<float> gammaLocal, betaLocal;
    uint32_t totalRows, rowSize, alignedRowSize;
    float eps, invRowSize;
};

extern "C" __global__ __aicore__ void layernorm_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta,
                                                        GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLayerNorm op;
    op.Init(x, gamma, beta, y, tiling_data.totalRows, tiling_data.rowSize, tiling_data.eps);
    op.Process();
}
