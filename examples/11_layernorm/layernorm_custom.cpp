/**
 * 11_layernorm - LayerNorm算子示例
 * 
 * 覆盖特性:
 * - 多步规约运算 (ReduceSum 求均值 + 方差)
 * - Sqrt / Rsqrt
 * - 仿射参数 (gamma, beta)
 * - 精度敏感的数值计算
 * 
 * 算法: y = gamma * (x - mean) / sqrt(var + eps) + beta
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
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalRows * rowSize);
        gammaGm.SetGlobalBuffer((__gm__ float*)gamma, rowSize);
        betaGm.SetGlobalBuffer((__gm__ float*)beta, rowSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalRows * rowSize);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, rowSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, rowSize * sizeof(float));
        pipe.InitBuffer(gammaBuffer, rowSize * sizeof(float));
        pipe.InitBuffer(betaBuffer, rowSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer1, rowSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer2, rowSize * sizeof(float));
        pipe.InitBuffer(workBuffer, rowSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        // 加载gamma和beta到本地
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
        DataCopy(gammaLocal, gammaGm, rowSize);
        DataCopy(betaLocal, betaGm, rowSize);
    }

    __aicore__ inline void CopyIn(uint32_t row) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[row * rowSize], rowSize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmp1 = tmpBuffer1.Get<float>();
        LocalTensor<float> tmp2 = tmpBuffer2.Get<float>();
        LocalTensor<float> work = workBuffer.Get<float>();
        
        // Step 1: mean = sum(x) / N
        ReduceSum(work, xLocal, work, rowSize);
        float meanVal = work.GetValue(0) * invRowSize;
        
        // Step 2: tmp1 = x - mean
        Adds(tmp1, xLocal, -meanVal, rowSize);
        
        // Step 3: tmp2 = (x - mean)^2
        Mul(tmp2, tmp1, tmp1, rowSize);
        
        // Step 4: var = sum((x - mean)^2) / N
        ReduceSum(work, tmp2, work, rowSize);
        float varVal = work.GetValue(0) * invRowSize;
        
        // Step 5: invStd = 1 / sqrt(var + eps)
        float invStd = 1.0f / (float)__builtin_sqrtf(varVal + eps);
        
        // Step 6: y = (x - mean) * invStd
        Muls(yLocal, tmp1, invStd, rowSize);
        
        // Step 7: y = y * gamma + beta
        Mul(yLocal, yLocal, gammaLocal, rowSize);
        Add(yLocal, yLocal, betaLocal, rowSize);
        
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
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> gammaBuffer, betaBuffer;
    TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, workBuffer;
    GlobalTensor<float> xGm, gammaGm, betaGm, yGm;
    LocalTensor<float> gammaLocal, betaLocal;
    uint32_t totalRows, rowSize;
    float eps, invRowSize;
};

extern "C" __global__ __aicore__ void layernorm_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, 
                                                        GM_ADDR y, uint32_t totalRows, 
                                                        uint32_t rowSize, float eps) {
    KernelLayerNorm op;
    op.Init(x, gamma, beta, y, totalRows, rowSize, eps);
    op.Process();
}
