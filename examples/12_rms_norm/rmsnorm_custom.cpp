/**
 * 12_rms_norm - RMSNorm算子示例 (LLaMA常用)
 * 
 * 覆盖特性:
 * - Rsqrt 倒数平方根
 * - 简化的归一化计算
 * - 无需均值计算
 * 
 * 算法: y = x * rsqrt(mean(x^2) + eps) * weight
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelRMSNorm {
public:
    __aicore__ inline KernelRMSNorm() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                                 uint32_t totalRows, uint32_t rowSize, float eps) {
        this->totalRows = totalRows;
        this->rowSize = rowSize;
        this->eps = eps;
        this->invRowSize = 1.0f / (float)rowSize;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalRows * rowSize);
        weightGm.SetGlobalBuffer((__gm__ float*)weight, rowSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalRows * rowSize);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, rowSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, rowSize * sizeof(float));
        pipe.InitBuffer(weightBuffer, rowSize * sizeof(float));
        pipe.InitBuffer(tmpBuffer, rowSize * sizeof(float));
        pipe.InitBuffer(workBuffer, rowSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        // 加载weight
        weightLocal = weightBuffer.Get<float>();
        DataCopy(weightLocal, weightGm, rowSize);
        
        for (uint32_t i = 0; i < totalRows; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t row) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[row * rowSize], rowSize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmp = tmpBuffer.Get<float>();
        LocalTensor<float> work = workBuffer.Get<float>();
        
        // Step 1: tmp = x^2
        Mul(tmp, xLocal, xLocal, rowSize);
        
        // Step 2: rms2 = mean(x^2) = sum(x^2) / N
        ReduceSum(work, tmp, work, rowSize);
        float rms2 = work.GetValue(0) * invRowSize;
        
        // Step 3: rsqrt_rms = 1 / sqrt(rms2 + eps)
        float rsqrt_rms = 1.0f / (float)__builtin_sqrtf(rms2 + eps);
        
        // Step 4: y = x * rsqrt_rms
        Muls(yLocal, xLocal, rsqrt_rms, rowSize);
        
        // Step 5: y = y * weight
        Mul(yLocal, yLocal, weightLocal, rowSize);
        
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
    TBuf<QuePosition::VECCALC> weightBuffer, tmpBuffer, workBuffer;
    GlobalTensor<float> xGm, weightGm, yGm;
    LocalTensor<float> weightLocal;
    uint32_t totalRows, rowSize;
    float eps, invRowSize;
};

extern "C" __global__ __aicore__ void rmsnorm_custom(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                                                      uint32_t totalRows, uint32_t rowSize, 
                                                      float eps) {
    KernelRMSNorm op;
    op.Init(x, weight, y, totalRows, rowSize, eps);
    op.Process();
}
