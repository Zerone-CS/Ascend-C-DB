/**
 * MatMul - AscendC Vector 实现
 * C[M,N] = A[M,K] * B[K,N]  (float32)
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelMatmul {
public:
    __aicore__ inline KernelMatmul() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                 int32_t M, int32_t N, int32_t K) {
        this->M = M;
        this->N = N;
        this->K = K;
        
        aGm.SetGlobalBuffer((__gm__ float*)a, M * K);
        bGm.SetGlobalBuffer((__gm__ float*)b, K * N);
        cGm.SetGlobalBuffer((__gm__ float*)c, M * N);
        
        alignedK = ((K + 7) / 8) * 8;
        alignedN = ((N + 7) / 8) * 8;
        
        pipe.InitBuffer(inQueueA, BUFFER_NUM, alignedK * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, alignedK * sizeof(float));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, alignedN * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < M; i++) {
            ComputeRow(i);
        }
    }

private:
    __aicore__ inline void ComputeRow(int32_t row) {
        LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();
        
        LocalTensor<float> aRow = inQueueA.AllocTensor<float>();
        DataCopy(aRow, aGm[row * K], alignedK);
        inQueueA.EnQue(aRow);
        aRow = inQueueA.DeQue<float>();
        
        for (int32_t j = 0; j < N; j++) {
            LocalTensor<float> bCol = inQueueB.AllocTensor<float>();
            
            for (int32_t k = 0; k < K; k++) {
                bCol.SetValue(k, bGm.GetValue(k * N + j));
            }
            for (int32_t k = K; k < alignedK; k++) {
                bCol.SetValue(k, 0.0f);
            }
            
            inQueueB.EnQue(bCol);
            bCol = inQueueB.DeQue<float>();
            
            float sum = 0.0f;
            for (int32_t k = 0; k < K; k++) {
                sum += aRow.GetValue(k) * bCol.GetValue(k);
            }
            cLocal.SetValue(j, sum);
            
            inQueueB.FreeTensor(bCol);
        }
        
        for (int32_t j = N; j < alignedN; j++) {
            cLocal.SetValue(j, 0.0f);
        }
        
        inQueueA.FreeTensor(aRow);
        outQueueC.EnQue(cLocal);
        cLocal = outQueueC.DeQue<float>();
        DataCopy(cGm[row * N], cLocal, alignedN);
        outQueueC.FreeTensor(cLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA, inQueueB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueC;
    GlobalTensor<float> aGm, bGm, cGm;
    int32_t M, N, K, alignedK, alignedN;
};

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                                     GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelMatmul op;
    op.Init(a, b, c, tilingData.M, tilingData.N, tilingData.K);
    op.Process();
}
