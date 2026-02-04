/*
 * AscendC MatMul Custom Operator - 修复版 Vector 实现
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelMatmul {
public:
    __aicore__ inline KernelMatmul() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                 int32_t M, int32_t N, int32_t K)
    {
        this->M = M;
        this->N = N;
        this->K = K;
        
        aGm.SetGlobalBuffer((__gm__ half*)a, M * K);
        bGm.SetGlobalBuffer((__gm__ half*)b, K * N);
        cGm.SetGlobalBuffer((__gm__ float*)c, M * N);
        
        // 计算对齐大小 (32字节对齐)
        // half: 16个元素 = 32字节; float: 8个元素 = 32字节
        alignedK = ((K + 15) / 16) * 16;  // half 对齐
        alignedN = ((N + 7) / 8) * 8;      // float 对齐
        
        pipe.InitBuffer(inQueueA, BUFFER_NUM, alignedK * sizeof(half));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, alignedK * sizeof(half));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, alignedN * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // C[i,j] = sum_k(A[i,k] * B[k,j])
        for (int32_t i = 0; i < M; i++) {
            ComputeRow(i);
        }
    }

private:
    __aicore__ inline void ComputeRow(int32_t row)
    {
        // 分配输出缓冲区
        LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();
        
        // 加载 A 的第 row 行
        LocalTensor<half> aRow = inQueueA.AllocTensor<half>();
        DataCopy(aRow, aGm[row * K], alignedK);
        inQueueA.EnQue(aRow);
        aRow = inQueueA.DeQue<half>();
        
        // 对每一列 j 计算
        for (int32_t j = 0; j < N; j++) {
            // 加载 B 的第 j 列 - B 是行主序 [K, N]
            LocalTensor<half> bCol = inQueueB.AllocTensor<half>();
            
            // 逻一加载 B[:, j]
            for (int32_t k = 0; k < K; k++) {
                half val = bGm.GetValue(k * N + j);
                bCol.SetValue(k, val);
            }
            // 填充对齐部分
            for (int32_t k = K; k < alignedK; k++) {
                bCol.SetValue(k, (half)0.0f);
            }
            
            inQueueB.EnQue(bCol);
            bCol = inQueueB.DeQue<half>();
            
            // 计算点积
            float sum = 0.0f;
            for (int32_t k = 0; k < K; k++) {
                float aVal = (float)aRow.GetValue(k);
                float bVal = (float)bCol.GetValue(k);
                sum += aVal * bVal;
            }
            cLocal.SetValue(j, sum);
            
            inQueueB.FreeTensor(bCol);
        }
        
        // 填充 C 的对齐部分
        for (int32_t j = N; j < alignedN; j++) {
            cLocal.SetValue(j, 0.0f);
        }
        
        inQueueA.FreeTensor(aRow);
        
        // 写出结果 - 只写实际数据，但要对齐
        outQueueC.EnQue(cLocal);
        cLocal = outQueueC.DeQue<float>();
        DataCopy(cGm[row * N], cLocal, alignedN);
        outQueueC.FreeTensor(cLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueC;
    GlobalTensor<half> aGm;
    GlobalTensor<half> bGm;
    GlobalTensor<float> cGm;
    int32_t M, N, K;
    int32_t alignedK, alignedN;
};

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelMatmul op;
    op.Init(a, b, c, tilingData.M, tilingData.N, tilingData.K);
    op.Process();
}
