/**
 * AscendC Fully Connected (MatMul) Kernel
 * y = x @ weight (without bias for simplicity)
 * 
 * 输入: x [M, K], weight [K, N]
 * 输出: y [M, N]
 * 
 * 这是一个基于 Vector 单元的朴素实现
 * 生产环境应使用 Cube 单元或 aclnnMatmul
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_SIZE = 256;

class KernelFC {
public:
    __aicore__ inline KernelFC() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                                 uint32_t M, uint32_t K, uint32_t N) {
        this->M = M;
        this->K = K;
        this->N = N;

        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        
        // 每个 block 处理若干行输出
        uint32_t rowsPerBlock = (M + blockNum - 1) / blockNum;
        uint32_t startRow = blockIdx * rowsPerBlock;
        uint32_t endRow = (startRow + rowsPerBlock > M) ? M : (startRow + rowsPerBlock);
        this->myStartRow = startRow;
        this->myEndRow = endRow;

        xGm.SetGlobalBuffer((__gm__ float*)x, M * K);
        wGm.SetGlobalBuffer((__gm__ float*)weight, K * N);
        yGm.SetGlobalBuffer((__gm__ float*)y, M * N);

        // 用于存储一行 x 和一列 weight
        pipe.InitBuffer(xRowBuf, TILE_SIZE * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t m = myStartRow; m < myEndRow; m++) {
            ProcessRow(m);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t row) {
        // 对于每个输出列
        for (uint32_t n = 0; n < N; n++) {
            float sum = 0.0f;
            
            // 点积: x[row, :] * weight[:, n]
            for (uint32_t k = 0; k < K; k++) {
                float xVal = xGm.GetValue(row * K + k);
                float wVal = wGm.GetValue(k * N + n);
                sum += xVal * wVal;
            }
            
            yGm.SetValue(row * N + n, sum);
        }
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> xRowBuf;
    GlobalTensor<float> xGm, wGm, yGm;
    uint32_t M, K, N;
    uint32_t myStartRow, myEndRow;
};

extern "C" __global__ __aicore__ void fc_kernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelFC op;
    op.Init(x, weight, y, tilingData.M, tilingData.K, tilingData.N);
    op.Process();
}
