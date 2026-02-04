#include "kernel_operator.h"
using namespace AscendC;

class KernelFc {
public:
    __aicore__ inline KernelFc() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                                 uint32_t M, uint32_t K, uint32_t N) {
        this->M = M; this->K = K; this->N = N;
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t rowsPerBlock = (M + blockNum - 1) / blockNum;
        this->myStartRow = blockIdx * rowsPerBlock;
        this->myEndRow = (myStartRow + rowsPerBlock > M) ? M : (myStartRow + rowsPerBlock);
        xGm.SetGlobalBuffer((__gm__ float*)x, M * K);
        wGm.SetGlobalBuffer((__gm__ float*)weight, K * N);
        yGm.SetGlobalBuffer((__gm__ float*)y, M * N);
    }

    __aicore__ inline void Process() {
        for (uint32_t m = myStartRow; m < myEndRow; m++) {
            for (uint32_t n = 0; n < N; n++) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < K; k++) {
                    sum += xGm.GetValue(m * K + k) * wGm.GetValue(k * N + n);
                }
                yGm.SetValue(m * N + n, sum);
            }
        }
    }

private:
    GlobalTensor<float> xGm, wGm, yGm;
    uint32_t M, K, N, myStartRow, myEndRow;
};

extern "C" __global__ __aicore__ void fc_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelFc op;
    op.Init(x, weight, y, tilingData.M, tilingData.K, tilingData.N);
    op.Process();
}
