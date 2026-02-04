#include "kernel_operator.h"
using namespace AscendC;

class KernelAvgPool {
public:
    __aicore__ inline KernelAvgPool() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t N, uint32_t C, uint32_t HW) {
        this->N = N; this->C = C; this->HW = HW;
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t totalCh = N * C;
        uint32_t chPerBlock = (totalCh + blockNum - 1) / blockNum;
        this->myStart = blockIdx * chPerBlock;
        this->myEnd = (myStart + chPerBlock > totalCh) ? totalCh : (myStart + chPerBlock);
        xGm.SetGlobalBuffer((__gm__ float*)x, N * C * HW);
        yGm.SetGlobalBuffer((__gm__ float*)y, N * C);
    }

    __aicore__ inline void Process() {
        for (uint32_t ch = myStart; ch < myEnd; ch++) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < HW; i++) {
                sum += xGm.GetValue(ch * HW + i);
            }
            yGm.SetValue(ch, sum / (float)HW);
        }
    }

private:
    GlobalTensor<float> xGm, yGm;
    uint32_t N, C, HW, myStart, myEnd;
};

extern "C" __global__ __aicore__ void avg_pool_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelAvgPool op;
    op.Init(x, y, tilingData.N, tilingData.C, tilingData.HW);
    op.Process();
}
