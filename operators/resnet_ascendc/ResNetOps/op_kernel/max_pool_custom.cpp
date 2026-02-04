#include "kernel_operator.h"
using namespace AscendC;

class KernelMaxPool {
public:
    __aicore__ inline KernelMaxPool() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t N, uint32_t C, uint32_t H, uint32_t W,
                                 uint32_t K, uint32_t stride, uint32_t pad) {
        this->N = N; this->C = C; this->H = H; this->W = W;
        this->K = K; this->stride = stride; this->pad = pad;
        this->outH = (H + 2 * pad - K) / stride + 1;
        this->outW = (W + 2 * pad - K) / stride + 1;

        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        uint32_t totalCh = N * C;
        uint32_t chPerBlock = (totalCh + blockNum - 1) / blockNum;
        this->myStart = blockIdx * chPerBlock;
        this->myEnd = (myStart + chPerBlock > totalCh) ? totalCh : (myStart + chPerBlock);

        xGm.SetGlobalBuffer((__gm__ float*)x, N * C * H * W);
        yGm.SetGlobalBuffer((__gm__ float*)y, N * C * outH * outW);
    }

    __aicore__ inline void Process() {
        for (uint32_t ch = myStart; ch < myEnd; ch++) {
            uint32_t inOff = ch * H * W;
            uint32_t outOff = ch * outH * outW;
            for (uint32_t oh = 0; oh < outH; oh++) {
                for (uint32_t ow = 0; ow < outW; ow++) {
                    float maxVal = -1e30f;
                    for (uint32_t kh = 0; kh < K; kh++) {
                        for (uint32_t kw = 0; kw < K; kw++) {
                            int32_t ih = (int32_t)(oh * stride + kh) - (int32_t)pad;
                            int32_t iw = (int32_t)(ow * stride + kw) - (int32_t)pad;
                            if (ih >= 0 && ih < (int32_t)H && iw >= 0 && iw < (int32_t)W) {
                                float val = xGm.GetValue(inOff + ih * W + iw);
                                if (val > maxVal) maxVal = val;
                            }
                        }
                    }
                    yGm.SetValue(outOff + oh * outW + ow, maxVal);
                }
            }
        }
    }

private:
    GlobalTensor<float> xGm, yGm;
    uint32_t N, C, H, W, K, stride, pad, outH, outW;
    uint32_t myStart, myEnd;
};

extern "C" __global__ __aicore__ void max_pool_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelMaxPool op;
    op.Init(x, y, tilingData.N, tilingData.C, tilingData.H, tilingData.W,
            tilingData.K, tilingData.stride, tilingData.pad);
    op.Process();
}
