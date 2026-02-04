/**
 * AscendC BatchNorm Inference Kernel
 * y = gamma * (x - mean) / sqrt(var + eps) + beta
 * 
 * 输入: x [N, C, H, W], mean [C], var [C], gamma [C], beta [C]
 * 输出: y [N, C, H, W]
 * 
 * 为简化实现，预计算: scale = gamma / sqrt(var + eps), shift = beta - mean * scale
 * 则: y = x * scale + shift
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_SIZE = 256;
constexpr float EPS = 1e-5f;

class KernelBatchNorm {
public:
    __aicore__ inline KernelBatchNorm() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta,
                                 GM_ADDR mean, GM_ADDR var, GM_ADDR y,
                                 uint32_t N, uint32_t C, uint32_t HW) {
        this->N = N;
        this->C = C;
        this->HW = HW;
        this->totalLength = N * C * HW;

        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        
        // 每个 block 处理若干个 channel
        uint32_t totalChannels = N * C;
        uint32_t channelsPerBlock = (totalChannels + blockNum - 1) / blockNum;
        uint32_t startChannel = blockIdx * channelsPerBlock;
        uint32_t endChannel = (startChannel + channelsPerBlock > totalChannels) 
                              ? totalChannels : (startChannel + channelsPerBlock);
        this->myStartChannel = startChannel;
        this->myEndChannel = endChannel;

        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        gammaGm.SetGlobalBuffer((__gm__ float*)gamma, C);
        betaGm.SetGlobalBuffer((__gm__ float*)beta, C);
        meanGm.SetGlobalBuffer((__gm__ float*)mean, C);
        varGm.SetGlobalBuffer((__gm__ float*)var, C);

        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_SIZE * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t ch = myStartChannel; ch < myEndChannel; ch++) {
            uint32_t c = ch % C;  // channel index for params
            
            // 读取该 channel 的参数
            float g = gammaGm.GetValue(c);
            float b = betaGm.GetValue(c);
            float m = meanGm.GetValue(c);
            float v = varGm.GetValue(c);
            
            // 预计算 scale 和 shift
            float invStd = 1.0f / sqrt(v + EPS);
            float scale = g * invStd;
            float shift = b - m * scale;
            
            // 处理该 channel 的所有 HW 元素
            uint32_t offset = ch * HW;
            ProcessChannel(offset, HW, scale, shift);
        }
    }

private:
    __aicore__ inline void ProcessChannel(uint32_t offset, uint32_t len, 
                                          float scale, float shift) {
        uint32_t loopCount = (len + TILE_SIZE - 1) / TILE_SIZE;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t tileLen = (i == loopCount - 1) ? (len - i * TILE_SIZE) : TILE_SIZE;
            uint32_t tileOffset = offset + i * TILE_SIZE;
            
            // CopyIn
            LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
            uint32_t alignedLen = ((tileLen + 7) / 8) * 8;
            DataCopy(xLocal, xGm[tileOffset], alignedLen);
            inQueue.EnQue(xLocal);
            
            // Compute: y = x * scale + shift
            xLocal = inQueue.DeQue<float>();
            LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
            
            Muls(yLocal, xLocal, scale, alignedLen);  // y = x * scale
            Adds(yLocal, yLocal, shift, alignedLen);  // y = y + shift
            
            outQueue.EnQue(yLocal);
            inQueue.FreeTensor(xLocal);
            
            // CopyOut
            yLocal = outQueue.DeQue<float>();
            DataCopy(yGm[tileOffset], yLocal, alignedLen);
            outQueue.FreeTensor(yLocal);
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<float> xGm, yGm, gammaGm, betaGm, meanGm, varGm;
    uint32_t N, C, HW, totalLength;
    uint32_t myStartChannel, myEndChannel;
};

extern "C" __global__ __aicore__ void batchnorm_kernel(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean, GM_ADDR var,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelBatchNorm op;
    op.Init(x, gamma, beta, mean, var, y, 
            tilingData.N, tilingData.C, tilingData.HW);
    op.Process();
}
