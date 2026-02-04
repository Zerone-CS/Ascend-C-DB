/**
 * AscendC Global Average Pooling Kernel
 * 对每个 channel 的 H*W 元素求平均
 * 
 * 输入: x [N, C, H, W]
 * 输出: y [N, C, 1, 1] -> flattened to [N, C]
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_SIZE = 256;

class KernelGlobalAvgPool {
public:
    __aicore__ inline KernelGlobalAvgPool() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t N, uint32_t C, uint32_t HW) {
        this->N = N;
        this->C = C;
        this->HW = HW;

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

        xGm.SetGlobalBuffer((__gm__ float*)x, N * C * HW);
        yGm.SetGlobalBuffer((__gm__ float*)y, N * C);

        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(workBuf, 8 * sizeof(float));  // 用于 reduce 的工作缓冲区
    }

    __aicore__ inline void Process() {
        for (uint32_t ch = myStartChannel; ch < myEndChannel; ch++) {
            float sum = 0.0f;
            uint32_t offset = ch * HW;
            
            // 分块累加
            uint32_t loopCount = (HW + TILE_SIZE - 1) / TILE_SIZE;
            for (uint32_t i = 0; i < loopCount; i++) {
                uint32_t tileLen = (i == loopCount - 1) ? (HW - i * TILE_SIZE) : TILE_SIZE;
                sum += ReduceTileSum(offset + i * TILE_SIZE, tileLen);
            }
            
            // 求平均
            float avg = sum / (float)HW;
            yGm.SetValue(ch, avg);
        }
    }

private:
    __aicore__ inline float ReduceTileSum(uint32_t offset, uint32_t len) {
        LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        uint32_t alignedLen = ((len + 7) / 8) * 8;
        DataCopy(xLocal, xGm[offset], alignedLen);
        inQueue.EnQue(xLocal);
        
        xLocal = inQueue.DeQue<float>();
        
        // 手动累加（简化实现）
        float sum = 0.0f;
        for (uint32_t i = 0; i < len; i++) {
            sum += xLocal.GetValue(i);
        }
        
        inQueue.FreeTensor(xLocal);
        return sum;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TBuf<QuePosition::VECCALC> workBuf;
    GlobalTensor<float> xGm, yGm;
    uint32_t N, C, HW;
    uint32_t myStartChannel, myEndChannel;
};

extern "C" __global__ __aicore__ void avgpool_global_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelGlobalAvgPool op;
    op.Init(x, y, tilingData.N, tilingData.C, tilingData.HW);
    op.Process();
}
